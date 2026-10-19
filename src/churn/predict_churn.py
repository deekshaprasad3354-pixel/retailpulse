"""Stage 4 — Churn prediction.

The setup is a standard *observation window* / *prediction window* design, which
is what makes churn modelling honest:

    |<---- features come from here ---->|<---- label comes from here ---->|
    start of history              cutoff T                       end of data

A customer is labelled **churned** if they place no order in the 90 days after
T, and **active** otherwise. Every feature is computed strictly from data
*before* T, so the model never sees the future it is asked to predict.

Models compared:
    * Logistic Regression      - interpretable baseline, gives signed weights
    * Random Forest            - nonlinear, robust, gives importances
    * Gradient Boosting        - usually the strongest on tabular data
    * Majority-class baseline  - the "always predict active" sanity check

Outputs:
    data/processed/customer_churn.csv
    data/processed/churn_model_comparison.csv
    data/processed/churn_feature_importance.csv
    models/churn_model.joblib
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config as cfg
from src.evaluation.metrics import precision_recall_f1, roc_auc

FEATURES = [
    "Recency",
    "Frequency",
    "Monetary",
    "Units",
    "UniqueProducts",
    "AvgOrderValue",
    "AvgUnitsPerOrder",
    "TenureDays",
    "PurchaseRate",
    "ActiveMonths",
    "RevenueTrend",
    "ReturnRate",
]


# --------------------------------------------------------------------------
# Feature / label construction
# --------------------------------------------------------------------------
def build_churn_dataset(df: pd.DataFrame, cutoff: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Build one row per customer with pre-cutoff features and a post-cutoff label."""
    end = df["Date"].max()
    cutoff = cutoff or (end - pd.Timedelta(days=cfg.CHURN_HORIZON_DAYS))

    hist = df[df["Date"] < cutoff]
    future = df[df["Date"] >= cutoff]
    print(f"[churn] cutoff {cutoff:%Y-%m-%d} | history rows {len(hist):,} | future rows {len(future):,}")

    snap = cutoff
    feat = hist.groupby("CustomerID").agg(
        Recency=("Date", lambda s: (snap - s.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("Revenue", "sum"),
        Units=("Quantity", "sum"),
        UniqueProducts=("StockCode", "nunique"),
        FirstPurchase=("Date", "min"),
        LastPurchase=("Date", "max"),
        LineItems=("Revenue", "size"),
        ActiveMonths=("Date", lambda s: s.dt.to_period("M").nunique()),
    ).reset_index()

    feat["AvgOrderValue"] = (feat["Monetary"] / feat["Frequency"]).round(2)
    feat["AvgUnitsPerOrder"] = (feat["Units"] / feat["Frequency"]).round(2)
    feat["TenureDays"] = (feat["LastPurchase"] - feat["FirstPurchase"]).dt.days.clip(lower=1)
    feat["PurchaseRate"] = (feat["Frequency"] / feat["TenureDays"] * 30).round(3)

    # Momentum: revenue in the last 90 days vs the 90 days before that.
    # Falling momentum is one of the strongest churn signals in retail.
    mid = snap - pd.Timedelta(days=90)
    recent_rev = hist[hist["Date"] >= mid].groupby("CustomerID")["Revenue"].sum()
    prior_rev = hist[hist["Date"] < mid].groupby("CustomerID")["Revenue"].sum()
    feat["RecentRevenue"] = feat["CustomerID"].map(recent_rev).fillna(0.0)
    feat["PriorRevenue"] = feat["CustomerID"].map(prior_rev).fillna(0.0)
    feat["RevenueTrend"] = np.where(
        feat["PriorRevenue"] > 0,
        (feat["RecentRevenue"] - feat["PriorRevenue"]) / feat["PriorRevenue"],
        0.0,
    ).clip(-1, 5).round(4)
    feat["ReturnRate"] = 0.0  # cancellations were removed in cleaning; kept for schema stability

    # Label: did they come back inside the prediction window?
    returned = set(future["CustomerID"].unique())
    feat["Churn"] = (~feat["CustomerID"].isin(returned)).astype(int)
    feat["CutoffDate"] = cutoff
    feat["ChurnLabel"] = np.where(feat["Churn"] == 1, "Churned", "Active")

    feat["Monetary"] = feat["Monetary"].round(2)
    return feat, cutoff


def stratified_split(y: np.ndarray, test_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Train/test indices that preserve the churn rate in both halves."""
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_test = max(int(round(len(idx) * test_frac)), 1)
        test_idx.append(idx[:n_test])
        train_idx.append(idx[n_test:])
    return np.sort(np.concatenate(train_idx)), np.sort(np.concatenate(test_idx))


# --------------------------------------------------------------------------
# Modelling
# --------------------------------------------------------------------------
class _Scaler:
    """Standard-scaler shim so the code path is identical with or without sklearn."""

    def fit(self, x):
        self.mu_ = x.mean(axis=0)
        self.sd_ = np.where(x.std(axis=0) == 0, 1.0, x.std(axis=0))
        return self

    def transform(self, x):
        return (x - self.mu_) / self.sd_

    def fit_transform(self, x):
        return self.fit(x).transform(x)


def _build_models(x_tr_s, x_te_s, x_tr, x_te):
    """Instantiate the three candidate classifiers plus a scaler.

    Uses scikit-learn when it is installed and the built-in NumPy
    implementations otherwise, so results are produced either way.
    """
    from src.optional_deps import SKLEARN

    if SKLEARN.ok:
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler().fit(x_tr)
        models = {
            "Logistic Regression": (
                LogisticRegression(max_iter=2000, class_weight="balanced",
                                   random_state=cfg.RANDOM_STATE),
                scaler.transform(x_tr), scaler.transform(x_te),
            ),
            "Random Forest": (
                RandomForestClassifier(n_estimators=400, min_samples_leaf=3, max_depth=14,
                                       class_weight="balanced_subsample", n_jobs=-1,
                                       random_state=cfg.RANDOM_STATE),
                x_tr, x_te,
            ),
            "Gradient Boosting": (
                GradientBoostingClassifier(n_estimators=250, learning_rate=0.06, max_depth=3,
                                           subsample=0.9, random_state=cfg.RANDOM_STATE),
                x_tr, x_te,
            ),
        }
    else:
        from src.models.numpy_ml import GradientBoosting, LogisticRegression, RandomForest

        scaler = _Scaler().fit(x_tr)
        models = {
            "Logistic Regression": (
                LogisticRegression(lr=0.5, epochs=1500, l2=1e-3, class_weight="balanced",
                                   random_state=cfg.RANDOM_STATE),
                scaler.transform(x_tr), scaler.transform(x_te),
            ),
            "Random Forest": (
                RandomForest(n_estimators=60, max_depth=12, min_samples_leaf=3,
                             random_state=cfg.RANDOM_STATE),
                x_tr, x_te,
            ),
            "Gradient Boosting": (
                GradientBoosting(n_estimators=80, learning_rate=0.10, max_depth=3,
                                 min_samples_leaf=5, subsample=0.9,
                                 random_state=cfg.RANDOM_STATE),
                x_tr, x_te,
            ),
        }
    return models, scaler


def train_models(feat: pd.DataFrame, test_frac: float = cfg.CHURN_TEST_FRACTION) -> tuple[dict, pd.DataFrame]:
    x = feat[FEATURES].to_numpy(dtype=float)
    y = feat["Churn"].to_numpy(dtype=int)
    tr, te = stratified_split(y, test_frac, cfg.RANDOM_STATE)
    x_tr, x_te, y_tr, y_te = x[tr], x[te], y[tr], y[te]

    models, scaler = _build_models(None, None, x_tr, x_te)

    pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    print(f"[churn] train {len(tr):,} | test {len(te):,} | train churn rate {y_tr.mean():.1%}")

    results = []
    fitted: dict = {}

    # Baseline: always predict the majority class
    majority = int(pd.Series(y_tr).mode().iloc[0])
    base_pred = np.full_like(y_te, majority)
    base_m = precision_recall_f1(y_te, base_pred)
    results.append({
        "model": "Baseline (majority class)",
        "auc": 0.5, "ks": 0.0, "lift_top10": 1.0, **base_m,
    })

    for name, (model, xtr, xte) in models.items():
        model.fit(xtr, y_tr)
        proba = model.predict_proba(xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        m = precision_recall_f1(y_te, pred)
        from src.evaluation.metrics import ks_statistic, lift_at_k

        results.append({
            "model": name,
            "auc": round(roc_auc(y_te, proba), 4),
            "ks": round(ks_statistic(y_te, proba), 4),
            "lift_top10": round(lift_at_k(y_te, proba, 0.10), 3),
            **m,
        })
        fitted[name] = {"model": model, "scaler": scaler if name == "Logistic Regression" else None}

    comparison = pd.DataFrame(results).sort_values("auc", ascending=False).reset_index(drop=True)
    return {"fitted": fitted, "x": x, "y": y, "tr": tr, "te": te,
            "y_te": y_te, "scaler": scaler, "pos_weight": pos_weight}, comparison


def _importances(model, n_features: int) -> np.ndarray:
    """Feature importances from either a scikit-learn or a built-in model."""
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)
    if hasattr(model, "feature_importance"):
        return np.asarray(model.feature_importance(n_features), dtype=float)
    return np.zeros(n_features)


def feature_importance(feat: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    n_features = len(FEATURES)
    rows = []

    rf = bundle["fitted"]["Random Forest"]["model"]
    rows.append(pd.DataFrame({"feature": FEATURES, "importance": _importances(rf, n_features),
                              "model": "Random Forest"}))

    gb = bundle["fitted"]["Gradient Boosting"]["model"]
    rows.append(pd.DataFrame({"feature": FEATURES, "importance": _importances(gb, n_features),
                              "model": "Gradient Boosting"}))

    lr = bundle["fitted"]["Logistic Regression"]["model"]
    coef = np.asarray(getattr(lr, "coef_", np.zeros(n_features)), dtype=float).ravel()
    rows.append(pd.DataFrame({
        "feature": FEATURES,
        "importance": np.abs(coef[:n_features]),
        "model": "Logistic Regression (|coef|)",
    }))

    out = pd.concat(rows, ignore_index=True)
    out["importance"] = out["importance"].round(5)
    # normalise within model so bars are comparable
    out["importance_pct"] = (
        out.groupby("model")["importance"].transform(lambda s: s / s.sum() * 100).round(2)
    )
    return out.sort_values(["model", "importance"], ascending=[True, False]).reset_index(drop=True)


def score_all_customers(feat: pd.DataFrame, bundle: dict,
                        best_name: str | None = None) -> pd.DataFrame:
    """Score the whole base with the best model and bucket into risk tiers."""
    comparison_models = bundle["fitted"]
    best_name = best_name if best_name in comparison_models else next(iter(comparison_models))
    entry = comparison_models[best_name]
    x = feat[FEATURES].to_numpy(dtype=float)
    if entry["scaler"] is not None:
        x = entry["scaler"].transform(x)
    proba = entry["model"].predict_proba(x)[:, 1]

    out = feat.copy()
    out["ChurnProbability"] = proba.round(4)
    out["ChurnPrediction"] = (out["ChurnProbability"] >= 0.5).astype(int)
    out["RiskTier"] = pd.cut(
        out["ChurnProbability"],
        bins=[-0.001, 0.35, 0.65, 1.001],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    )
    out["ScoredBy"] = best_name
    return out


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def _roc_points(y_true: np.ndarray, scores: np.ndarray) -> tuple[list, list]:
    """ROC curve points without scikit-learn (sort by score, sweep the threshold)."""
    order = np.argsort(-scores)
    y = y_true[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    n_pos = max(int((y_true == 1).sum()), 1)
    n_neg = max(int((y_true == 0).sum()), 1)
    tpr = np.concatenate([[0.0], tp / n_pos])
    fpr = np.concatenate([[0.0], fp / n_neg])
    return fpr.tolist(), tpr.tolist()


def _make_svg_figures(scored: pd.DataFrame, comparison: pd.DataFrame, imp: pd.DataFrame,
                      bundle: dict) -> None:
    """Dependency-free SVG fallback for the churn charts."""
    from src.viz import bar_chart, line_chart, save

    y_te = bundle["y_te"]
    curves = {}
    for name in ["Logistic Regression", "Random Forest", "Gradient Boosting"]:
        entry = bundle["fitted"][name]
        x_te = bundle["x"][bundle["te"]]
        if entry["scaler"] is not None:
            x_te = entry["scaler"].transform(x_te)
        p = entry["model"].predict_proba(x_te)[:, 1]
        fpr, tpr = _roc_points(y_te, p)
        curves[f"{name} (AUC {roc_auc(y_te, p):.3f})"] = (fpr, tpr)
    curves["Random (AUC 0.500)"] = ([0, 1], [0, 1])
    save(line_chart(curves, "ROC curves — churn classifier", x_label="false positive rate",
                    y_label="true positive rate"),
         cfg.FIGURES / "churn_roc.svg")

    top = imp[imp["model"] == "Gradient Boosting"].nlargest(10, "importance").sort_values("importance")
    save(bar_chart(top["feature"].tolist(), top["importance"].tolist(),
                   "Top 10 churn drivers (Gradient Boosting)", horizontal=True),
         cfg.FIGURES / "churn_feature_importance.svg")

    order = ["Low Risk", "Medium Risk", "High Risk"]
    counts = scored["RiskTier"].value_counts().reindex(order).fillna(0)
    save(bar_chart(order, counts.tolist(), "Customers by churn risk tier",
                   colors=["#16A34A", "#F59E0B", "#DC2626"]),
         cfg.FIGURES / "churn_risk_tiers.svg")

    hist, edges = np.histogram(scored["ChurnProbability"], bins=30, range=(0, 1))
    centres = ((edges[:-1] + edges[1:]) / 2).round(3)
    save(bar_chart([f"{c:.2f}" for c in centres], hist.tolist(),
                   "Predicted churn probability distribution", x_label="churn probability"),
         cfg.FIGURES / "churn_probability_distribution.svg")


def make_figures(scored: pd.DataFrame, comparison: pd.DataFrame, imp: pd.DataFrame,
                 bundle: dict) -> None:
    from src.optional_deps import MATPLOTLIB

    if not MATPLOTLIB.ok:
        _make_svg_figures(scored, comparison, imp, bundle)
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import roc_curve

    sns.set_theme(style="whitegrid")

    # 1. ROC curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    y_te = bundle["y_te"]
    for name in ["Logistic Regression", "Random Forest", "Gradient Boosting"]:
        entry = bundle["fitted"][name]
        x_te = bundle["x"][bundle["te"]]
        if entry["scaler"] is not None:
            x_te = entry["scaler"].transform(x_te)
        p = entry["model"].predict_proba(x_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, p)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC {roc_auc(y_te, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC 0.500)")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves — churn classifier", fontsize=12, weight="bold")
    ax.legend(fontsize=8, loc="lower right")

    # 2. feature importance
    ax = axes[1]
    top = imp[imp["model"] == "Gradient Boosting"].nlargest(10, "importance")
    sns.barplot(data=top, y="feature", x="importance", ax=ax, color="#2563eb")
    ax.set_title("Top 10 churn drivers (Gradient Boosting)", fontsize=12, weight="bold")
    ax.set_xlabel("importance"); ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(cfg.FIGURES / "churn_roc_and_importance.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # 3. risk tiers + probability distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    order = ["Low Risk", "Medium Risk", "High Risk"]
    counts = scored["RiskTier"].value_counts().reindex(order).fillna(0)
    sns.barplot(x=counts.index, y=counts.values, ax=axes[0],
                palette=["#16a34a", "#f59e0b", "#dc2626"])
    axes[0].set_title("Customers by churn risk tier", fontsize=12, weight="bold")
    axes[0].set_ylabel("customers"); axes[0].set_xlabel("")
    for c in axes[0].containers:
        axes[0].bar_label(c, fmt="%.0f", padding=2, fontsize=9)
    sns.histplot(data=scored, x="ChurnProbability", hue="ChurnLabel", bins=40,
                 ax=axes[1], palette={"Churned": "#dc2626", "Active": "#2563eb"}, alpha=0.75)
    axes[1].set_title("Predicted churn probability vs actual outcome", fontsize=12, weight="bold")
    axes[1].set_xlabel("predicted churn probability")
    fig.tight_layout()
    fig.savefig(cfg.FIGURES / "churn_risk_tiers.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("=" * 78)
    print("RetailPulse :: Stage 4 - churn prediction")
    print("=" * 78)

    clean = cfg.read_cache(cfg.CLEAN_CACHE)
    clean["Date"] = pd.to_datetime(clean["Date"])

    feat, cutoff = build_churn_dataset(clean)
    print(f"[churn] customers scored: {len(feat):,} | churn rate {feat['Churn'].mean():.1%}")

    bundle, comparison = train_models(feat)
    print("\n--- model comparison ---")
    print(comparison.to_string(index=False))

    imp = feature_importance(feat, bundle)
    print("\n--- top churn drivers (Gradient Boosting) ---")
    print(imp[imp["model"] == "Gradient Boosting"].head(8)[["feature", "importance_pct"]].to_string(index=False))

    scored = score_all_customers(feat, bundle, best_name=str(comparison.iloc[0]["model"]))
    tiers = scored["RiskTier"].value_counts().reindex(["Low Risk", "Medium Risk", "High Risk"]).fillna(0)
    print("\n--- risk tiers ---")
    for k, v in tiers.items():
        print(f"  {k:<12s} {int(v):>6,}  ({v / len(scored):.1%})")

    seg_file = cfg.PROCESSED / "customer_segments.csv"
    if seg_file.exists():
        try:
            seg_df = pd.read_csv(seg_file, usecols=["CustomerID", "Segment"])
            scored = scored.merge(seg_df, on="CustomerID", how="left")
        except Exception:
            pass

    scored.to_csv(cfg.PROCESSED / "customer_churn.csv", index=False)
    comparison.to_csv(cfg.PROCESSED / "churn_model_comparison.csv", index=False)
    imp.to_csv(cfg.PROCESSED / "churn_feature_importance.csv", index=False)

    make_figures(scored, comparison, imp, bundle)

    try:
        from src.models.persistence import save_model

        best = comparison.iloc[0]["model"]
        written = save_model(
            {"model": bundle["fitted"][best]["model"], "features": FEATURES,
             "scaler": bundle["fitted"][best]["scaler"], "cutoff": str(cutoff),
             "metrics": comparison.iloc[0].to_dict()},
            cfg.MODELS / "churn_model.joblib",
        )
        print(f"\n[churn] saved model ({best}) -> {written.name}")
    except Exception as exc:  # pragma: no cover
        print(f"[churn] could not persist model: {exc}")

    print(f"[churn] wrote outputs -> {cfg.PROCESSED}")


if __name__ == "__main__":
    main()
