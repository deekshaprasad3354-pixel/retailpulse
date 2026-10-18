"""Stage 3 — Customer segmentation.

Two complementary views of the customer base:

1. **Rule-based RFM tiers.** Recency/Frequency/Monetary quintiles are combined
   into six named business segments (Champions, Loyal, ... Dormant). This is
   what a marketing team can act on directly.

2. **K-Means clusters.** A data-driven grouping on standardised RFM. We choose
   *k* by silhouette score rather than guessing, and profile each cluster so the
   numbers can be turned back into a business story.

Outputs:
    data/processed/customer_segments.csv
    data/processed/segment_profiles.csv
    reports/figures/segmentation_*.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config as cfg


# --------------------------------------------------------------------------
# Rule-based RFM tiers
# --------------------------------------------------------------------------
def assign_rfm_segment(row: pd.Series) -> str:
    """Map an R/F/M score triplet to one of six actionable segments.

    R, F and M each run 1-5 (5 = best), so the combined score runs 3-15.
    """
    r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]

    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    if r >= 3 and f >= 3 and m >= 3:
        return "Loyal Customers"
    if r >= 3 and f <= 2:
        return "New / Promising"
    if r >= 2 and f >= 2:
        return "Need Attention"
    if r >= 2:
        return "At Risk"
    return "Dormant"


# Order used in every chart and table so segments read consistently
SEGMENT_ORDER = [
    "Champions",
    "Loyal Customers",
    "New / Promising",
    "Need Attention",
    "At Risk",
    "Dormant",
]

# Recommended action per segment — turns the analysis into a decision
SEGMENT_PLAYBOOK = {
    "Champions": "Reward loyalty: early access, VIP tiers, referral incentives.",
    "Loyal Customers": "Upsell and cross-sell; enrol in a subscription or loyalty scheme.",
    "New / Promising": "Onboarding journey, second-purchase discount, product education.",
    "Need Attention": "Re-engagement email with a time-limited offer.",
    "At Risk": "Win-back campaign; ask why they left; offer a strong incentive.",
    "Dormant": "Low-cost reactivation only, or suppress to protect margin.",
}


def segment_customers(rfm: pd.DataFrame) -> pd.DataFrame:
    out = rfm.copy()
    out["Segment"] = out.apply(assign_rfm_segment, axis=1)
    out["Segment"] = pd.Categorical(out["Segment"], categories=SEGMENT_ORDER, ordered=True)
    out["RecommendedAction"] = out["Segment"].map(SEGMENT_PLAYBOOK)
    return out


def segment_profiles(seg: pd.DataFrame) -> pd.DataFrame:
    prof = seg.groupby("Segment", observed=True).agg(
        Customers=("CustomerID", "count"),
        AvgRecency=("Recency", "mean"),
        AvgFrequency=("Frequency", "mean"),
        AvgMonetary=("Monetary", "mean"),
        TotalMonetary=("Monetary", "sum"),
        AvgRFMScore=("RFM_Score", "mean"),
    ).round(2).reset_index()
    prof["CustomerSharePct"] = (prof["Customers"] / prof["Customers"].sum() * 100).round(1)
    prof["RevenueSharePct"] = (prof["TotalMonetary"] / prof["TotalMonetary"].sum() * 100).round(1)
    prof["RecommendedAction"] = prof["Segment"].map(SEGMENT_PLAYBOOK)
    return prof.sort_values("TotalMonetary", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# K-Means clustering
# --------------------------------------------------------------------------
def _standardise(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd == 0] = 1.0
    return (x - mu) / sd


def _silhouette(x: np.ndarray, labels: np.ndarray, sample: int = 2000, seed: int = cfg.RANDOM_STATE) -> float:
    """Mean silhouette coefficient, computed on a random sample for speed.

    We implement this directly so the metric is available even without
    scikit-learn; the maths is the textbook definition.
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    idx = rng.choice(n, size=min(sample, n), replace=False)
    xs, ls = x[idx], labels[idx]
    uniq = np.unique(ls)
    if len(uniq) < 2:
        return -1.0

    # pairwise distances within the sample
    d = np.sqrt(((xs[:, None, :] - xs[None, :, :]) ** 2).sum(axis=2))
    scores = np.zeros(len(xs))
    for i in range(len(xs)):
        same = ls == ls[i]
        same[i] = False
        a = d[i, same].mean() if same.any() else 0.0
        b = min(
            d[i, ls == c].mean()
            for c in uniq
            if c != ls[i] and (ls == c).any()
        )
        scores[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(scores.mean())


def _calinski_harabasz(x: np.ndarray, labels: np.ndarray) -> float:
    """Between-cluster dispersion / within-cluster dispersion. Higher is better."""
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return float("nan")
    overall = x.mean(axis=0)
    n = len(x)
    between = sum(len(x[labels == c]) * ((x[labels == c].mean(axis=0) - overall) ** 2).sum()
                  for c in uniq)
    within = sum(((x[labels == c] - x[labels == c].mean(axis=0)) ** 2).sum() for c in uniq)
    if within == 0:
        return float("nan")
    return float((between / (len(uniq) - 1)) / (within / (n - len(uniq))))


def _davies_bouldin(x: np.ndarray, labels: np.ndarray) -> float:
    """Average worst-case cluster separation ratio. Lower is better."""
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return float("nan")
    centres = {c: x[labels == c].mean(axis=0) for c in uniq}
    spread = {c: float(np.sqrt(((x[labels == c] - centres[c]) ** 2).sum(axis=1).mean()))
              for c in uniq}
    scores = []
    for a in uniq:
        worst = 0.0
        for b in uniq:
            if a == b:
                continue
            dist = float(np.sqrt(((centres[a] - centres[b]) ** 2).sum()))
            if dist > 0:
                worst = max(worst, (spread[a] + spread[b]) / dist)
        scores.append(worst)
    return float(np.mean(scores))


def _make_kmeans(k: int):
    """Prefer scikit-learn's K-Means; fall back to the built-in NumPy version."""
    from src.optional_deps import SKLEARN

    if SKLEARN.ok:
        from sklearn.cluster import KMeans

        return KMeans(n_clusters=k, random_state=cfg.RANDOM_STATE, n_init=10)

    from src.models.numpy_ml import KMeans

    return KMeans(n_clusters=k, n_init=10, random_state=cfg.RANDOM_STATE)


def kmeans_cluster(
    rfm: pd.DataFrame,
    k: int = cfg.SEGMENT_K,
    features: tuple[str, ...] = ("Recency", "Frequency", "Monetary"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster customers on standardised RFM and profile the clusters."""
    x_raw = rfm[list(features)].to_numpy(dtype=float)
    x = _standardise(x_raw)

    model = _make_kmeans(k)
    labels = model.fit_predict(x)

    out = rfm.copy()
    out["Cluster"] = labels

    # Rank clusters by monetary value so "Cluster 0" is always the best one
    rank = (
        out.groupby("Cluster")["Monetary"].mean().sort_values(ascending=False).index.tolist()
    )
    remap = {old: new for new, old in enumerate(rank)}
    out["Cluster"] = out["Cluster"].map(remap).astype(int)
    out["ClusterName"] = out["Cluster"].map(lambda c: f"Cluster {c + 1}")

    metrics = pd.DataFrame([{
        "k": k,
        "inertia": round(float(model.inertia_), 2),
        "silhouette": round(_silhouette(x, labels), 4),
        "calinski_harabasz": round(_calinski_harabasz(x, labels), 2),
        "davies_bouldin": round(_davies_bouldin(x, labels), 4),
    }])

    prof = out.groupby(["Cluster", "ClusterName"], as_index=False).agg(
        Customers=("CustomerID", "count"),
        AvgRecency=("Recency", "mean"),
        AvgFrequency=("Frequency", "mean"),
        AvgMonetary=("Monetary", "mean"),
    ).round(2)
    prof["CustomerSharePct"] = (prof["Customers"] / prof["Customers"].sum() * 100).round(1)
    prof["RevenueSharePct"] = (
        out.groupby("Cluster", as_index=False)["Monetary"].sum()
        .assign(pct=lambda d: d["Monetary"] / d["Monetary"].sum() * 100)
        .round(1)["pct"]
    )
    # Attach the cluster-level profile to the single metrics row as a
    # JSON-serialisable summary, so downstream consumers get both at once.
    comp = prof.drop(columns=["ClusterName"])
    for _, r in comp.iterrows():
        c = int(r["Cluster"])
        metrics[f"cluster{c + 1}_customers"] = int(r["Customers"])
        metrics[f"cluster{c + 1}_avg_monetary"] = float(r["AvgMonetary"])
        metrics[f"cluster{c + 1}_revenue_share_pct"] = float(r["RevenueSharePct"])
    return out, metrics


def choose_k(rfm: pd.DataFrame, ks: range = range(2, 9)) -> pd.DataFrame:
    """Score several values of k so the choice is evidence-based."""
    x = _standardise(rfm[["Recency", "Frequency", "Monetary"]].to_numpy(dtype=float))
    rows = []
    for k in ks:
        model = _make_kmeans(k)
        labels = model.fit_predict(x)
        rows.append({
            "k": k,
            "inertia": round(float(model.inertia_), 2),
            "silhouette": round(_silhouette(x, labels), 4),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def _make_svg_figures(seg: pd.DataFrame, prof: pd.DataFrame, kscan: pd.DataFrame) -> None:
    """Dependency-free SVG fallback for the segmentation charts."""
    from src.viz import bar_chart, save, scatter_chart

    order = prof["Segment"].tolist()
    save(bar_chart(order, prof["CustomerSharePct"].tolist(),
                   "RFM segments — share of customers (%)", horizontal=True,
                   value_suffix="%"),
         cfg.FIGURES / "segmentation_rfm_customers.svg")
    save(bar_chart(order, prof["RevenueSharePct"].tolist(),
                   "RFM segments — share of revenue (%)", horizontal=True,
                   value_suffix="%"),
         cfg.FIGURES / "segmentation_rfm_revenue.svg")

    save(bar_chart(kscan["k"].astype(str).tolist(), kscan["silhouette"].tolist(),
                   "Choosing k — silhouette score by number of clusters",
                   x_label="number of clusters k"),
         cfg.FIGURES / "segmentation_k_selection.svg")

    sample = seg.sample(min(len(seg), 2000), random_state=cfg.RANDOM_STATE)
    save(scatter_chart(
        list(zip(sample["Recency"], sample["Monetary"], sample["Segment"].astype(str))),
        "Recency vs Monetary by RFM segment",
        x_label="days since last purchase", y_label="revenue (GBP)"),
        cfg.FIGURES / "segmentation_clusters_rfm.svg")
    save(scatter_chart(
        list(zip(sample["Frequency"], sample["Monetary"], sample["ClusterName"].astype(str))),
        "Frequency vs Monetary by K-Means cluster",
        x_label="number of orders", y_label="revenue (GBP)"),
        cfg.FIGURES / "segmentation_clusters_kmeans.svg")


def make_figures(seg: pd.DataFrame, prof: pd.DataFrame, kscan: pd.DataFrame) -> None:
    """Render segmentation charts — PNG via matplotlib, else dependency-free SVG."""
    from src.optional_deps import MATPLOTLIB

    if not MATPLOTLIB.ok:
        _make_svg_figures(seg, prof, kscan)
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("viridis", len(prof))

    # 1. customers vs revenue share per segment
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    order = prof["Segment"].tolist()
    sns.barplot(data=prof, y="Segment", x="CustomerSharePct", order=order, ax=axes[0], palette=palette)
    axes[0].set_title("Share of customers by segment (%)", fontsize=12, weight="bold")
    axes[0].set_xlabel("% of customers"); axes[0].set_ylabel("")
    sns.barplot(data=prof, y="Segment", x="RevenueSharePct", order=order, ax=axes[1], palette=palette)
    axes[1].set_title("Share of revenue by segment (%)", fontsize=12, weight="bold")
    axes[1].set_xlabel("% of revenue"); axes[1].set_ylabel("")
    for ax in axes:
        for c in ax.containers:
            ax.bar_label(c, fmt="%.1f", padding=2, fontsize=8)
    fig.suptitle("RFM segmentation — the 80/20 in one picture", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(cfg.FIGURES / "segmentation_rfm_overview.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # 2. cluster scan
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(kscan["k"], kscan["inertia"], "o-", color="#2563eb", label="Inertia (elbow)")
    ax1.set_xlabel("number of clusters k"); ax1.set_ylabel("Inertia", color="#2563eb")
    ax2 = ax1.twinx()
    ax2.plot(kscan["k"], kscan["silhouette"], "s--", color="#dc2626", label="Silhouette")
    ax2.set_ylabel("Silhouette score", color="#dc2626")
    ax1.set_title("Choosing k — elbow + silhouette", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(cfg.FIGURES / "segmentation_k_selection.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # 3. cluster scatter
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.scatterplot(data=seg, x="Recency", y="Monetary", hue="Segment", alpha=0.55,
                    s=18, ax=axes[0], palette="viridis")
    axes[0].set_title("Recency vs Monetary by RFM segment", fontsize=12, weight="bold")
    axes[0].legend(fontsize=7, loc="upper right")
    sns.scatterplot(data=seg, x="Frequency", y="Monetary", hue="ClusterName", alpha=0.55,
                    s=18, ax=axes[1], palette="magma")
    axes[1].set_title("Frequency vs Monetary by K-Means cluster", fontsize=12, weight="bold")
    axes[1].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(cfg.FIGURES / "segmentation_clusters.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("=" * 78)
    print("RetailPulse :: Stage 3 - customer segmentation")
    print("=" * 78)

    rfm = pd.read_csv(cfg.PROCESSED / "customer_rfm.csv")
    print(f"[segment] customers: {len(rfm):,}")

    seg = segment_customers(rfm)
    prof = segment_profiles(seg)
    print("\n--- rule-based RFM segments ---")
    print(prof[["Segment", "Customers", "CustomerSharePct", "RevenueSharePct",
                "AvgRecency", "AvgFrequency", "AvgMonetary"]].to_string(index=False))

    kscan = choose_k(rfm)
    best_k = int(kscan.loc[kscan["silhouette"].idxmax(), "k"])
    print("\n--- k selection ---")
    print(kscan.to_string(index=False))
    print(f"[segment] best k by silhouette = {best_k}")

    seg, kmetrics = kmeans_cluster(rfm, k=cfg.SEGMENT_K)
    # Bring the rule-based tier alongside the data-driven cluster label so every
    # customer carries both views.
    seg = seg.merge(
        segment_customers(rfm)[["CustomerID", "Segment", "RecommendedAction"]],
        on="CustomerID", how="left",
    )
    cluster_prof = seg.groupby(["Cluster", "ClusterName"], as_index=False).agg(
        Customers=("CustomerID", "count"),
        AvgRecency=("Recency", "mean"),
        AvgFrequency=("Frequency", "mean"),
        AvgMonetary=("Monetary", "mean"),
        AvgRFM=("RFM_Score", "mean"),
    ).round(2)
    print("\n--- K-Means clusters (ranked by value) ---")
    print(cluster_prof.to_string(index=False))
    print("\n--- cluster quality ---")
    print(kmetrics.to_string(index=False))

    seg.to_csv(cfg.PROCESSED / "customer_segments.csv", index=False)
    prof.to_csv(cfg.PROCESSED / "segment_profiles.csv", index=False)
    cluster_prof.to_csv(cfg.PROCESSED / "cluster_profiles.csv", index=False)
    kscan.to_csv(cfg.PROCESSED / "kmeans_k_scan.csv", index=False)
    kmetrics.to_csv(cfg.PROCESSED / "kmeans_metrics.csv", index=False)

    make_figures(seg, prof, kscan)
    print(f"\n[segment] wrote outputs -> {cfg.PROCESSED}")


if __name__ == "__main__":
    main()
