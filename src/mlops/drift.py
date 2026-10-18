"""Stage 7 — Data drift detection (MLOps).

A model is only as good as the data it was trained on. When the world moves, a
model that was accurate in training silently degrades in production. This stage
compares an **early reference window** against the **current window** and flags
features whose distribution has shifted.

Two standard tests, both implemented without SciPy:

    PSI  (Population Stability Index)
        PSI < 0.10  -> stable
        0.10 - 0.25 -> moderate shift, investigate
        PSI > 0.25  -> major shift, retrain

    KS   (Kolmogorov-Smirnov two-sample test)
        The largest gap between the two empirical CDFs. We also report an
        asymptotic p-value using the Kolmogorov distribution.

Outputs:
    data/processed/drift_report.csv
    reports/figures/drift_*.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config as cfg

PSI_BINS = 10
PSI_MODERATE = 0.10
PSI_MAJOR = 0.25


def psi(expected, actual, bins: int = PSI_BINS, eps: float = 1e-6) -> float:
    """Population Stability Index between two samples of one feature.

    Quantile bins come from the *expected* (reference) sample, then both samples
    are bucketed and compared as proportions.
    """
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    e = e[np.isfinite(e)]
    a = a[np.isfinite(a)]
    if len(e) == 0 or len(a) == 0:
        return float("nan")

    # Bin edges from reference quantiles; fall back to equal-width if degenerate
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        lo, hi = float(min(e.min(), a.min())), float(max(e.max(), a.max()))
        if lo == hi:
            return 0.0
        edges = np.linspace(lo, hi, bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf

    e_pct = np.histogram(e, bins=edges)[0] / len(e)
    a_pct = np.histogram(a, bins=edges)[0] / len(a)
    e_pct = np.clip(e_pct, eps, None)
    a_pct = np.clip(a_pct, eps, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def ks_drift(expected, actual) -> tuple[float, float]:
    """Two-sample KS statistic D and its asymptotic p-value."""
    e = np.sort(np.asarray(expected, dtype=float))
    a = np.sort(np.asarray(actual, dtype=float))
    e = e[np.isfinite(e)]
    a = a[np.isfinite(a)]
    n1, n2 = len(e), len(a)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")

    grid = np.concatenate([e, a])
    cdf_e = np.searchsorted(e, grid, side="right") / n1
    cdf_a = np.searchsorted(a, grid, side="right") / n2
    d = float(np.max(np.abs(cdf_e - cdf_a)))

    # Kolmogorov asymptotic survival function
    en = np.sqrt(n1 * n2 / (n1 + n2))
    lam = (en + 0.12 + 0.11 / en) * d
    p = 2 * sum((-1) ** (k - 1) * np.exp(-2 * k * k * lam * lam) for k in range(1, 101))
    return d, float(min(max(p, 0.0), 1.0))


def classify_psi(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value < PSI_MODERATE:
        return "stable"
    if value < PSI_MAJOR:
        return "moderate shift"
    return "major shift - retrain"


def window_features(daily: pd.DataFrame, columns: list[str] | None = None) -> list[str]:
    columns = columns or [
        "Revenue", "Units", "Orders", "Customers", "AvgOrderValue",
        "UnitsPerOrder", "LineItems",
    ]
    return [c for c in columns if c in daily.columns]


def run_drift_report(daily: pd.DataFrame, split_frac: float = 0.5,
                     columns: list[str] | None = None) -> pd.DataFrame:
    """Compare the first half of history against the second half, feature by feature."""
    daily = daily.sort_values("Date").reset_index(drop=True)
    cut = int(len(daily) * split_frac)
    reference, current = daily.iloc[:cut], daily.iloc[cut:]

    rows = []
    for col in window_features(daily, columns):
        ref = reference[col].to_numpy(float)
        cur = current[col].to_numpy(float)
        p = psi(ref, cur)
        d, pval = ks_drift(ref, cur)
        rows.append({
            "feature": col,
            "reference_mean": round(float(np.mean(ref)), 2),
            "current_mean": round(float(np.mean(cur)), 2),
            "mean_change_pct": round(float((np.mean(cur) - np.mean(ref)) / np.mean(ref) * 100), 2)
            if np.mean(ref) else np.nan,
            "psi": round(p, 4),
            "psi_verdict": classify_psi(p),
            "ks_statistic": round(d, 4),
            "ks_p_value": round(pval, 6),
            "drift_detected": bool(pval < 0.05) if np.isfinite(pval) else False,
        })
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def _make_svg_figures(daily: pd.DataFrame, report: pd.DataFrame) -> None:
    """Dependency-free SVG fallback for the drift charts."""
    from src.viz import bar_chart, line_chart, save

    d = report.sort_values("psi")
    colors = ["#16A34A" if v < PSI_MODERATE else "#F59E0B" if v < PSI_MAJOR else "#DC2626"
              for v in d["psi"]]
    save(bar_chart(d["feature"].tolist(), d["psi"].tolist(),
                   "Population Stability Index by feature (reference vs current half)",
                   horizontal=True, colors=colors),
         cfg.FIGURES / "drift_psi.svg")

    # distribution comparison for the two most-drifted features, as histograms
    top = report.nlargest(2, "psi")["feature"].tolist()
    cut = len(daily) // 2
    series = {}
    for col in top[:1]:
        ref = daily[col].iloc[:cut].to_numpy(float)
        cur = daily[col].iloc[cut:].to_numpy(float)
        lo, hi = float(min(ref.min(), cur.min())), float(max(ref.max(), cur.max()))
        if hi <= lo:
            continue
        edges = np.linspace(lo, hi, 26)
        centres = ((edges[:-1] + edges[1:]) / 2).round(2)
        series[f"{col} — reference"] = (centres.tolist(), np.histogram(ref, bins=edges)[0].tolist())
        series[f"{col} — current"] = (centres.tolist(), np.histogram(cur, bins=edges)[0].tolist())
    if series:
        save(line_chart(series, "Distribution shift — most drifted feature",
                        x_label="value", y_label="frequency"),
             cfg.FIGURES / "drift_distributions.svg")


def make_figures(daily: pd.DataFrame, report: pd.DataFrame) -> None:
    from src.optional_deps import MATPLOTLIB

    if not MATPLOTLIB.ok:
        _make_svg_figures(daily, report)
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")

    # 1. PSI bars with thresholds
    fig, ax = plt.subplots(figsize=(10, 5))
    d = report.sort_values("psi")
    colors = ["#16a34a" if v < PSI_MODERATE else "#f59e0b" if v < PSI_MAJOR else "#dc2626"
              for v in d["psi"]]
    ax.barh(d["feature"], d["psi"], color=colors)
    ax.axvline(PSI_MODERATE, color="#f59e0b", ls="--", lw=1.2)
    ax.axvline(PSI_MAJOR, color="#dc2626", ls="--", lw=1.2)
    ax.text(PSI_MODERATE + 0.01, -0.5, "0.10 moderate", color="#b45309", fontsize=8)
    ax.text(PSI_MAJOR + 0.01, -0.5, "0.25 retrain", color="#dc2626", fontsize=8)
    ax.set_title("Population Stability Index by feature (reference vs current half)",
                 fontsize=12, weight="bold")
    ax.set_xlabel("PSI")
    fig.tight_layout()
    fig.savefig(cfg.FIGURES / "drift_psi.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # 2. distribution overlay for the two most-drifted features
    top = report.nlargest(2, "psi")["feature"].tolist()
    if top:
        fig, axes = plt.subplots(1, len(top), figsize=(7 * len(top), 4.6), squeeze=False)
        cut = len(daily) // 2
        for ax, col in zip(axes[0], top):
            sns.kdeplot(daily[col].iloc[:cut], ax=ax, label="reference (first half)",
                        fill=True, color="#2563eb", alpha=0.4)
            sns.kdeplot(daily[col].iloc[cut:], ax=ax, label="current (second half)",
                        fill=True, color="#dc2626", alpha=0.4)
            row = report[report["feature"] == col].iloc[0]
            ax.set_title(f"{col}\nPSI={row['psi']:.3f} ({row['psi_verdict']})",
                         fontsize=11, weight="bold")
            ax.legend(fontsize=8)
        fig.suptitle("Distribution shift — most drifted features", fontsize=13, weight="bold")
        fig.tight_layout()
        fig.savefig(cfg.FIGURES / "drift_distributions.png", dpi=140, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    print("=" * 78)
    print("RetailPulse :: Stage 7 - data drift detection")
    print("=" * 78)

    daily = pd.read_csv(cfg.PROCESSED / "daily_sales_features.csv", parse_dates=["Date"])
    print(f"[drift] daily rows: {len(daily):,}")

    report = run_drift_report(daily)
    print("\n--- drift report (reference = first half, current = second half) ---")
    print(report[["feature", "reference_mean", "current_mean", "mean_change_pct",
                  "psi", "psi_verdict", "ks_statistic", "ks_p_value"]].to_string(index=False))

    major = report[report["psi"] >= PSI_MAJOR]
    moderate = report[(report["psi"] >= PSI_MODERATE) & (report["psi"] < PSI_MAJOR)]
    print(f"\n[drift] major shift: {len(major)} features | moderate: {len(moderate)} | "
          f"stable: {len(report) - len(major) - len(moderate)}")
    if len(major):
        print("[drift] ACTION: schedule retraining — "
              f"{', '.join(major['feature'].tolist())}")
    else:
        print("[drift] ACTION: no retraining required; continue monitoring")

    report.to_csv(cfg.PROCESSED / "drift_report.csv", index=False)
    make_figures(daily, report)
    print(f"[drift] wrote outputs -> {cfg.PROCESSED}")


if __name__ == "__main__":
    main()
