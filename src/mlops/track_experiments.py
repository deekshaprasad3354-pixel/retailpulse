"""Stage 8 — Experiment tracking with MLflow (optional).

Every headline number the pipeline produces is logged to MLflow as a run, so
model results are reproducible and comparable over time rather than living in a
notebook cell.

This module is deliberately defensive: if MLflow is not installed, or the
tracking store cannot be reached, the pipeline continues. Tracking is valuable
but it must never be the reason a build fails.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config as cfg

EXPERIMENT = "retailpulse"


def _load(name: str) -> pd.DataFrame | None:
    path = cfg.PROCESSED / name
    return pd.read_csv(path) if path.exists() else None


def build_run_payload() -> dict:
    """Collect the metrics and parameters that describe this pipeline run."""
    metrics: dict[str, float] = {}
    params: dict[str, object] = {
        "rfm_quantiles": cfg.RFM_QUANTILES,
        "segment_k": cfg.SEGMENT_K,
        "churn_days": cfg.CHURN_DAYS,
        "churn_horizon_days": cfg.CHURN_HORIZON_DAYS,
        "forecast_horizon_days": cfg.FORECAST_HORIZON_DAYS,
        "backtest_folds": cfg.BACKTEST_FOLDS,
        "service_level": cfg.SERVICE_LEVEL,
        "lead_time_days": cfg.LEAD_TIME_DAYS,
        "default_margin": cfg.DEFAULT_MARGIN,
        "random_state": cfg.RANDOM_STATE,
    }

    comp = _load("model_comparison_daily.csv")
    if comp is None:
        comp = _load("model_comparison.csv")
    if comp is not None and not comp.empty:
        best = comp.sort_values("mape").iloc[0]
        params["forecast_model"] = best["model"]
        metrics["forecast_mape"] = float(best["mape"])
        metrics["forecast_mae"] = float(best["mae"])
        metrics["forecast_rmse"] = float(best["rmse"])
        metrics["forecast_r2"] = float(best["r2"])

    churn = _load("churn_model_comparison.csv")
    if churn is not None and not churn.empty:
        best = churn.sort_values("auc", ascending=False).iloc[0]
        params["churn_model"] = best["model"]
        metrics["churn_auc"] = float(best["auc"])
        metrics["churn_ks"] = float(best["ks"])
        metrics["churn_f1"] = float(best["f1"])
        metrics["churn_lift_top10"] = float(best["lift_top10"])

    seg = _load("segment_profiles.csv")
    if seg is not None and not seg.empty:
        metrics["segments"] = int(len(seg))
        metrics["customers"] = int(seg["Customers"].sum())
        champs = seg[seg["Segment"] == "Champions"]
        if not champs.empty:
            metrics["champion_revenue_share_pct"] = float(champs["RevenueSharePct"].iloc[0])

    inv = _load("inventory_metrics.csv")
    if inv is not None and not inv.empty:
        inv = inv.set_index("policy")
        if "optimised" in inv.index:
            metrics["fill_rate_pct"] = float(inv.loc["optimised", "fill_rate_pct"])
            metrics["stockout_days"] = float(inv.loc["optimised", "stockout_days"])
        if "naive" in inv.index and "optimised" in inv.index:
            metrics["stockout_reduction_pct"] = float(
                (1 - inv.loc["optimised", "units_lost"] / max(inv.loc["naive", "units_lost"], 1)) * 100
            )

    drift = _load("drift_report.csv")
    if drift is not None and not drift.empty:
        metrics["drifted_features"] = int((drift["psi"] >= 0.25).sum())
        metrics["max_psi"] = float(drift["psi"].max())

    daily = _load("daily_sales_features.csv")
    if daily is not None and not daily.empty:
        metrics["total_revenue"] = float(daily["Revenue"].sum())
        metrics["trading_days"] = float(daily["IsTradingDay"].sum())

    return {"metrics": metrics, "params": params}


def log_to_mlflow(payload: dict | None = None) -> bool:
    """Log one run. Returns True if tracking succeeded."""
    try:
        import mlflow
    except ImportError:
        print("[mlflow] not installed — skipping experiment tracking")
        print("         install with: pip install mlflow")
        return False

    payload = payload or build_run_payload()
    metrics, params = payload["metrics"], payload["params"]

    try:
        mlflow.set_tracking_uri(f"file:///{cfg.MLRUNS.as_posix()}")
        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name="pipeline-run") as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            for fig in sorted(cfg.FIGURES.glob("*.png")):
                mlflow.log_artifact(str(fig), artifact_path="figures")
            for name in ("model_comparison_daily.csv", "churn_model_comparison.csv",
                         "segment_profiles.csv", "inventory_metrics.csv", "drift_report.csv"):
                path = cfg.PROCESSED / name
                if path.exists():
                    mlflow.log_artifact(str(path), artifact_path="reports")
            print(f"[mlflow] logged run {run.info.run_id}")
            print(f"[mlflow] view with: mlflow ui --backend-store-uri {cfg.MLRUNS}")
        return True
    except Exception as exc:
        print(f"[mlflow] tracking skipped ({type(exc).__name__}: {exc})")
        return False


def main() -> None:
    print("=" * 78)
    print("RetailPulse :: Stage 8 - experiment tracking")
    print("=" * 78)

    payload = build_run_payload()
    print("\n--- parameters ---")
    for k, v in payload["params"].items():
        print(f"  {k:<26s} {v}")
    print("\n--- metrics ---")
    for k, v in payload["metrics"].items():
        print(f"  {k:<26s} {v:,.4f}" if isinstance(v, float) else f"  {k:<26s} {v}")

    pd.DataFrame(
        [{"key": k, "value": str(v)} for k, v in {**payload["params"], **payload["metrics"]}.items()]
    ).to_csv(cfg.REPORTS / "run_metrics.csv", index=False)

    log_to_mlflow(payload)


if __name__ == "__main__":
    main()
