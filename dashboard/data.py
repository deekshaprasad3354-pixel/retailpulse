"""RetailPulse dashboard — data access layer.

All file reads go through here and are cached by Streamlit, so switching pages
never re-reads a CSV. Every loader fails softly: if a stage has not been run
yet, the page shows a helpful message instead of a traceback.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# make `src` importable when Streamlit runs this file directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402


@st.cache_data(show_spinner=False)
def _csv(path_str: str) -> pd.DataFrame | None:
    path = Path(path_str)
    if not path.exists():
        return None
    return pd.read_csv(path)


def require(df: pd.DataFrame | None, stage: str, hint: str = "") -> pd.DataFrame:
    """Stop the page politely when an upstream stage has not produced its output."""
    if df is None:
        st.warning(
            f"**{stage}** is not available yet.\n\n"
            f"Run the pipeline first:\n\n"
            f"```bash\npython -m src.run_pipeline --only {hint or stage.lower()}\n```"
        )
        st.stop()
    return df


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading daily sales…")
def daily_sales() -> pd.DataFrame:
    df = _csv(str(cfg.PROCESSED / "daily_sales_features.csv"))
    if df is not None:
        df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_data(show_spinner="Loading weekly sales…")
def weekly_sales() -> pd.DataFrame:
    df = _csv(str(cfg.PROCESSED / "weekly_sales_features.csv"))
    if df is not None:
        df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_data(show_spinner="Loading customers…")
def customer_segments() -> pd.DataFrame:
    return _csv(str(cfg.PROCESSED / "customer_segments.csv"))


@st.cache_data(show_spinner="Loading segment profiles…")
def segment_profiles() -> pd.DataFrame:
    return _csv(str(cfg.PROCESSED / "segment_profiles.csv"))


@st.cache_data(show_spinner="Loading cluster profiles…")
def cluster_profiles() -> pd.DataFrame:
    return _csv(str(cfg.PROCESSED / "cluster_profiles.csv"))


@st.cache_data(show_spinner="Loading churn scores…")
def churn_scores() -> pd.DataFrame:
    return _csv(str(cfg.PROCESSED / "customer_churn.csv"))


@st.cache_data(show_spinner="Loading churn metrics…")
def churn_comparison() -> pd.DataFrame:
    return _csv(str(cfg.PROCESSED / "churn_model_comparison.csv"))


@st.cache_data(show_spinner="Loading feature importances…")
def churn_importance() -> pd.DataFrame:
    return _csv(str(cfg.PROCESSED / "churn_feature_importance.csv"))


@st.cache_data(show_spinner="Loading forecast…")
def forecast_daily() -> pd.DataFrame:
    df = _csv(str(cfg.PROCESSED / "forecast_daily.csv"))
    if df is not None:
        df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_data(show_spinner="Loading weekly forecast…")
def forecast_weekly() -> pd.DataFrame:
    df = _csv(str(cfg.PROCESSED / "forecast_weekly_12w.csv"))
    if df is not None:
        df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_data(show_spinner="Loading backtest…")
def backtest() -> pd.DataFrame:
    return _csv(str(cfg.PROCESSED / "forecast_backtest.csv"))


@st.cache_data(show_spinner="Loading model comparison…")
def model_comparison() -> pd.DataFrame:
    df = _csv(str(cfg.PROCESSED / "model_comparison_daily.csv"))
    if df is None:
        df = _csv(str(cfg.PROCESSED / "model_comparison.csv"))
    return df


@st.cache_data(show_spinner="Loading inventory policy…")
def inventory_policy() -> pd.DataFrame:
    return _csv(str(cfg.PROCESSED / "inventory_policy.csv"))


@st.cache_data(show_spinner="Loading inventory metrics…")
def inventory_metrics() -> pd.DataFrame:
    return _csv(str(cfg.PROCESSED / "inventory_metrics.csv"))


@st.cache_data(show_spinner="Loading ABC/XYZ matrix…")
def abc_xyz_matrix() -> pd.DataFrame:
    return _csv(str(cfg.PROCESSED / "abc_xyz_matrix.csv"))


@st.cache_data(show_spinner="Loading drift report…")
def drift_report() -> pd.DataFrame:
    return _csv(str(cfg.PROCESSED / "drift_report.csv"))


@st.cache_data(show_spinner="Loading transactions…")
def transactions() -> pd.DataFrame | None:
    path = cfg.CLEAN_CACHE
    if not path.exists():
        return None
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, usecols=[
        "Invoice", "StockCode", "Description", "Quantity", "Date", "Price",
        "CustomerID", "Country", "Revenue",
    ])


@st.cache_data(show_spinner=False)
def pipeline_status() -> pd.DataFrame:
    """Which pipeline outputs exist — drives the Status page."""
    expected = {
        "features": ["daily_sales_features.csv", "weekly_sales_features.csv", "customer_rfm.csv"],
        "segmentation": ["customer_segments.csv", "segment_profiles.csv", "cluster_profiles.csv"],
        "forecasting": ["forecast_daily.csv", "forecast_weekly_12w.csv", "model_comparison_daily.csv"],
        "churn": ["customer_churn.csv", "churn_model_comparison.csv"],
        "inventory": ["inventory_policy.csv", "inventory_metrics.csv"],
        "drift": ["drift_report.csv"],
    }
    rows = []
    for stage, files in expected.items():
        missing = [f for f in files if not (cfg.PROCESSED / f).exists()]
        rows.append({
            "stage": stage,
            "status": "ready" if not missing else "not run",
            "files_present": f"{len(files) - len(missing)}/{len(files)}",
            "missing": ", ".join(missing) if missing else "-",
        })
    return pd.DataFrame(rows)
