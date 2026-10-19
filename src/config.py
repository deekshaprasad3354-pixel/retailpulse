"""RetailPulse — central configuration.

Every path, business rule and model setting lives here so nothing is hard-coded
inside the pipeline scripts. Change a value once, and every stage picks it up.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# PROJECT_ROOT is the retailpulse/ folder that contains src/, dashboard/ and
# requirements.txt. Everything the pipeline writes lives under it by default, so
# the project stays self-contained and portable: copy the folder anywhere and it
# still works.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Override with RETAILPULSE_DATA_ROOT when the (large) raw data should live
# somewhere else — a shared drive, a scratch disk, or outside a git checkout.
# Example:
#   RETAILPULSE_DATA_ROOT=D:\retaildata python -m src.run_pipeline
ROOT = Path(os.environ.get("RETAILPULSE_DATA_ROOT", PROJECT_ROOT)).resolve()

DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS = ROOT / "models"
MLRUNS = ROOT / "mlruns"

for _p in (RAW, INTERIM, PROCESSED, REPORTS, FIGURES, MODELS, MLRUNS):
    _p.mkdir(parents=True, exist_ok=True)

# Source workbooks downloaded from the UCI ML Repository
ONLINE_RETAIL_II = RAW / "online_retail_ii" / "online_retail_II.xlsx"
ONLINE_RETAIL_I = RAW / "online_retail_i" / "Online Retail.xlsx"

# Canonical intermediate files.
# Parquet is used when a Parquet engine is installed (pyarrow / fastparquet);
# otherwise the pipeline transparently falls back to gzip CSV, so it runs on a
# bare pandas install too.
def _prefer_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


USE_PARQUET = _prefer_parquet()


def _resolve_cache(stem: str) -> Path:
    p_parquet = INTERIM / f"{stem}.parquet"
    p_gz = INTERIM / f"{stem}.csv.gz"
    if p_parquet.exists():
        return p_parquet
    if p_gz.exists():
        return p_gz
    return p_parquet if USE_PARQUET else p_gz


TX_CACHE = _resolve_cache("transactions")
CLEAN_CACHE = _resolve_cache("transactions_clean")

# --------------------------------------------------------------------------
# Business rules / data-quality thresholds
# --------------------------------------------------------------------------
CURRENCY = "GBP"
CURRENCY_SYMBOL = "£"

# Sheets that make up the primary dataset (Online Retail II, Dec 2009 - Dec 2011)
RETAIL_II_SHEETS = ("Year 2009-2010", "Year 2010-2011")

# A row is treated as a cancellation when its invoice starts with "C"
CANCELLATION_PREFIX = "C"

# Rows with these stock codes are not real products (postage, fees, discounts)
NON_PRODUCT_STOCKCODES = {
    "POST", "D", "M", "BANK CHARGES", "PAD", "CRUK", "S", "AMAZONFEE",
    "B", "C2", "C3", "DOT", "gift_0001_10", "gift_0001_20", "gift_0001_30",
    "gift_0001_40", "gift_0001_50", "gift_0001_80", "gift_0001_90",
    "TEST001", "TEST002", "ADJUST", "ADJUST2", "SP1002", "DCGS0003",
    "DCGS0004", "DCGS0005", "DCGS0006", "DCGS0007", "DCGS0008", "DCGS0009",
    "DCGS0010", "DCGS0011", "DCGS0012", "DCGS0013", "DCGS0014", "DCGS0015",
    "DCGS0016", "DCGS0017", "DCGS0018", "DCGS0019", "DCGS0020",
}

# Estimated gross margin used when no cost price is available in the source
DEFAULT_MARGIN = 0.35

# --------------------------------------------------------------------------
# Customer analytics
# --------------------------------------------------------------------------
RFM_QUANTILES = 5
SEGMENT_K = 4                       # K-Means clusters for RFM segmentation

CHURN_DAYS = 90                     # no purchase within 90 days => churned
CHURN_HORIZON_DAYS = 90             # observation gap between features and label
CHURN_TEST_FRACTION = 0.25

# --------------------------------------------------------------------------
# Forecasting
# --------------------------------------------------------------------------
FORECAST_HORIZON_DAYS = 90
FORECAST_HORIZON_WEEKS = 12
BACKTEST_FOLDS = 4
SEASONAL_PERIOD = 7                 # weekly seasonality for daily sales

# --------------------------------------------------------------------------
# Inventory optimisation
# --------------------------------------------------------------------------
SERVICE_LEVEL = 0.95                # 95% cycle service level => z = 1.645
ORDERING_COST = 25.0                # cost to place one purchase order
HOLDING_RATE = 0.20                 # annual holding cost as % of unit cost
LEAD_TIME_DAYS = 7
REVIEW_PERIOD_DAYS = 7

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RANDOM_STATE = 42


# --------------------------------------------------------------------------
# Cache helpers (Parquet when available, gzip CSV otherwise)
# --------------------------------------------------------------------------
def write_cache(df, path: Path) -> Path:
    """Persist a DataFrame to `path`, honouring the configured format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    return path


def read_cache(path: Path):
    """Load a DataFrame written by :func:`write_cache`."""
    import pandas as pd

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)
