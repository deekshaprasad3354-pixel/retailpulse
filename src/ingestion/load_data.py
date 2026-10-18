"""Stage 1 — Data ingestion.

Reads the raw UCI Online Retail workbooks, normalises the two slightly
different schemas into one transaction table, and caches it as Parquet so the
rest of the pipeline never has to pay the Excel-reading cost again.

Source:
    Online Retail II  (UCI id 502) — Dec 2009 to Dec 2011, two sheets
    Online Retail     (UCI id 352) — Dec 2010 to Dec 2011, one sheet

The two Online Retail II sheets overlap on December 2010. We concatenate them
and drop exact duplicates, which recovers the full two-year history.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config as cfg


# Canonical column names used everywhere downstream
COLUMNS = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "CustomerID", "Country"]


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to the canonical schema and enforce dtypes."""
    df = df.rename(columns={"Customer ID": "CustomerID", "UnitPrice": "Price"})
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Unexpected schema, missing columns: {missing}")
    df = df[COLUMNS].copy()

    df["Invoice"] = df["Invoice"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    df["Description"] = df["Description"].astype(str).str.strip()
    df["Country"] = df["Country"].astype(str).str.strip()
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").astype("Int64")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce").astype("float64")
    df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce").astype("Int64")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    return df


def load_excel(path: Path, sheet_name: str | int = 0) -> pd.DataFrame:
    """Load one sheet of a raw workbook and normalise it."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            "Run `python -m src.ingestion.download_data` first, or place the UCI "
            "workbook at that path."
        )
    print(f"  reading {path.name} :: sheet {sheet_name!r} ...", flush=True)
    df = pd.read_excel(path, sheet_name=sheet_name)
    return _normalise(df)


def build_transactions(force: bool = False) -> pd.DataFrame:
    """Merge every source sheet into one de-duplicated transaction table."""
    if cfg.TX_CACHE.exists() and not force:
        print(f"[ingest] cache hit -> {cfg.TX_CACHE.name}")
        return cfg.read_cache(cfg.TX_CACHE)

    frames: list[pd.DataFrame] = []
    for sheet in cfg.RETAIL_II_SHEETS:
        frames.append(load_excel(cfg.ONLINE_RETAIL_II, sheet))

    tx = pd.concat(frames, ignore_index=True)
    before = len(tx)
    tx = tx.drop_duplicates()
    print(f"[ingest] {before:,} raw rows -> {len(tx):,} after dropping {before - len(tx):,} duplicates")

    tx = tx.sort_values("InvoiceDate").reset_index(drop=True)
    cfg.write_cache(tx, cfg.TX_CACHE)
    print(f"[ingest] wrote {cfg.TX_CACHE} ({len(tx):,} rows)")
    return tx


def quality_report(tx: pd.DataFrame) -> pd.DataFrame:
    """Summarise the raw data-quality issues we are about to fix."""
    rows = [
        ("total rows", len(tx)),
        ("duplicate rows", int(tx.duplicated().sum())),
        ("missing CustomerID", int(tx["CustomerID"].isna().sum())),
        ("missing Description", int(tx["Description"].isna().sum())),
        ("cancelled invoices (C*)", int(tx["Invoice"].str.startswith(cfg.CANCELLATION_PREFIX).sum())),
        ("zero/negative quantity", int((tx["Quantity"] <= 0).sum())),
        ("zero/negative price", int((tx["Price"] <= 0).sum())),
        ("non-product stock codes", int(tx["StockCode"].isin(cfg.NON_PRODUCT_STOCKCODES).sum())),
        ("date range", f"{tx['InvoiceDate'].min():%Y-%m-%d} to {tx['InvoiceDate'].max():%Y-%m-%d}"),
        ("unique invoices", int(tx["Invoice"].nunique())),
        ("unique customers", int(tx["CustomerID"].nunique())),
        ("unique products", int(tx["StockCode"].nunique())),
        ("unique countries", int(tx["Country"].nunique())),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def main() -> None:
    print("=" * 78)
    print("RetailPulse :: Stage 1 - ingestion")
    print("=" * 78)
    tx = build_transactions(force=True)
    report = quality_report(tx)
    print(report.to_string(index=False))
    out = cfg.REPORTS / "data_quality_report.csv"
    report.to_csv(out, index=False)
    print(f"\n[ingest] wrote {out}")


if __name__ == "__main__":
    main()
