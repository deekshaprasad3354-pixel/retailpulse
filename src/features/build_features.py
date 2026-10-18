"""Stage 2 — Data cleaning and feature engineering.

Turns the raw transaction table into three analysis-ready tables:

    transactions_clean.parquet  line-item level, quality-filtered
    daily_sales_features.csv    one row per calendar day (time series base)
    customer_rfm.csv            one row per customer (RFM base)

Cleaning rules (each one is a business decision, not just a tidy-up):

    1. Drop exact duplicate rows.
    2. Drop rows with no CustomerID  -> can't attribute behaviour to a customer.
    3. Drop cancellations ("C..." invoices) -> they are returns, not demand.
    4. Drop Quantity <= 0            -> returns / stock write-offs.
    5. Drop Price  <= 0              -> freebies and data-entry errors.
    6. Drop non-product stock codes  -> postage, bank charges, manual adjustments.
    7. Attribute revenue to the invoice date (no delivery-date column exists).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config as cfg
from src.ingestion.load_data import build_transactions


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def clean_transactions(tx: pd.DataFrame, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the quality rules and return (clean_df, funnel_report)."""
    funnel: list[tuple[str, int]] = [("raw rows", len(tx))]
    df = tx.copy()

    def step(label: str, mask: pd.Series) -> None:
        nonlocal df
        removed = int((~mask).sum())
        df = df[mask]
        funnel.append((label, len(df)))
        if verbose:
            print(f"  {label:<38s} -{removed:>8,}  -> {len(df):>9,}")

    step("drop exact duplicates", ~df.duplicated())
    step("drop missing CustomerID", df["CustomerID"].notna())
    step(f"drop cancellations ({cfg.CANCELLATION_PREFIX}*)",
         ~df["Invoice"].str.startswith(cfg.CANCELLATION_PREFIX))
    step("drop Quantity <= 0", df["Quantity"] > 0)
    step("drop Price <= 0", df["Price"] > 0)
    step("drop non-product stock codes", ~df["StockCode"].isin(cfg.NON_PRODUCT_STOCKCODES))

    df["CustomerID"] = df["CustomerID"].astype("int64")
    df["Quantity"] = df["Quantity"].astype("int64")
    df["Revenue"] = (df["Quantity"] * df["Price"]).round(2)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Date"] = df["InvoiceDate"].dt.normalize()
    df["Year"] = df["Date"].dt.year.astype("int16")
    df["Month"] = df["Date"].dt.month.astype("int8")
    df["DayOfWeek"] = df["Date"].dt.dayofweek.astype("int8")
    df["Hour"] = df["InvoiceDate"].dt.hour.astype("int8")
    df["IsWeekend"] = df["DayOfWeek"] >= 5
    df["IsCancelled"] = False

    # Estimated cost/profit: the source has no cost price, so we apply a
    # configurable gross margin. Documented assumption, not ground truth.
    df["EstimatedCost"] = (df["Revenue"] * (1 - cfg.DEFAULT_MARGIN)).round(2)
    df["EstimatedProfit"] = (df["Revenue"] - df["EstimatedCost"]).round(2)

    df = df.sort_values(["InvoiceDate", "Invoice"]).reset_index(drop=True)
    report = pd.DataFrame(funnel, columns=["step", "rows_remaining"])
    report["rows_removed"] = report["rows_remaining"].shift(1).fillna(report["rows_remaining"]) - report["rows_remaining"]
    report["rows_removed"] = report["rows_removed"].astype(int)
    return df, report


# --------------------------------------------------------------------------
# Time-series features
# --------------------------------------------------------------------------
def _calendar_frame(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Continuous daily calendar, so gap days become explicit zeros."""
    days = pd.date_range(start, end, freq="D")
    cal = pd.DataFrame({"Date": days})
    cal["Year"] = cal["Date"].dt.year.astype("int16")
    cal["Month"] = cal["Date"].dt.month.astype("int8")
    cal["Day"] = cal["Date"].dt.day.astype("int8")
    cal["DayOfWeek"] = cal["Date"].dt.dayofweek.astype("int8")
    cal["DayName"] = cal["Date"].dt.day_name()
    cal["Week"] = cal["Date"].dt.isocalendar().week.astype("int8")
    cal["Quarter"] = cal["Date"].dt.quarter.astype("int8")
    cal["YearMonth"] = cal["Date"].dt.to_period("M").astype(str)
    cal["IsWeekend"] = cal["DayOfWeek"] >= 5
    return cal


def build_daily_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate line items to one row per trading calendar day."""
    cal = _calendar_frame(df["Date"].min(), df["Date"].max())

    daily = df.groupby("Date").agg(
        Revenue=("Revenue", "sum"),
        Units=("Quantity", "sum"),
        Orders=("Invoice", "nunique"),
        Customers=("CustomerID", "nunique"),
        LineItems=("Revenue", "size"),
        Products=("StockCode", "nunique"),
        Profit=("EstimatedProfit", "sum"),
    ).reset_index()

    out = cal.merge(daily, on="Date", how="left")
    out[["Revenue", "Units", "Orders", "Customers", "LineItems", "Products", "Profit"]] = (
        out[["Revenue", "Units", "Orders", "Customers", "LineItems", "Products", "Profit"]]
        .fillna(0)
    )
    out["IsTradingDay"] = out["Revenue"] > 0
    out["AvgOrderValue"] = np.where(out["Orders"] > 0, out["Revenue"] / out["Orders"], 0.0)
    out["UnitsPerOrder"] = np.where(out["Orders"] > 0, out["Units"] / out["Orders"], 0.0)

    # --- lag features: "what happened yesterday / a week ago" ---------------
    for lag in (1, 7, 14, 28, 364):
        out[f"RevenueLag{lag}"] = out["Revenue"].shift(lag)

    # --- rolling averages: smooth out daily noise --------------------------
    shifted = out["Revenue"].shift(1)          # shift first: never leak today into today
    for window in (7, 14, 28, 30, 90):
        out[f"RevenueRollMean{window}"] = shifted.rolling(window, min_periods=1).mean().round(2)
        out[f"RevenueRollStd{window}"] = shifted.rolling(window, min_periods=2).std().round(2)

    out["RevenueRollMax28"] = shifted.rolling(28, min_periods=1).max()
    out["RevenueRollMin28"] = shifted.rolling(28, min_periods=1).min()

    # --- trend / momentum --------------------------------------------------
    out["RevenueWoW"] = (out["RevenueRollMean7"] / out["RevenueRollMean28"].replace(0, np.nan) - 1).round(4)
    out["DayIndex"] = np.arange(len(out))
    out["IsMonthStart"] = out["Date"].dt.is_month_start
    out["IsMonthEnd"] = out["Date"].dt.is_month_end
    out["DaysToChristmas"] = (
        pd.to_datetime(dict(year=out["Date"].dt.year, month=12, day=25)) - out["Date"]
    ).dt.days
    return out


def build_weekly_sales(daily: pd.DataFrame) -> pd.DataFrame:
    """Weekly aggregation — smoother series, better forecast accuracy."""
    d = daily.set_index("Date")
    weekly = d.resample("W-SUN").agg(
        Revenue=("Revenue", "sum"),
        Units=("Units", "sum"),
        Orders=("Orders", "sum"),
        Customers=("Customers", "sum"),
        TradingDays=("IsTradingDay", "sum"),
    ).reset_index()
    weekly["AvgOrderValue"] = np.where(weekly["Orders"] > 0, weekly["Revenue"] / weekly["Orders"], 0.0)
    weekly["RevenuePerTradingDay"] = np.where(
        weekly["TradingDays"] > 0, weekly["Revenue"] / weekly["TradingDays"], 0.0
    )
    weekly["WeekIndex"] = np.arange(len(weekly))
    return weekly


# --------------------------------------------------------------------------
# Customer features (RFM)
# --------------------------------------------------------------------------
def build_customer_rfm(df: pd.DataFrame, snapshot: pd.Timestamp | None = None) -> pd.DataFrame:
    """Recency / Frequency / Monetary per customer, with quintile scores."""
    snapshot = snapshot or (df["Date"].max() + pd.Timedelta(days=1))

    rfm = df.groupby("CustomerID").agg(
        Recency=("Date", lambda s: (snapshot - s.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("Revenue", "sum"),
        Units=("Quantity", "sum"),
        FirstPurchase=("Date", "min"),
        LastPurchase=("Date", "max"),
        Products=("StockCode", "nunique"),
        Countries=("Country", "nunique"),
        AvgOrderValue=("Revenue", lambda s: float(s.mean())),
    ).reset_index()

    rfm["TenureDays"] = (rfm["LastPurchase"] - rfm["FirstPurchase"]).dt.days
    rfm["PurchaseRate"] = (rfm["Frequency"] / rfm["TenureDays"].clip(lower=1) * 30).round(3)
    rfm["Monetary"] = rfm["Monetary"].round(2)
    rfm["AvgOrderValue"] = rfm["AvgOrderValue"].round(2)

    # Quintile scores: 5 = best. Recency is inverted (more recent = better).
    rfm["R_Score"] = pd.qcut(rfm["Recency"], cfg.RFM_QUANTILES, labels=range(5, 0, -1)).astype(int)
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), cfg.RFM_QUANTILES,
                             labels=range(1, 6)).astype(int)
    rfm["M_Score"] = pd.qcut(rfm["Monetary"], cfg.RFM_QUANTILES, labels=range(1, 6)).astype(int)
    rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]
    rfm["RFM_Cell"] = (
        rfm["R_Score"].astype(str) + rfm["F_Score"].astype(str) + rfm["M_Score"].astype(str)
    )
    rfm["SnapshotDate"] = snapshot
    return rfm.sort_values("Monetary", ascending=False).reset_index(drop=True)


def main() -> None:
    print("=" * 78)
    print("RetailPulse :: Stage 2 - cleaning and feature engineering")
    print("=" * 78)

    tx = build_transactions()
    print(f"\n[clean] raw transactions: {len(tx):,}\n")
    clean, funnel = clean_transactions(tx)
    print(f"\n[clean] clean transactions: {len(clean):,}")

    daily = build_daily_sales(clean)
    weekly = build_weekly_sales(daily)
    rfm = build_customer_rfm(clean)

    cfg.write_cache(clean, cfg.CLEAN_CACHE)
    daily.to_csv(cfg.PROCESSED / "daily_sales_features.csv", index=False)
    weekly.to_csv(cfg.PROCESSED / "weekly_sales_features.csv", index=False)
    rfm.to_csv(cfg.PROCESSED / "customer_rfm.csv", index=False)
    funnel.to_csv(cfg.REPORTS / "cleaning_funnel.csv", index=False)

    print(f"\n[daily ] {daily.shape[0]:,} days x {daily.shape[1]} cols")
    print(f"[weekly] {weekly.shape[0]:,} weeks x {weekly.shape[1]} cols")
    print(f"[rfm   ] {rfm.shape[0]:,} customers x {rfm.shape[1]} cols")
    print(f"\nTotal revenue   : {cfg.CURRENCY_SYMBOL}{clean['Revenue'].sum():,.2f}")
    print(f"Unique customers: {clean['CustomerID'].nunique():,}")
    print(f"Unique products : {clean['StockCode'].nunique():,}")
    print(f"Trading days    : {int(daily['IsTradingDay'].sum()):,}")
    print(f"\n[clean] wrote -> {cfg.PROCESSED}")


if __name__ == "__main__":
    main()
