"""Dashboard page 1 — Sales analytics.

Descriptive analytics: what sold, when, and where. This is the page a store
manager opens in the morning.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard import data as ds
from dashboard.design import (
    AMBER, BLUE, CYAN, GREEN, MUTED, PURPLE, RED, hero, kpi_row, money, note,
    number, section, style_fig,
)


def _filter(daily: pd.DataFrame) -> pd.DataFrame:
    """Date-range control in the main area so it is visible on first load."""
    lo, hi = daily["Date"].min().date(), daily["Date"].max().date()
    c1, c2, c3 = st.columns([1.2, 1.2, 2.4])
    with c1:
        start = st.date_input("From", value=lo, min_value=lo, max_value=hi, key="sales_from")
    with c2:
        end = st.date_input("To", value=hi, min_value=lo, max_value=hi, key="sales_to")
    with c3:
        grain = st.radio("Aggregate by", ["Day", "Week", "Month"], horizontal=True, key="sales_grain")

    mask = (daily["Date"].dt.date >= start) & (daily["Date"].dt.date <= end)
    out = daily.loc[mask].copy()
    if out.empty:
        st.info("No data in the selected range.")
        st.stop()
    return out, grain


def render() -> None:
    daily = ds.require(ds.daily_sales(), "Sales analytics", "features")
    hero(
        "Sales Analytics",
        "Revenue, order and basket behaviour across the full trading history. "
        "Use the filters to narrow the window; every number below responds to them.",
    )

    daily, grain = _filter(daily)

    total_revenue = daily["Revenue"].sum()
    total_orders = daily["Orders"].sum()
    total_units = daily["Units"].sum()
    avg_order = total_revenue / total_orders if total_orders else 0
    trading_days = int(daily["IsTradingDay"].sum())
    customers = int(daily["Customers"].max())

    kpi_row([
        {"label": "Total revenue", "value": money(total_revenue), "hint": f"{trading_days:,} trading days"},
        {"label": "Orders", "value": number(total_orders), "hint": f"{number(total_units)} units sold"},
        {"label": "Average order value", "value": money(avg_order, "£"), "hint": "revenue ÷ orders"},
        {"label": "Revenue / trading day", "value": money(total_revenue / max(trading_days, 1)),
         "hint": "excludes closed days"},
    ])

    # ---- revenue trend -----------------------------------------------------
    section("Revenue trend", f"aggregated by {grain.lower()}")
    if grain == "Day":
        trend = daily[["Date", "Revenue"]].copy()
    elif grain == "Week":
        trend = daily.set_index("Date")["Revenue"].resample("W-SUN").sum().reset_index()
    else:
        trend = daily.set_index("Date")["Revenue"].resample("MS").sum().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend["Date"], y=trend["Revenue"], mode="lines",
        line=dict(color=BLUE, width=2), fill="tozeroy",
        fillcolor="rgba(37,99,235,0.12)", name="Revenue",
    ))
    # 28-day moving average to expose the underlying trend
    if len(trend) > 28:
        fig.add_trace(go.Scatter(
            x=trend["Date"], y=trend["Revenue"].rolling(28, min_periods=1).mean(),
            mode="lines", line=dict(color=AMBER, width=2, dash="dash"), name="28-period average",
        ))
    st.plotly_chart(style_fig(fig, "Revenue over time", 400), use_container_width=True)

    # ---- seasonality -------------------------------------------------------
    section("Seasonality", "weekday and monthly patterns")
    c1, c2 = st.columns(2)

    with c1:
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow = daily.copy()
        dow["DayName"] = dow["Date"].dt.day_name()
        agg = dow.groupby("DayName", as_index=False)["Revenue"].mean()
        agg["DayName"] = pd.Categorical(agg["DayName"], categories=dow_order, ordered=True)
        agg = agg.sort_values("DayName")
        f = px.bar(agg, x="DayName", y="Revenue", color="Revenue",
                   color_continuous_scale=["#DBEAFE", BLUE])
        f.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_fig(f, "Average revenue by weekday", 350), use_container_width=True)

    with c2:
        mon = daily.copy()
        mon["MonthName"] = mon["Date"].dt.strftime("%b")
        agg = mon.groupby(["MonthName", mon["Date"].dt.month], as_index=False)["Revenue"].sum()
        agg.columns = ["MonthName", "MonthNo", "Revenue"]
        agg = agg.sort_values("MonthNo")
        f = px.bar(agg, x="MonthName", y="Revenue", color="Revenue",
                   color_continuous_scale=["#CFFAFE", CYAN])
        f.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_fig(f, "Total revenue by month (both years combined)", 350),
                        use_container_width=True)

    # ---- orders vs basket --------------------------------------------------
    section("Order volume and basket size", "how many orders, and how big each one is")
    c1, c2 = st.columns(2)
    with c1:
        f = px.line(daily, x="Date", y="RevenueRollMean7" if "RevenueRollMean7" in daily else "Revenue",
                    labels={"RevenueRollMean7": "7-day average revenue"})
        f.update_traces(line=dict(color=PURPLE, width=2))
        st.plotly_chart(style_fig(f, "7-day average revenue (noise removed)", 340),
                        use_container_width=True)
    with c2:
        f = px.scatter(daily, x="Orders", y="AvgOrderValue", size="Revenue",
                       color="Revenue", color_continuous_scale="Viridis",
                       labels={"Orders": "orders per day", "AvgOrderValue": "average order value (£)"})
        f.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_fig(f, "Orders per day vs average order value", 340),
                        use_container_width=True)

    # ---- top products and countries ---------------------------------------
    tx = ds.transactions()
    if tx is not None:
        section("Product and country performance", "the commercial detail behind the totals")
        tx["Date"] = pd.to_datetime(tx["Date"])
        mask = (tx["Date"].dt.date >= daily["Date"].min().date()) & \
               (tx["Date"].dt.date <= daily["Date"].max().date())
        sub = tx.loc[mask]

        c1, c2 = st.columns(2)
        with c1:
            top = (sub.groupby("Description", as_index=False)
                   .agg(Revenue=("Revenue", "sum"), Units=("Quantity", "sum"))
                   .nlargest(12, "Revenue").sort_values("Revenue"))
            f = px.bar(top, x="Revenue", y="Description", orientation="h",
                       color="Revenue", color_continuous_scale=["#DBEAFE", BLUE])
            f.update_layout(coloraxis_showscale=False, height=440)
            st.plotly_chart(style_fig(f, "Top 12 products by revenue", 440), use_container_width=True)

        with c2:
            ctry = (sub.groupby("Country", as_index=False)["Revenue"].sum()
                    .nlargest(12, "Revenue").sort_values("Revenue"))
            f = px.bar(ctry, x="Revenue", y="Country", orientation="h",
                       color="Revenue", color_continuous_scale=["#DCFCE7", GREEN])
            f.update_layout(coloraxis_showscale=False, height=440)
            st.plotly_chart(style_fig(f, "Top 12 countries by revenue", 440), use_container_width=True)

        section("Country mix", "proportion of revenue by market")
        ctry_all = (sub.groupby("Country", as_index=False)["Revenue"].sum()
                    .sort_values("Revenue", ascending=False))
        top10 = ctry_all.head(10).copy()
        other = ctry_all["Revenue"].iloc[10:].sum()
        if other > 0:
            top10 = pd.concat([top10, pd.DataFrame([{"Country": "Other", "Revenue": other}])],
                              ignore_index=True)
        f = px.pie(top10, names="Country", values="Revenue", hole=0.55,
                   color_discrete_sequence=px.colors.qualitative.Bold)
        f.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(style_fig(f, "Revenue share by country", 420), use_container_width=True)

        st.download_button(
            "Download top products (CSV)",
            top.sort_values("Revenue", ascending=False).to_csv(index=False).encode("utf-8"),
            file_name="retailpulse_top_products.csv", mime="text/csv",
        )
    else:
        note("Transaction-level file not found, so product and country breakdowns are hidden. "
             "Run <code>python -m src.features.build_features</code> to generate it.")
