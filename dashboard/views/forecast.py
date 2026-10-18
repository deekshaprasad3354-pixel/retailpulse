"""Dashboard page 3 — Demand forecasting.

What we expect to sell, how confident we are, and how accurate the model has
actually been. Includes an adjustable what-if scenario so the page is a tool,
not just a report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard import data as ds
from dashboard.design import (
    AMBER, BLUE, CYAN, GREEN, MUTED, PURPLE, RED, hero, kpi_row, money, note,
    number, section, style_fig,
)


def render() -> None:
    daily = ds.require(ds.daily_sales(), "Demand forecasting", "forecasting")
    fc = ds.require(ds.forecast_daily(), "Demand forecast", "forecasting")
    weekly_fc = ds.forecast_weekly()
    comparison = ds.model_comparison()
    bt = ds.backtest()

    model_name = fc["Model"].iloc[0] if "Model" in fc.columns else "model"
    hero(
        "Demand Forecasting",
        f"Forward revenue projection from the <b>{model_name}</b> model, chosen on "
        "rolling-origin backtest accuracy rather than a single lucky split.",
    )

    horizon_days = len(fc)
    total = fc["Forecast"].sum()
    lo, hi = fc["Lower90"].sum(), fc["Upper90"].sum()
    daily_avg = fc["Forecast"].mean()

    recent_28 = daily["Revenue"].tail(28).mean()
    direction = "up" if daily_avg > recent_28 else "down"
    delta_color = GREEN if direction == "up" else RED

    kpi_row([
        {"label": f"Next {horizon_days} days", "value": money(total),
         "delta": f"{direction} vs last 28-day average", "delta_color": delta_color},
        {"label": "Average per day", "value": money(daily_avg)},
        {"label": "90% interval (low)", "value": money(lo)},
        {"label": "90% interval (high)", "value": money(hi)},
    ])

    # ---- history + forecast ------------------------------------------------
    section("Forecast", "recent actuals with the projected path and uncertainty band")
    hist = daily.tail(180)
    f = go.Figure()
    f.add_trace(go.Scatter(x=hist["Date"], y=hist["Revenue"], name="actual",
                           line=dict(color="#0F172A", width=1.4)))
    f.add_trace(go.Scatter(x=fc["Date"], y=fc["Upper90"], name="90% upper",
                           line=dict(color="rgba(37,99,235,0)"), showlegend=False))
    f.add_trace(go.Scatter(x=fc["Date"], y=fc["Lower90"], name="90% interval",
                           fill="tonexty", fillcolor="rgba(37,99,235,0.18)",
                           line=dict(color="rgba(37,99,235,0)"), showlegend=True))
    f.add_trace(go.Scatter(x=fc["Date"], y=fc["Forecast"], name=f"forecast ({model_name})",
                           line=dict(color=BLUE, width=2.2)))
    st.plotly_chart(style_fig(f, "Daily revenue — history and forecast", 430),
                    use_container_width=True)

    # ---- what-if scenario --------------------------------------------------
    section("What-if scenario", "stress-test the forecast before committing to stock")
    c1, c2, c3 = st.columns(3)
    with c1:
        uplift = st.slider("Marketing uplift on demand (%)", -30, 60, 0, step=5,
                           help="Simulate a promotion or a demand shock.")
    with c2:
        price_chg = st.slider("Price change (%)", -20, 20, 0, step=1,
                              help="Approximate: revenue moves roughly with price.")
    with c3:
        exclude_weekends = st.checkbox("Exclude weekend trading", value=False)

    adj = fc.copy()
    adj["Scenario"] = adj["Forecast"] * (1 + uplift / 100) * (1 + price_chg / 100)
    if exclude_weekends:
        weekend = adj["Date"].dt.dayofweek >= 5
        adj.loc[weekend, "Scenario"] = 0.0

    scen_total = adj["Scenario"].sum()
    diff = scen_total - total
    kpi_row([
        {"label": "Scenario revenue", "value": money(scen_total)},
        {"label": "Change vs base", "value": f"{'+' if diff >= 0 else ''}{money(diff)}",
         "delta_color": GREEN if diff >= 0 else RED},
        {"label": "Change (%)", "value": f"{(diff / total * 100) if total else 0:+.1f}%",
         "delta_color": GREEN if diff >= 0 else RED},
    ])

    f = go.Figure()
    f.add_trace(go.Scatter(x=adj["Date"], y=adj["Forecast"], name="base forecast",
                           line=dict(color=BLUE, width=2, dash="dash")))
    f.add_trace(go.Scatter(x=adj["Date"], y=adj["Scenario"], name="scenario",
                           line=dict(color=PURPLE, width=2.4), fill="tozeroy",
                           fillcolor="rgba(124,58,237,0.12)"))
    st.plotly_chart(style_fig(f, "Base forecast vs scenario", 380), use_container_width=True)

    st.download_button(
        "Download scenario (CSV)",
        adj[["Date", "Forecast", "Lower90", "Upper90", "Scenario"]].to_csv(index=False).encode("utf-8"),
        file_name="retailpulse_forecast_scenario.csv", mime="text/csv",
    )

    # ---- weekly plan -------------------------------------------------------
    if weekly_fc is not None:
        section("Weekly plan", "the same forecast aggregated for planning meetings")
        w = weekly_fc.head(12).copy()
        f = go.Figure()
        f.add_trace(go.Bar(x=w["Date"], y=w["Forecast"], name="weekly forecast",
                           marker_color=CYAN,
                           error_y=dict(type="data",
                                        array=w["Upper90"] - w["Forecast"],
                                        arrayminus=w["Forecast"] - w["Lower90"],
                                        visible=True, color=MUTED)))
        st.plotly_chart(style_fig(f, "Next 12 weeks — forecast with 90% interval", 380),
                        use_container_width=True)
        st.dataframe(
            w[["Date", "Forecast", "Lower90", "Upper90"]].style.format(
                {"Forecast": "£{:,.0f}", "Lower90": "£{:,.0f}", "Upper90": "£{:,.0f}"}),
            use_container_width=True, hide_index=True,
        )

    # ---- accuracy ----------------------------------------------------------
    section("How accurate is this?", "backtest evidence, not a claim")
    if comparison is not None:
        c1, c2 = st.columns([1.1, 1])
        with c1:
            d = comparison.sort_values("mape")
            f = px.bar(d, x="mape", y="model", orientation="h", text="mape",
                       color="mape", color_continuous_scale=["#DCFCE7", AMBER, RED])
            f.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            f.update_layout(coloraxis_showscale=False)
            f.add_vline(x=10, line_dash="dash", line_color=RED)
            st.plotly_chart(style_fig(f, "Mean absolute percentage error by model", 380),
                            use_container_width=True)
        with c2:
            st.dataframe(comparison, use_container_width=True, hide_index=True, height=340)
            best = comparison.sort_values("mape").iloc[0]
            note(f"<b>{best['model']}</b> wins with a mean MAPE of <b>{best['mape']:.2f}%</b> "
                 f"across {int(best['folds'])} backtest folds. Anything at or below 10-12% is a "
                 "usable retail forecast.")

        if bt is not None:
            with st.expander("Fold-by-fold results", expanded=False):
                f = px.line(bt, x="fold", y="mape", color="model", markers=True,
                            labels={"mape": "MAPE %", "fold": "backtest fold"})
                st.plotly_chart(style_fig(f, "Accuracy stability across folds", 380),
                                use_container_width=True)
                st.dataframe(bt, use_container_width=True, hide_index=True)

    note("Revenue is used as the demand proxy because the source data has no separate "
         "quantity-demanded field, and returns are excluded. Read forecasts as "
         "<i>expected sales revenue</i>, not unconstrained demand.")
