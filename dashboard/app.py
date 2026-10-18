"""RetailPulse — Streamlit dashboard entry point.

Run from the project root:

    streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard import data as ds          # noqa: E402
from dashboard.design import apply_theme   # noqa: E402

st.set_page_config(
    page_title="RetailPulse — AI Retail Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.markdown("""
<div style="padding:10px 0 16px 0;border-bottom:1px solid rgba(255,255,255,0.12);margin-bottom:16px;">
  <div style="color:#FFFFFF;font-size:1.3rem;font-weight:800;letter-spacing:-0.01em;">
    RetailPulse
  </div>
  <div style="color:#94A3B8;font-size:0.7rem;margin-top:3px;
              text-transform:uppercase;letter-spacing:0.1em;font-weight:600;">
    AI Retail Analytics
  </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown(
    '<div style="color:#94A3B8;font-size:0.68rem;text-transform:uppercase;'
    'letter-spacing:0.1em;font-weight:700;margin-bottom:8px;">Modules</div>',
    unsafe_allow_html=True,
)

PAGES = {
    "Overview": "Executive summary",
    "Sales Analytics": "Revenue, products, countries",
    "Customer Intelligence": "Segments, RFM, churn",
    "Demand Forecasting": "Forecast and scenarios",
    "Inventory Optimisation": "EOQ, safety stock, reorder",
    "Data Quality & Monitoring": "Drift, pipeline status",
}

page = st.sidebar.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")

status = ds.pipeline_status()
ready = int((status["status"] == "ready").sum())
total = len(status)
bar_color = "#16A34A" if ready == total else "#F59E0B"

st.sidebar.markdown(f"""
<div style="margin-top:28px;padding-top:16px;border-top:1px solid rgba(255,255,255,0.10);">
  <div style="color:#94A3B8;font-size:0.66rem;text-transform:uppercase;letter-spacing:0.09em;
              font-weight:700;margin-bottom:8px;">Pipeline</div>
  <div style="color:#E2E8F0;font-size:0.82rem;font-weight:600;">{ready} / {total} stages ready</div>
  <div style="background:rgba(255,255,255,0.12);border-radius:999px;height:5px;margin-top:7px;">
    <div style="width:{ready / total * 100:.0f}%;background:{bar_color};height:5px;
                border-radius:999px;"></div>
  </div>
  <div style="color:#64748B;font-size:0.68rem;line-height:1.7;margin-top:16px;">
    <div style="color:#94A3B8;font-size:0.64rem;text-transform:uppercase;letter-spacing:0.08em;
                font-weight:700;margin-bottom:5px;">Dataset</div>
    Online Retail II<br>
    <span style="color:#CBD5E1;">UCI ML Repository</span><br>
    Dec 2009 – Dec 2011<br>
    <span style="color:#CBD5E1;">1.03M raw transactions</span>
  </div>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Overview page (inline — it is mostly composition of other outputs)
# --------------------------------------------------------------------------
def render_overview() -> None:
    from dashboard.design import hero, kpi_row, money, note, number, section

    hero(
        "RetailPulse — AI-Powered Customer Analytics &amp; Demand Forecasting",
        "An end-to-end data science platform for retail: demand forecasting, customer "
        "segmentation, churn prediction and inventory optimisation, built on the "
        "UCI Online Retail II dataset.",
    )

    daily = ds.daily_sales()
    seg = ds.customer_segments()
    churn = ds.churn_scores()
    fc = ds.forecast_daily()
    comparison = ds.model_comparison()
    metrics = ds.inventory_metrics()

    if daily is None:
        st.warning("No pipeline outputs found. Build them first:")
        st.code("python -m src.run_pipeline", language="bash")
        st.dataframe(ds.pipeline_status(), use_container_width=True, hide_index=True)
        return

    revenue = daily["Revenue"].sum()
    orders = int(daily["Orders"].sum())
    customers = len(seg) if seg is not None else 0
    forecast_total = fc["Forecast"].sum() if fc is not None else 0
    best_mape = comparison.sort_values("mape").iloc[0]["mape"] if comparison is not None else None
    high_risk = int((churn["RiskTier"] == "High Risk").sum()) if churn is not None else 0

    kpi_row([
        {"label": "Total revenue", "value": money(revenue),
         "hint": f"{daily['IsTradingDay'].sum():,.0f} trading days"},
        {"label": "Orders", "value": number(orders), "hint": f"{customers:,} customers"},
        {"label": "Forecast (next 90 days)", "value": money(forecast_total),
         "hint": f"MAPE {best_mape:.1f}%" if best_mape is not None else ""},
        {"label": "Customers at high churn risk", "value": number(high_risk),
         "delta": "retention priority" if high_risk else "", "delta_color": "#DC2626"},
    ])

    section("The four capabilities", "what this platform actually does")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("#### 📈 Demand forecasting\n"
                    "Rolling-origin backtest across four model families, with a "
                    "90% prediction interval for planning.")
    with c2:
        st.markdown("#### 👥 Customer segmentation\n"
                    "RFM quintiles into six actionable tiers, validated with "
                    "K-Means clusters and silhouette scoring.")
    with c3:
        st.markdown("#### ⚠️ Churn prediction\n"
                    "Gradient boosting on behavioural features, with AUC, lift "
                    "and a ranked retention worklist.")
    with c4:
        st.markdown("#### 📦 Inventory optimisation\n"
                    "EOQ, safety stock and reorder points per SKU, proven by a "
                    "demand-replay simulation.")

    section("Business impact", "measured, not asserted")
    if metrics is not None and {"naive", "optimised"}.issubset(set(metrics["policy"])):
        m = metrics.set_index("policy")
        fill_gain = m.loc["optimised", "fill_rate_pct"] - m.loc["naive", "fill_rate_pct"]
        lost_cut = 1 - (m.loc["optimised", "units_lost"] / max(m.loc["naive", "units_lost"], 1))
        rows = [
            ("Stockouts reduced", f"{lost_cut:.0%}", "optimised vs naive reorder policy"),
            ("Fill rate", f"{m.loc['optimised', 'fill_rate_pct']:.1f}%",
             f"{fill_gain:+.1f} points over the naive policy"),
            ("Forecast accuracy", f"{best_mape:.2f}% MAPE" if best_mape is not None else "n/a",
             "rolling-origin backtest, best model"),
            ("Customers scored", f"{customers:,}", "RFM segments and churn probability"),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["Metric", "Result", "How it was measured"]),
                     use_container_width=True, hide_index=True)

    section("How to read this dashboard", "the honest caveats")
    note(
        "• <b>Revenue is the demand proxy.</b> The dataset has no separate quantity-demanded "
        "field, and returns are removed, so forecasts represent expected sales, not "
        "unconstrained demand.<br>"
        "• <b>Profit is estimated.</b> No cost price exists in the source; a configurable 35% "
        "margin is applied. Treat profit as directional.<br>"
        "• <b>Churn is defined operationally</b> as no purchase within 90 days.<br>"
        "• <b>Every accuracy figure comes from a rolling backtest</b>, not a single train/test "
        "split, so it reflects stability across time."
    )

    section("Pipeline outputs", "files produced by the last run")
    st.dataframe(ds.pipeline_status(), use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------
if page == "Overview":
    render_overview()
elif page == "Sales Analytics":
    from dashboard.views import sales
    sales.render()
elif page == "Customer Intelligence":
    from dashboard.views import customers
    customers.render()
elif page == "Demand Forecasting":
    from dashboard.views import forecast
    forecast.render()
elif page == "Inventory Optimisation":
    from dashboard.views import inventory
    inventory.render()
elif page == "Data Quality & Monitoring":
    from dashboard.views import status as status_view
    status_view.render()

st.sidebar.markdown(
    '<div style="color:#475569;font-size:0.66rem;margin-top:22px;line-height:1.6;">'
    'Zidio Development<br>Data Science &amp; Analytics'
    '</div>', unsafe_allow_html=True)
