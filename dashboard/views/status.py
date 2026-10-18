"""Dashboard page 5 — Data quality, drift and pipeline status.

The page that answers "can I trust what I just saw?" before anyone acts on a
number from the other four pages.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard import data as ds
from dashboard.design import (
    AMBER, BLUE, CYAN, GREEN, MUTED, RED, hero, kpi_row, note, number, section,
    style_fig,
)


def render() -> None:
    hero(
        "Data Quality & Monitoring",
        "Pipeline health, data-quality funnel and drift detection — the evidence "
        "that the numbers on the other pages can be trusted.",
    )

    # ---- pipeline status ---------------------------------------------------
    section("Pipeline status", "which stages have produced their outputs")
    status = ds.pipeline_status()
    ready = (status["status"] == "ready").sum()
    kpi_row([
        {"label": "Stages ready", "value": f"{ready} / {len(status)}",
         "delta_color": GREEN if ready == len(status) else AMBER},
        {"label": "Processed files", "value": str(len(list(ds.cfg.PROCESSED.glob('*'))))},
        {"label": "Figures generated", "value": str(len(list(ds.cfg.FIGURES.glob('*.png'))))},
    ])
    st.dataframe(status, use_container_width=True, hide_index=True)

    if ready < len(status):
        note("Some stages have not been run yet. "
             "Run <code>python -m src.run_pipeline</code> from the project root to build everything.")

    # ---- cleaning funnel ---------------------------------------------------
    section("Data-quality funnel", "how 1.07M raw rows became the analysis dataset")
    funnel = ds._csv(str(ds.cfg.REPORTS / "cleaning_funnel.csv"))
    if funnel is not None:
        f = go.Figure(go.Bar(
            x=funnel["rows_remaining"], y=funnel["step"], orientation="h",
            marker_color=[BLUE if i == 0 else CYAN for i in range(len(funnel))],
            text=funnel["rows_remaining"].map(lambda v: f"{v:,.0f}"),
            textposition="outside",
        ))
        f.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="rows remaining")
        st.plotly_chart(style_fig(f, "Rows surviving each cleaning rule", 430),
                        use_container_width=True)

        raw = funnel["rows_remaining"].iloc[0]
        clean = funnel["rows_remaining"].iloc[-1]
        kpi_row([
            {"label": "Raw rows", "value": number(raw)},
            {"label": "Clean rows", "value": number(clean)},
            {"label": "Rows removed", "value": number(raw - clean),
             "hint": f"{(raw - clean) / raw:.1%} of raw data"},
            {"label": "Retention", "value": f"{clean / raw:.1%}", "delta_color": GREEN},
        ])
    else:
        note("Cleaning funnel not found. Run <code>python -m src.features.build_features</code>.")

    # ---- drift -------------------------------------------------------------
    section("Drift detection", "has the underlying data distribution moved?")
    drift = ds.drift_report()
    if drift is not None:
        major = drift[drift["psi"] >= 0.25]
        moderate = drift[(drift["psi"] >= 0.10) & (drift["psi"] < 0.25)]

        kpi_row([
            {"label": "Features monitored", "value": str(len(drift))},
            {"label": "Major shift (PSI > 0.25)", "value": str(len(major)),
             "delta": "retrain recommended" if len(major) else "none",
             "delta_color": RED if len(major) else GREEN},
            {"label": "Moderate shift", "value": str(len(moderate)),
             "delta_color": AMBER if len(moderate) else GREEN},
            {"label": "Stable", "value": str(len(drift) - len(major) - len(moderate)),
             "delta_color": GREEN},
        ])

        d = drift.sort_values("psi")
        colors = [GREEN if v < 0.10 else AMBER if v < 0.25 else RED for v in d["psi"]]
        f = go.Figure(go.Bar(x=d["psi"], y=d["feature"], orientation="h", marker_color=colors,
                             text=d["psi"].map(lambda v: f"{v:.3f}"), textposition="outside"))
        f.add_vline(x=0.10, line_dash="dash", line_color=AMBER)
        f.add_vline(x=0.25, line_dash="dash", line_color=RED)
        st.plotly_chart(style_fig(f, "Population Stability Index by feature", 400),
                        use_container_width=True)

        st.dataframe(drift, use_container_width=True, hide_index=True)

        if len(major):
            note(f"<b>Action required.</b> {len(major)} feature(s) drifted beyond the 0.25 "
                 "threshold: " + ", ".join(major["feature"].tolist()) +
                 ". Distribution shift of this size means the model is being asked to predict "
                 "a world it was not trained on — schedule retraining.")
        else:
            note("<b>No action required.</b> All features sit below the moderate-shift "
                 "threshold, so the current models remain valid. Keep monitoring.")
    else:
        note("Drift report not found. Run <code>python -m src.run_pipeline --only drift</code>.")

    # ---- data dictionary ---------------------------------------------------
    section("Data dictionary", "what each column in the modelling tables means")
    dictionary = pd.DataFrame([
        ("Invoice", "Invoice number. A leading 'C' marks a cancellation."),
        ("StockCode", "Product code."),
        ("Description", "Product name."),
        ("Quantity", "Units on the line. Negative would mean a return."),
        ("InvoiceDate", "Timestamp the invoice was raised."),
        ("Price", "Unit price in GBP."),
        ("CustomerID", "Customer identifier. Blank in the raw data for guest/unattributed orders."),
        ("Country", "Customer's country."),
        ("Revenue", "Quantity x Price."),
        ("EstimatedCost", "Revenue x (1 - margin). The source has no cost column, so a "
                          "configurable 35% gross margin is applied."),
        ("EstimatedProfit", "Revenue - EstimatedCost."),
        ("Recency", "Days since the customer's last purchase."),
        ("Frequency", "Distinct invoices for the customer."),
        ("Monetary", "Total revenue from the customer."),
        ("R/F/M_Score", "Quintile score 1-5 (5 is best). Recency is inverted."),
        ("RFM_Score", "R + F + M, ranging 3-15."),
        ("Segment", "Named RFM tier, e.g. Champions or At Risk."),
        ("ChurnProbability", "Model's estimated probability of no purchase in the next 90 days."),
        ("RiskTier", "Low / Medium / High risk bucket from ChurnProbability."),
        ("EOQ", "Economic order quantity: the cost-minimising order size."),
        ("SafetyStock", "Buffer stock for the chosen service level."),
        ("ReorderPoint", "Stock level that triggers a replenishment order."),
    ], columns=["column", "meaning"])
    st.dataframe(dictionary, use_container_width=True, hide_index=True, height=420)

    note("The 35% margin is an <b>assumption</b>, not source data. Profit figures are therefore "
         "directional. Replace <code>DEFAULT_MARGIN</code> in <code>src/config.py</code> with a "
         "real cost file to make them exact.")
