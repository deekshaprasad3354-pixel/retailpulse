"""Dashboard page 2 — Customer intelligence.

Who our customers are, which segments carry the revenue, and who is about to
leave. Combines the RFM segmentation and the churn model into one story.
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

SEGMENT_COLORS = {
    "Champions": GREEN,
    "Loyal Customers": BLUE,
    "New / Promising": CYAN,
    "Need Attention": AMBER,
    "At Risk": "#EA580C",
    "Dormant": RED,
}


def render() -> None:
    seg = ds.require(ds.customer_segments(), "Customer segmentation", "segmentation")
    prof = ds.segment_profiles()
    churn = ds.churn_scores()

    hero(
        "Customer Intelligence",
        "RFM segmentation, K-Means clusters and churn risk in one view — "
        "so retention effort goes to the customers who are actually worth keeping.",
    )

    # ---- headline numbers --------------------------------------------------
    total_customers = len(seg)
    total_revenue = seg["Monetary"].sum()
    champions = prof.loc[prof["Segment"] == "Champions", "Customers"].sum() if prof is not None else 0
    champions_rev = prof.loc[prof["Segment"] == "Champions", "RevenueSharePct"].sum() if prof is not None else 0
    avg_orders = seg["Frequency"].mean()

    kpi_row([
        {"label": "Customers", "value": number(total_customers), "hint": "with an identifiable ID"},
        {"label": "Customer revenue", "value": money(total_revenue), "hint": "attributable to known customers"},
        {"label": "Champions", "value": number(champions),
         "hint": f"{champions_rev:.1f}% of revenue", "delta_color": GREEN},
        {"label": "Avg orders / customer", "value": f"{avg_orders:.1f}", "hint": "over the full period"},
    ])

    # ---- segment profile ---------------------------------------------------
    section("RFM segments", "recency, frequency and monetary value combined into named tiers")
    if prof is not None:
        c1, c2 = st.columns([1, 1])
        with c1:
            d = prof.sort_values("Customers", ascending=True)
            f = px.bar(d, x="Customers", y="Segment", orientation="h",
                       color="Segment", color_discrete_map=SEGMENT_COLORS,
                       text="CustomerSharePct")
            f.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            f.update_layout(showlegend=False)
            st.plotly_chart(style_fig(f, "Customers per segment", 380), use_container_width=True)
        with c2:
            d = prof.sort_values("RevenueSharePct", ascending=True)
            f = px.bar(d, x="RevenueSharePct", y="Segment", orientation="h",
                       color="Segment", color_discrete_map=SEGMENT_COLORS,
                       text="RevenueSharePct")
            f.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            f.update_layout(showlegend=False)
            st.plotly_chart(style_fig(f, "Revenue share per segment", 380), use_container_width=True)

        # the 80/20 picture
        sec = prof.sort_values("RevenueSharePct", ascending=False)
        f = go.Figure()
        f.add_trace(go.Bar(x=sec["Segment"], y=sec["CustomerSharePct"], name="% of customers",
                           marker_color=CYAN))
        f.add_trace(go.Bar(x=sec["Segment"], y=sec["RevenueSharePct"], name="% of revenue",
                           marker_color=BLUE))
        f.update_layout(barmode="group")
        st.plotly_chart(style_fig(f, "Customers vs revenue share — where the value concentrates", 380),
                        use_container_width=True)

        with st.expander("Segment profiles and recommended actions", expanded=False):
            show = prof[["Segment", "Customers", "CustomerSharePct", "RevenueSharePct",
                         "AvgRecency", "AvgFrequency", "AvgMonetary", "RecommendedAction"]]
            st.dataframe(show, use_container_width=True, hide_index=True)

    # ---- RFM distribution --------------------------------------------------
    section("Customer value distribution", "the long tail every retailer has")
    c1, c2 = st.columns(2)
    with c1:
        f = px.histogram(seg, x="Monetary", nbins=60, log_y=True,
                         labels={"Monetary": "total revenue per customer (£)"},
                         color_discrete_sequence=[BLUE])
        st.plotly_chart(style_fig(f, "Revenue per customer (log scale)", 350),
                        use_container_width=True)
    with c2:
        f = px.scatter(seg.sample(min(len(seg), 4000), random_state=42),
                       x="Recency", y="Monetary", color="Segment",
                       color_discrete_map=SEGMENT_COLORS, opacity=0.6,
                       labels={"Recency": "days since last purchase", "Monetary": "revenue (£)"})
        f.update_traces(marker=dict(size=5))
        st.plotly_chart(style_fig(f, "Recency vs monetary value", 350), use_container_width=True)

    # ---- K-Means clusters --------------------------------------------------
    cprof = ds.cluster_profiles()
    if cprof is not None:
        section("K-Means clusters", "an unsupervised view — the data groups itself")
        f = px.scatter_3d(
            seg, x="Recency", y="Frequency", z="Monetary", color="ClusterName",
            opacity=0.5, labels={"ClusterName": "cluster"},
        )
        f.update_traces(marker=dict(size=3))
        st.plotly_chart(style_fig(f, "3D view of the customer clusters", 520),
                        use_container_width=True)
        st.dataframe(cprof, use_container_width=True, hide_index=True)

    # ---- churn -------------------------------------------------------------
    if churn is not None:
        section("Churn risk", "who is likely to stop buying in the next 90 days")
        tiers = (churn["RiskTier"].value_counts()
                 .reindex(["Low Risk", "Medium Risk", "High Risk"]).fillna(0))
        high = churn[churn["RiskTier"] == "High Risk"]
        high_value = high["Monetary"].sum()

        kpi_row([
            {"label": "High-risk customers", "value": number(tiers["High Risk"]),
             "hint": f"{tiers['High Risk'] / max(len(churn), 1):.1%} of the base",
             "delta": "needs action", "delta_color": RED},
            {"label": "Revenue at risk", "value": money(high_value),
             "hint": "historic revenue from high-risk customers", "delta_color": RED},
            {"label": "Average recency (high risk)", "value": f"{high['Recency'].mean():.0f} days",
             "hint": "since their last order"},
        ])

        c1, c2 = st.columns(2)
        with c1:
            f = px.bar(x=tiers.index, y=tiers.values,
                       color=tiers.index,
                       color_discrete_map={"Low Risk": GREEN, "Medium Risk": AMBER, "High Risk": RED},
                       text=tiers.values)
            f.update_traces(textposition="outside")
            f.update_layout(showlegend=False, xaxis_title="", yaxis_title="customers")
            st.plotly_chart(style_fig(f, "Customers by churn risk tier", 360), use_container_width=True)
        with c2:
            f = px.histogram(churn, x="ChurnProbability", color="RiskTier", nbins=40,
                             color_discrete_map={"Low Risk": GREEN, "Medium Risk": AMBER,
                                                 "High Risk": RED},
                             labels={"ChurnProbability": "predicted churn probability"})
            st.plotly_chart(style_fig(f, "Distribution of churn probability", 360),
                            use_container_width=True)

        imp = ds.churn_importance()
        if imp is not None:
            gb = imp[imp["model"] == "Gradient Boosting"].nlargest(10, "importance")
            f = px.bar(gb, x="importance", y="feature", orientation="h",
                       color="importance", color_continuous_scale=["#FEE2E2", RED])
            f.update_layout(coloraxis_showscale=False)
            st.plotly_chart(style_fig(f, "What drives churn — feature importance", 380),
                            use_container_width=True)

        cmp_df = ds.churn_comparison()
        if cmp_df is not None:
            with st.expander("Model comparison", expanded=False):
                st.dataframe(cmp_df, use_container_width=True, hide_index=True)

        # actionable list — the output a retention team actually uses
        section("Retention worklist", "highest-value customers who are at risk")
        if "Segment" not in high.columns and seg is not None and "Segment" in seg.columns:
            high = high.merge(seg[["CustomerID", "Segment"]], on="CustomerID", how="left")

        worklist_cols = [c for c in ["CustomerID", "Recency", "Frequency", "Monetary",
                                     "ChurnProbability", "RiskTier", "Segment"]
                         if c in high.columns]
        worklist = (high.sort_values("Monetary", ascending=False)
                    [worklist_cols]
                    .head(200))
        st.dataframe(worklist, use_container_width=True, hide_index=True, height=340)
        st.download_button(
            "Download full churn worklist (CSV)",
            high.sort_values("ChurnProbability", ascending=False).to_csv(index=False).encode("utf-8"),
            file_name="retailpulse_churn_worklist.csv", mime="text/csv",
        )

        note("Prioritise by <b>Monetary x ChurnProbability</b> (value at risk), not by risk alone — "
             "a 95% churn probability on a £30 customer is not worth a phone call.")
    else:
        note("Churn scores not found. Run <code>python -m src.run_pipeline --only churn</code>.")

    # ---- segment explorer --------------------------------------------------
    section("Segment explorer", "inspect the customers inside any segment")
    # present the tiers in value order rather than alphabetically — the ordering
    # is the whole point of a tiered segmentation
    present = set(seg["Segment"].dropna().unique())
    ordered = [s for s in SEGMENT_COLORS if s in present]
    ordered += [s for s in sorted(present) if s not in ordered]
    choice = st.selectbox("Segment", ordered, key="seg_pick")
    subset = seg[seg["Segment"] == choice]
    st.caption(f"{len(subset):,} customers · "
               f"average revenue {money(subset['Monetary'].mean())} · "
               f"average recency {subset['Recency'].mean():.0f} days")
    st.dataframe(
        subset[["CustomerID", "Recency", "Frequency", "Monetary", "R_Score",
                "F_Score", "M_Score", "RFM_Score", "RecommendedAction"]].head(300),
        use_container_width=True, hide_index=True,
    )
