"""Dashboard page 4 — Inventory optimisation.

Turns the forecast into stock decisions: how much to hold, when to reorder, and
which SKUs deserve the tightest control.
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

ABC_COLORS = {"A": RED, "B": AMBER, "C": GREEN}


def render() -> None:
    policy = ds.require(ds.inventory_policy(), "Inventory optimisation", "inventory")
    metrics = ds.inventory_metrics()

    hero(
        "Inventory Optimisation",
        "Economic order quantity, safety stock and reorder points per SKU, "
        "validated by replaying real demand through a weekly-review simulation.",
    )

    # ---- headline ----------------------------------------------------------
    a_items = policy[policy["ABC"] == "A"]
    a_rev = a_items["TotalRevenue"].sum() / policy["TotalRevenue"].sum() * 100
    kpi_row([
        {"label": "SKUs analysed", "value": number(len(policy)),
         "hint": f"{len(a_items):,} class A items ({a_rev:.0f}% of revenue)"},
        {"label": "Total safety stock", "value": number(policy["SafetyStock"].sum()) + " units",
         "hint": "95% service level"},
        {"label": "Total EOQ", "value": number(policy["EOQ"].sum()) + " units",
         "hint": "cost-optimal order size"},
        {"label": "Annual ordering + holding", "value": money(policy["AnnualTotalCost"].sum()),
         "hint": "modelled cost at these settings"},
    ])

    # ---- policy comparison -------------------------------------------------
    if metrics is not None:
        section("Policy simulation", "naive reordering vs the optimised policy")
        m = metrics.set_index("policy")
        naive = m.loc["naive"] if "naive" in m.index else None
        opt = m.loc["optimised"] if "optimised" in m.index else None

        if naive is not None and opt is not None:
            gain = opt["fill_rate_pct"] - naive["fill_rate_pct"]
            kpi_row([
                {"label": "Fill rate — naive", "value": f"{naive['fill_rate_pct']:.1f}%",
                 "delta": f"{int(naive['stockout_days'])} stockout days", "delta_color": RED},
                {"label": "Fill rate — optimised", "value": f"{opt['fill_rate_pct']:.1f}%",
                 "delta": f"{gain:+.1f} points vs naive", "delta_color": GREEN},
                {"label": "Units lost — naive", "value": number(naive["units_lost"])},
                {"label": "Units lost — optimised", "value": number(opt["units_lost"]),
                 "delta_color": GREEN},
            ])

            f = go.Figure()
            for name, color in (("naive", RED), ("optimised", GREEN)):
                f.add_trace(go.Bar(name=name, x=["Fill rate %", "Service level %"],
                                   y=[m.loc[name, "fill_rate_pct"], m.loc[name, "service_level_achieved"]],
                                   marker_color=color))
            f.update_layout(barmode="group", yaxis_range=[0, 105])
            st.plotly_chart(style_fig(f, "Service performance by policy", 360),
                            use_container_width=True)

        st.dataframe(metrics, use_container_width=True, hide_index=True)

        sim = ds._csv(str(ds.cfg.PROCESSED / "inventory_simulation.csv"))
        if sim is not None:
            f = px.line(sim, x="day", y="closing_stock", color="policy",
                        color_discrete_map={"naive": RED, "optimised": GREEN},
                        labels={"closing_stock": "units on hand", "day": "simulation day"})
            st.plotly_chart(style_fig(f, "Inventory level through the simulation", 380),
                            use_container_width=True)

    # ---- ABC / XYZ ---------------------------------------------------------
    matrix = ds.abc_xyz_matrix()
    if matrix is not None:
        section("ABC / XYZ classification", "which SKUs to control tightly, and which are unpredictable")
        c1, c2 = st.columns([1, 1])
        with c1:
            pivot = (matrix.pivot_table(index="ABC", columns="XYZ", values="StockCode",
                                        aggfunc="sum", fill_value=0)
                     .reindex(index=["A", "B", "C"], columns=["X", "Y", "Z"], fill_value=0))
            f = px.imshow(pivot, text_auto=True, color_continuous_scale="YlOrRd",
                          labels=dict(x="XYZ — demand variability", y="ABC — revenue share",
                                      color="SKUs"))
            f.update_layout(coloraxis_showscale=False)
            st.plotly_chart(style_fig(f, "SKU count by ABC x XYZ", 380), use_container_width=True)
        with c2:
            rev = matrix.groupby("ABC", as_index=False)["Revenue"].sum()
            f = px.pie(rev, names="ABC", values="Revenue", hole=0.55,
                       color="ABC", color_discrete_map=ABC_COLORS)
            f.update_traces(textinfo="percent+label")
            st.plotly_chart(style_fig(f, "Revenue share by ABC class", 380),
                            use_container_width=True)

        note("<b>A-X</b> items are the sweet spot: high revenue, predictable demand. "
             "Automate their replenishment. <b>C-Z</b> items are cheap and erratic — "
             "review them quarterly, not weekly.")

    # ---- reorder worklist --------------------------------------------------
    section("Reorder worklist", "SKUs ranked by value, with their calculated parameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        abc_pick = st.multiselect("ABC class", ["A", "B", "C"], default=["A"], key="inv_abc")
    with c2:
        xyz_pick = st.multiselect("XYZ class", ["X", "Y", "Z"], default=["X", "Y", "Z"], key="inv_xyz")
    with c3:
        top_n = st.slider("Rows to show", 10, 200, 40, step=10, key="inv_n")

    view = policy
    if abc_pick:
        view = view[view["ABC"].isin(abc_pick)]
    if xyz_pick:
        view = view[view["XYZ"].isin(xyz_pick)]

    show_cols = ["StockCode", "ABC", "XYZ", "AvgDailyUnits", "StdDailyUnits", "EOQ",
                 "SafetyStock", "ReorderPoint", "DaysBetweenOrders", "UnitCost",
                 "TotalRevenue"]
    st.dataframe(view[show_cols].head(top_n), use_container_width=True, hide_index=True)

    st.download_button(
        f"Download inventory policy ({len(view):,} SKUs, CSV)",
        view.to_csv(index=False).encode("utf-8"),
        file_name="retailpulse_inventory_policy.csv", mime="text/csv",
    )

    # ---- EOQ explainer -----------------------------------------------------
    section("Why EOQ works", "the trade-off between ordering often and holding a lot")
    top = policy.nlargest(1, "TotalRevenue").iloc[0]
    q = np.linspace(max(top["EOQ"] * 0.1, 1), max(top["EOQ"] * 3, 10), 200)
    annual = max(top["AnnualDemand"], 1)
    ordering = 25.0 * annual / q
    holding = q / 2 * max(top["UnitCost"], 0.01) * 0.20

    f = go.Figure()
    f.add_trace(go.Scatter(x=q, y=ordering, name="ordering cost", line=dict(color=BLUE)))
    f.add_trace(go.Scatter(x=q, y=holding, name="holding cost", line=dict(color=RED)))
    f.add_trace(go.Scatter(x=q, y=ordering + holding, name="total cost",
                           line=dict(color="#0F172A", width=2.5)))
    f.add_vline(x=top["EOQ"], line_dash="dash", line_color=GREEN,
                annotation_text=f"EOQ {top['EOQ']:,.0f}")
    st.plotly_chart(style_fig(f, f"Cost trade-off — {top['StockCode']}", 400),
                    use_container_width=True)

    note("EOQ is where ordering cost and holding cost cross. Ordering more often raises "
         "ordering cost; ordering more at once ties up cash and warehouse space. "
         "The crossing point is the cheapest place to sit.")
