"""Stage 6 — Inventory optimisation.

Answers the question the business actually asks: *how much should we hold, and
when should we reorder?*

Applied to each SKU:

    ABC / XYZ classification - which products deserve tight control
    Economic Order Quantity  - the order size that minimises ordering + holding cost
    Safety stock             - the buffer that protects a chosen service level
    Reorder point            - the stock level that triggers a new order

Then a **weekly-review simulation** replays real daily demand under two
policies and compares them:

    Naive policy      - order what we sold last week (the intuitive default)
    Optimised policy  - order EOQ when stock falls to the reorder point

Reported outcomes: fill rate, stockout days, average inventory, holding cost.

Outputs:
    data/processed/inventory_policy.csv          per-SKU recommended parameters
    data/processed/inventory_simulation.csv      day-by-day simulation trace
    data/processed/inventory_metrics.csv         policy comparison
    data/processed/abc_xyz_matrix.csv            classification counts
    reports/figures/inventory_*.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config as cfg

# z-score for a given cycle service level
Z_TABLE = {0.80: 0.842, 0.85: 1.036, 0.90: 1.282, 0.95: 1.645, 0.975: 1.960, 0.99: 2.326}


def z_for_service_level(service_level: float = cfg.SERVICE_LEVEL) -> float:
    """Nearest tabulated z-score, or a normal approximation in between."""
    if service_level in Z_TABLE:
        return Z_TABLE[service_level]
    lo = max((k for k in Z_TABLE if k <= service_level), default=0.80)
    hi = min((k for k in Z_TABLE if k >= service_level), default=0.99)
    if lo == hi:
        return Z_TABLE[lo]
    frac = (service_level - lo) / (hi - lo)
    return Z_TABLE[lo] + frac * (Z_TABLE[hi] - Z_TABLE[lo])


# --------------------------------------------------------------------------
# Demand statistics per SKU
# --------------------------------------------------------------------------
def sku_demand_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Daily demand statistics per SKU, ready for the inventory formulas.

    Important subtlety: a SKU's demand variability must be measured across the
    **whole calendar**, including the days it did not sell. Averaging only over
    selling days makes sporadic products look far more stable than they are, and
    that understates the safety stock they actually need.
    """
    daily = (
        df.groupby(["StockCode", "Date"], as_index=False)
        .agg(Units=("Quantity", "sum"), Revenue=("Revenue", "sum"))
    )

    span_days = (df["Date"].max() - df["Date"].min()).days + 1

    # Reindex every SKU onto the full calendar so zero-demand days are explicit
    full_index = pd.MultiIndex.from_product(
        [daily["StockCode"].unique(), pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")],
        names=["StockCode", "Date"],
    )
    full = (
        daily.set_index(["StockCode", "Date"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    stats = full.groupby("StockCode").agg(
        DaysSold=("Units", lambda s: int((s > 0).sum())),
        TotalUnits=("Units", "sum"),
        TotalRevenue=("Revenue", "sum"),
        AvgDailyUnits=("Units", "mean"),
        StdDailyUnits=("Units", "std"),
    ).reset_index()

    stats["DaysInPeriod"] = span_days
    stats["ZeroDemandDays"] = span_days - stats["DaysSold"]
    stats["AvgDailyUnits"] = stats["AvgDailyUnits"].round(3)
    stats["StdDailyUnits"] = stats["StdDailyUnits"].fillna(0).round(3)
    stats["DemandCV"] = np.where(
        stats["AvgDailyUnits"] > 0, stats["StdDailyUnits"] / stats["AvgDailyUnits"], 0.0
    ).round(3)

    # Selling price and estimated unit cost (see config.DEFAULT_MARGIN)
    rev = df.groupby("StockCode", as_index=False).agg(
        TotalRevenue=("Revenue", "sum"), TotalUnits=("Quantity", "sum")
    )
    rev["UnitPrice"] = np.where(
        rev["TotalUnits"] > 0, rev["TotalRevenue"] / rev["TotalUnits"], 0.0
    ).round(2)
    stats = stats.drop(columns=["TotalRevenue"]).merge(
        rev[["StockCode", "TotalRevenue", "UnitPrice"]], on="StockCode", how="left"
    )
    stats["UnitCost"] = (stats["UnitPrice"] * (1 - cfg.DEFAULT_MARGIN)).round(2)
    return stats.sort_values("TotalRevenue", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# ABC / XYZ classification
# --------------------------------------------------------------------------
def abc_xyz_classification(stats: pd.DataFrame) -> pd.DataFrame:
    """ABC by revenue contribution, XYZ by demand variability.

    ABC uses the classic Pareto cut-off (A ≈ 80% of revenue, B ≈ next 15%).

    XYZ uses **quantile** cut-offs on the coefficient of variation rather than
    fixed numbers. Retail SKU demand is heavily intermittent — most products do
    not sell every day — so a fixed CV threshold such as 0.5 puts almost every
    SKU in "Z" and the matrix carries no information. Ranking against the SKU
    population keeps the classification discriminating.
    """
    s = stats.copy().sort_values("TotalRevenue", ascending=False)
    total = s["TotalRevenue"].sum()
    s["RevenueSharePct"] = (s["TotalRevenue"] / total * 100).round(4)
    s["CumulativeSharePct"] = s["RevenueSharePct"].cumsum().round(4)

    s["ABC"] = np.select(
        [s["CumulativeSharePct"] <= 80, s["CumulativeSharePct"] <= 95],
        ["A", "B"], default="C",
    )

    s["CV"] = np.where(s["AvgDailyUnits"] > 0, s["StdDailyUnits"] / s["AvgDailyUnits"], np.nan)
    cv = s["CV"].fillna(s["CV"].max() if s["CV"].notna().any() else 1.0)
    q33, q67 = np.nanquantile(cv, [1 / 3, 2 / 3])
    s["XYZ"] = np.select([s["CV"] <= q33, s["CV"] <= q67], ["X", "Y"], default="Z")
    s["Class"] = s["ABC"] + s["XYZ"]
    s.attrs["xyz_cutoffs"] = (round(float(q33), 3), round(float(q67), 3))
    return s.reset_index(drop=True)


# --------------------------------------------------------------------------
# Inventory formulas
# --------------------------------------------------------------------------
def economic_order_quantity(annual_demand: float, ordering_cost: float = cfg.ORDERING_COST,
                            unit_cost: float = 1.0, holding_rate: float = cfg.HOLDING_RATE) -> float:
    """Wilson EOQ: sqrt(2 * D * S / (H * C)). Balances ordering vs holding cost."""
    holding = holding_rate * max(unit_cost, 0.01)
    if annual_demand <= 0:
        return 0.0
    return float(np.sqrt(2 * annual_demand * ordering_cost / holding))


def safety_stock(std_daily: float, lead_time_days: int = cfg.LEAD_TIME_DAYS,
                 service_level: float = cfg.SERVICE_LEVEL) -> float:
    """Buffer stock = z * sigma_demand_over_lead_time."""
    z = z_for_service_level(service_level)
    return float(z * std_daily * np.sqrt(max(lead_time_days, 1)))


def reorder_point(avg_daily: float, lead_time_days: int = cfg.LEAD_TIME_DAYS,
                  std_daily: float = 0.0, service_level: float = cfg.SERVICE_LEVEL) -> float:
    """Reorder when stock on hand falls to expected lead-time demand + safety stock."""
    return float(avg_daily * lead_time_days + safety_stock(std_daily, lead_time_days, service_level))


def base_stock_level(avg_daily: float, std_daily: float,
                     lead_time_days: int = cfg.LEAD_TIME_DAYS,
                     review_days: int = cfg.REVIEW_PERIOD_DAYS,
                     service_level: float = cfg.SERVICE_LEVEL) -> float:
    """Order-up-to level for a **periodic-review** system.

    Under periodic review the exposure window is the review period *plus* the
    lead time, because stock can only be topped up once per review cycle:

        S = avg_daily * (review + lead_time) + z * std_daily * sqrt(review + lead_time)

    This is the right companion to EOQ in the simulation. An EOQ-sized order
    every review cycle would only replenish a fraction of demand, so an
    EOQ-driven simulation reports stockouts that say more about the mismatch
    than about the policy.
    """
    exposure = max(review_days, 1) + max(lead_time_days, 1)
    z = z_for_service_level(service_level)
    return float(avg_daily * exposure + z * std_daily * np.sqrt(exposure))


def build_policy(stats: pd.DataFrame) -> pd.DataFrame:
    """Attach EOQ / safety stock / reorder point to every SKU."""
    p = stats.copy()
    p["AnnualDemand"] = (p["AvgDailyUnits"] * 365).round(0)
    p["EOQ"] = [
        economic_order_quantity(d, unit_cost=c)
        for d, c in zip(p["AnnualDemand"], p["UnitCost"])
    ]
    p["EOQ"] = p["EOQ"].round(0).clip(lower=1)
    p["SafetyStock"] = [
        safety_stock(sd) for sd in p["StdDailyUnits"]
    ]
    p["ReorderPoint"] = [
        reorder_point(ad, std_daily=sd)
        for ad, sd in zip(p["AvgDailyUnits"], p["StdDailyUnits"])
    ]
    p["SafetyStock"] = p["SafetyStock"].round(0)
    p["ReorderPoint"] = p["ReorderPoint"].round(0)
    p["OrderFrequencyPerYear"] = (p["AnnualDemand"] / p["EOQ"]).round(1)
    p["DaysBetweenOrders"] = np.where(
        p["OrderFrequencyPerYear"] > 0, (365 / p["OrderFrequencyPerYear"]).round(0), np.nan
    )
    p["AnnualHoldingCost"] = (p["EOQ"] / 2 * p["UnitCost"] * cfg.HOLDING_RATE).round(2)
    p["AnnualOrderingCost"] = (p["OrderFrequencyPerYear"] * cfg.ORDERING_COST).round(2)
    p["AnnualTotalCost"] = (p["AnnualHoldingCost"] + p["AnnualOrderingCost"]).round(2)
    return p


# --------------------------------------------------------------------------
# Policy simulation
# --------------------------------------------------------------------------
def simulate_policy(demand: np.ndarray, policy: str, *, avg_daily: float, std_daily: float,
                    unit_cost: float, lead_time: int = cfg.LEAD_TIME_DAYS,
                    review: int = cfg.REVIEW_PERIOD_DAYS, opening_stock: float | None = None,
                    service_level: float = cfg.SERVICE_LEVEL) -> pd.DataFrame:
    """Replay daily demand under one replenishment policy.

    The simulation runs at the **aggregate distribution-centre level**, so the
    policy parameters and the demand series are on the same scale. (Mixing a
    per-SKU EOQ with company-wide demand is a scale error that makes the
    optimised policy look catastrophic.)

    Both policies review inventory every `review` days and receive stock
    `lead_time` days later, and both order up to a target level. They differ
    only in **how the target is set**, which isolates the value of a statistical
    level over a rule of thumb:

        naive     -> target = LAST period's demand, projected forward
        optimised -> target = base_stock_level(...) from demand mean, standard
                     deviation, lead time, review period and service level

    Comparing EOQ-sized ordering against naive ordering would not be a fair
    test: an EOQ-sized order arrives on the EOQ cycle, not the review cycle, so
    it cannot sustain demand at a shorter review interval. EOQ answers "how much
    per order", not "what level to hold".
    """
    n = len(demand)
    statistical_target = base_stock_level(avg_daily, std_daily, lead_time, review, service_level)

    if opening_stock is None:
        opening = statistical_target
    else:
        opening = opening_stock

    stock = float(opening)
    on_order: list[tuple[int, float]] = []
    rows = []
    lost_units = 0
    orders_placed = 0
    units_ordered = 0.0

    for day in range(n):
        # receive anything scheduled for today
        arriving = sum(q for d, q in on_order if d == day)
        stock += arriving
        on_order = [(d, q) for d, q in on_order if d != day]

        d_today = float(demand[day])
        sold = min(stock, d_today)
        stock -= sold
        lost = d_today - sold
        lost_units += lost

        if day % review == 0:
            position = stock + sum(q for _, q in on_order)
            if policy == "optimised":
                target = statistical_target
            else:
                # naive: project the previous period's actual demand forward
                window = demand[max(day - review, 0):day]
                per_day = float(np.mean(window)) if len(window) else avg_daily
                target = per_day * (review + lead_time)

            order_qty = max(target - position, 0.0)
            if order_qty > 0:
                on_order.append((day + lead_time, order_qty))
                orders_placed += 1
                units_ordered += order_qty

        rows.append({
            "day": day, "demand": d_today, "sold": sold, "lost": lost,
            "closing_stock": stock, "on_order": sum(q for _, q in on_order),
        })

    sim = pd.DataFrame(rows)
    sim["policy"] = policy
    sim.attrs.update(lost_units=lost_units, orders_placed=orders_placed,
                     units_ordered=units_ordered)
    return sim


def simulation_metrics(sim: pd.DataFrame, unit_cost: float = 1.0,
                       ordering_cost: float = cfg.ORDERING_COST) -> dict:
    """Summarise one simulation run.

    Note on `service_level_achieved`: a cycle is counted as *served* when at
    least 95% of that day's demand was met. Requiring zero lost units would be
    unrealistic at distribution-centre scale, where day-to-day demand swings by
    tens of thousands of units.
    """
    demand = sim["demand"].sum()
    sold = sim["sold"].sum()
    lost = sim["lost"].sum()
    served = (sim["lost"] <= 0.05 * sim["demand"]).mean()
    position = sim["closing_stock"] + sim["on_order"]

    return {
        "fill_rate_pct": round(sold / demand * 100, 2) if demand > 0 else 100.0,
        "stockout_days": int((sim["lost"] > 0).sum()),
        "total_days": int(len(sim)),
        "units_demanded": round(demand, 0),
        "units_lost": round(lost, 0),
        "orders_placed": int(sim.attrs.get("orders_placed", 0)),
        "units_ordered": round(float(sim.attrs.get("units_ordered", 0.0)), 0),
        "avg_closing_stock": round(sim["closing_stock"].mean(), 1),
        "avg_inventory_position": round(position.mean(), 1),
        "avg_holding_value": round(sim["closing_stock"].mean() * unit_cost, 2),
        "annual_holding_cost": round(sim["closing_stock"].mean() * unit_cost * cfg.HOLDING_RATE * 365 / max(len(sim), 1), 2),
        "annual_ordering_cost": round(sim.attrs.get("orders_placed", 0) * ordering_cost * 365 / max(len(sim), 1), 2),
        "service_level_achieved": round(float(served) * 100, 2),
    }


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def _make_svg_figures(policy: pd.DataFrame, matrix: pd.DataFrame, metrics: pd.DataFrame,
                      sims: dict[str, pd.DataFrame]) -> None:
    """Dependency-free SVG fallback for the inventory charts."""
    from src.viz import bar_chart, grouped_bar_chart, heatmap, line_chart, save

    # ABC x XYZ matrix
    pivot = (matrix.pivot_table(index="ABC", columns="XYZ", values="StockCode",
                                aggfunc="sum", fill_value=0)
             .reindex(index=["A", "B", "C"], columns=["X", "Y", "Z"], fill_value=0))
    save(heatmap(pivot.index.tolist(), pivot.columns.tolist(),
                 pivot.to_numpy().tolist(),
                 "ABC x XYZ SKU matrix (counts)"),
         cfg.FIGURES / "inventory_abc_xyz.svg")

    # policy comparison
    m = metrics.set_index("policy")
    groups = {}
    for col, label in (("fill_rate_pct", "Fill rate %"),
                       ("service_level_achieved", "Service level %")):
        groups[label] = [float(m.loc[p, col]) for p in m.index]
    save(grouped_bar_chart(list(m.index), groups,
                           "Simulated service performance by policy",
                           value_suffix="%"),
         cfg.FIGURES / "inventory_policy_comparison.svg")

    # inventory trace
    series = {f"{name} closing stock": (s["day"].tolist(), s["closing_stock"].tolist())
              for name, s in sims.items()}
    save(line_chart(series, "Inventory level through the simulation window",
                    x_label="day", y_label="units on hand"),
         cfg.FIGURES / "inventory_simulation_trace.svg")

    # EOQ trade-off for the top SKU
    top = policy.iloc[0]
    q = np.linspace(max(top["EOQ"] * 0.1, 1), max(top["EOQ"] * 3, 10), 120)
    annual = max(top["AnnualDemand"], 1)
    ordering = cfg.ORDERING_COST * annual / q
    holding = q / 2 * max(top["UnitCost"], 0.01) * cfg.HOLDING_RATE
    save(line_chart(
        {"ordering cost": (q.tolist(), ordering.tolist()),
         "holding cost": (q.tolist(), holding.tolist()),
         "total cost": (q.tolist(), (ordering + holding).tolist())},
        f"EOQ trade-off — top SKU {top['StockCode']}",
        x_label="order quantity (units)", y_label="annual cost (GBP)"),
        cfg.FIGURES / "inventory_eoq_curve.svg")


def make_figures(policy: pd.DataFrame, matrix: pd.DataFrame, metrics: pd.DataFrame,
                 sims: dict[str, pd.DataFrame]) -> None:
    from src.optional_deps import MATPLOTLIB

    if not MATPLOTLIB.ok:
        _make_svg_figures(policy, matrix, metrics, sims)
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")

    # 1. ABC/XYZ matrix heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    pivot = (matrix.pivot_table(index="ABC", columns="XYZ", values="StockCode",
                                aggfunc="count", fill_value=0)
             .reindex(index=["A", "B", "C"], columns=["X", "Y", "Z"], fill_value=0))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", ax=axes[0], cbar=False)
    axes[0].set_title("ABC x XYZ SKU matrix (counts)", fontsize=12, weight="bold")
    axes[0].set_xlabel("XYZ — demand variability (X stable)")
    axes[0].set_ylabel("ABC — revenue contribution (A highest)")

    # 2. policy comparison
    m = metrics.set_index("policy")
    comp = m[["fill_rate_pct", "service_level_achieved"]].T
    comp.plot(kind="bar", ax=axes[1], color=["#dc2626", "#16a34a"], rot=0)
    axes[1].set_title("Simulated service performance by policy", fontsize=12, weight="bold")
    axes[1].set_ylabel("%"); axes[1].set_xlabel("")
    axes[1].legend(title="policy", fontsize=8)
    axes[1].tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(cfg.FIGURES / "inventory_abc_xyz_and_policy.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # 3. inventory trace
    fig, ax = plt.subplots(figsize=(14, 5))
    for name, s in sims.items():
        ax.plot(s["day"], s["closing_stock"], lw=1.2, label=f"{name} — closing stock")
    ax.set_title("Inventory level through the simulation window", fontsize=12, weight="bold")
    ax.set_xlabel("day"); ax.set_ylabel("units on hand"); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(cfg.FIGURES / "inventory_simulation_trace.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # 4. EOQ curve for the top SKU
    fig, ax = plt.subplots(figsize=(9, 4.8))
    top = policy.iloc[0]
    q = np.linspace(max(top["EOQ"] * 0.1, 1), max(top["EOQ"] * 3, 10), 200)
    d_annual = max(top["AnnualDemand"], 1)
    ordering = cfg.ORDERING_COST * d_annual / q
    holding = q / 2 * max(top["UnitCost"], 0.01) * cfg.HOLDING_RATE
    ax.plot(q, ordering, label="ordering cost", color="#2563eb")
    ax.plot(q, holding, label="holding cost", color="#dc2626")
    ax.plot(q, ordering + holding, label="total cost", color="#0f172a", lw=2)
    ax.axvline(top["EOQ"], color="#16a34a", ls="--", label=f"EOQ = {top['EOQ']:.0f}")
    ax.set_xlabel("order quantity (units)"); ax.set_ylabel("annual cost (GBP)")
    ax.set_title(f"EOQ trade-off — top SKU {top['StockCode']}", fontsize=12, weight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(cfg.FIGURES / "inventory_eoq_curve.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("=" * 78)
    print("RetailPulse :: Stage 6 - inventory optimisation")
    print("=" * 78)

    clean = cfg.read_cache(cfg.CLEAN_CACHE)
    clean["Date"] = pd.to_datetime(clean["Date"])
    print(f"[inventory] transactions: {len(clean):,}")

    stats = sku_demand_stats(clean)
    classified = abc_xyz_classification(stats)
    policy = build_policy(classified)
    print(f"[inventory] SKUs analysed: {len(policy):,}")

    print("\n--- ABC x XYZ distribution ---")
    matrix = policy.groupby(["ABC", "XYZ"], as_index=False).agg(
        StockCode=("StockCode", "count"), Revenue=("TotalRevenue", "sum")
    )
    print(matrix.to_string(index=False))

    print("\n--- policy parameters: top 8 SKUs by revenue ---")
    cols = ["StockCode", "ABC", "XYZ", "AvgDailyUnits", "StdDailyUnits", "EOQ",
            "SafetyStock", "ReorderPoint", "DaysBetweenOrders"]
    print(policy[cols].head(8).to_string(index=False))

    # --- simulate on aggregate daily demand ---------------------------------
    daily = clean.groupby("Date", as_index=False)["Quantity"].sum().sort_values("Date")
    demand = daily["Quantity"].to_numpy(float)
    avg_daily, std_daily = float(demand.mean()), float(demand.std())
    agg_cost = float((clean["Revenue"].sum() / max(clean["Quantity"].sum(), 1)) * (1 - cfg.DEFAULT_MARGIN))

    print(f"\n[inventory] aggregate demand: {demand.sum():,.0f} units over {len(demand)} days "
          f"(mean {avg_daily:,.0f}/day)")

    sims, rows = {}, []
    for pol in ("naive", "optimised"):
        sim = simulate_policy(demand, pol, avg_daily=avg_daily, std_daily=std_daily, unit_cost=agg_cost)
        m = simulation_metrics(sim, unit_cost=agg_cost)
        m["policy"] = pol
        rows.append(m)
        sims[pol] = sim

    metrics = pd.DataFrame(rows)[
        ["policy", "fill_rate_pct", "stockout_days", "total_days", "units_lost",
         "orders_placed", "avg_closing_stock", "avg_holding_value",
         "annual_holding_cost", "annual_ordering_cost", "service_level_achieved"]
    ]
    metrics["annual_total_cost"] = (metrics["annual_holding_cost"] + metrics["annual_ordering_cost"]).round(2)

    print("\n--- policy simulation results ---")
    print(metrics.to_string(index=False))

    naive = metrics[metrics["policy"] == "naive"].iloc[0]
    opt = metrics[metrics["policy"] == "optimised"].iloc[0]

    print("\n[inventory] comparison:")
    print(f"  fill rate          naive {naive['fill_rate_pct']:6.2f}%   "
          f"optimised {opt['fill_rate_pct']:6.2f}%   "
          f"({opt['fill_rate_pct'] - naive['fill_rate_pct']:+.2f} points)")
    print(f"  units lost         naive {naive['units_lost']:10,.0f}   "
          f"optimised {opt['units_lost']:10,.0f}")
    print(f"  orders placed      naive {naive['orders_placed']:10,.0f}   "
          f"optimised {opt['orders_placed']:10,.0f}")
    print(f"  avg stock on hand  naive {naive['avg_closing_stock']:10,.0f}   "
          f"optimised {opt['avg_closing_stock']:10,.0f}")
    print(f"  annual total cost  naive {cfg.CURRENCY_SYMBOL}{naive['annual_total_cost']:10,.0f}   "
          f"optimised {cfg.CURRENCY_SYMBOL}{opt['annual_total_cost']:10,.0f}")

    cost_delta = naive["annual_total_cost"] - opt["annual_total_cost"]
    stock_delta = opt["avg_closing_stock"] - naive["avg_closing_stock"]
    lost_delta = naive["units_lost"] - opt["units_lost"]

    if cost_delta > 0:
        print(f"\n[inventory] the optimised policy is cheaper by "
              f"{cfg.CURRENCY_SYMBOL}{cost_delta:,.0f}/year and holds "
              f"{abs(stock_delta):,.0f} {'more' if stock_delta > 0 else 'fewer'} units")
    else:
        print(f"\n[inventory] the optimised policy carries "
              f"{cfg.CURRENCY_SYMBOL}{-cost_delta:,.0f}/year more holding cost, and in exchange "
              f"serves {lost_delta:,.0f} more units "
              f"({lost_delta / max(naive['units_lost'], 1) * 100:.0f}% fewer units lost)")

    # Translate the service gain into money so the trade-off is concrete.
    unit_price = float(clean["Revenue"].sum() / max(clean["Quantity"].sum(), 1))
    recovered_value = lost_delta * unit_price
    print(f"[inventory] recovered units are worth roughly "
          f"{cfg.CURRENCY_SYMBOL}{recovered_value:,.0f} in revenue at the average selling price "
          f"of {cfg.CURRENCY_SYMBOL}{unit_price:.2f}/unit")

    cutoffs = classified.attrs.get("xyz_cutoffs")
    if cutoffs:
        print(f"[inventory] XYZ cut-offs on the coefficient of variation: "
              f"X <= {cutoffs[0]}, Y <= {cutoffs[1]}, Z above")

    policy.to_csv(cfg.PROCESSED / "inventory_policy.csv", index=False)
    metrics.to_csv(cfg.PROCESSED / "inventory_metrics.csv", index=False)
    matrix.to_csv(cfg.PROCESSED / "abc_xyz_matrix.csv", index=False)
    pd.concat(sims.values(), ignore_index=True).to_csv(
        cfg.PROCESSED / "inventory_simulation.csv", index=False)

    make_figures(policy, matrix, metrics, sims)
    print(f"[inventory] wrote outputs -> {cfg.PROCESSED}")


if __name__ == "__main__":
    main()
