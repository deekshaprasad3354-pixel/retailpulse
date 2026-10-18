# RetailPulse — Power BI Build Guide (12 pages)

The brief asks for a 12-page interactive Power BI report alongside the Streamlit
dashboard. This guide takes you from the raw CSV to a finished report, with the
DAX measures written out.

---

## Step 0 — What you will connect to

Run the pipeline first, then point Power BI at these files in
`data/processed/`:

| File | Grain | Use it for |
|---|---|---|
| `daily_sales_features.csv` | one row per day (739) | all time-series pages |
| `weekly_sales_features.csv` | one row per week (106) | weekly trend pages |
| `customer_segments.csv` | one row per customer (5,852) | customer analytics pages |
| `segment_profiles.csv` | one row per segment (6) | segment summary visuals |
| `customer_churn.csv` | one row per customer | churn pages |
| `inventory_policy.csv` | one row per SKU (4,620) | inventory pages |
| `inventory_metrics.csv` | one row per policy | policy comparison |
| `drift_report.csv` | one row per feature | monitoring page |

For product- and country-level pages you also need line-item data. Either load
`data/interim/transactions_clean.*` (large, ~777k rows) or run a small export:

```python
import pandas as pd
from src import config as cfg
tx = cfg.read_cache(cfg.CLEAN_CACHE)
tx.to_csv(cfg.REPORTS / "powerbi_lineitems.csv",
          columns=["Invoice","StockCode","Description","Quantity","Date","Price",
                   "CustomerID","Country","Revenue","EstimatedProfit"],
          index=False)
```

---

## Step 1 — Load into Power BI

1. **Home → Get data → Text/CSV** for each file.
2. In the preview click **Transform Data** (not *Load*) so you can fix types.
3. Set column types:
   - `Date` → **Date**
   - `CustomerID`, `StockCode` → **Text** (identifiers, not numbers to sum)
   - `Revenue`, `Monetary`, `Price`, `UnitCost` → **Decimal Number**
   - `ChurnProbability`, `CV`, `psi` → **Decimal Number**
   - `Orders`, `Frequency`, `Quantity`, `EOQ`, `SafetyStock` → **Whole Number**

> **Why CustomerID as Text:** Power BI will otherwise sum customer IDs, which
> is meaningless. This is the single most common mistake with this dataset.

---

## Step 2 — Build the Date table

A dedicated date table makes every time-intelligence measure work correctly.

**Modeling → New table:**

```dax
DateTable =
VAR MinDate = MIN ( daily_sales_features[Date] )
VAR MaxDate = DATE ( 2011, 12, 31 )
RETURN
ADDCOLUMNS (
    CALENDAR ( MinDate, MaxDate ),
    "Year",        YEAR ( [Date] ),
    "Quarter",     "Q" & QUARTER ( [Date] ),
    "MonthNo",     MONTH ( [Date] ),
    "MonthName",   FORMAT ( [Date], "MMM" ),
    "MonthYear",   FORMAT ( [Date], "MMM YYYY" ),
    "WeekNo",      WEEKNUM ( [Date], 2 ),
    "DayName",     FORMAT ( [Date], "DDD" ),
    "DayOfWeek",   WEEKDAY ( [Date], 2 ),
    "IsWeekend",   WEEKDAY ( [Date], 2 ) >= 6,
    "YearMonthKey", YEAR ( [Date] ) * 100 + MONTH ( [Date] )
)
```

Then:
1. **Mark as date table** → Modeling → Mark as Date Table → `[Date]`.
2. **Create the relationship**: `DateTable[Date]` (1) → `daily_sales_features[Date]` (*).
3. Repeat for `weekly_sales_features`, `customer_churn` (via a date bridge if
   needed) and any other dated table.
4. **Sort `MonthName` by `MonthNo`** (Column tools → Sort by column) so months
   appear in calendar order instead of alphabetical.

---

## Step 3 — DAX measure library

Create a **Measures** table (Enter Data → name it `_Measures`) and put every
measure there. Grouping measures in one table keeps the field list usable.

### Core sales

```dax
Total Revenue = SUM ( daily_sales_features[Revenue] )

Total Units = SUM ( daily_sales_features[Units] )

Total Orders = SUM ( daily_sales_features[Orders] )

Total Customers = SUM ( daily_sales_features[Customers] )

Trading Days = SUMX ( daily_sales_features, IF ( daily_sales_features[IsTradingDay], 1, 0 ) )

Average Order Value = DIVIDE ( [Total Revenue], [Total Orders] )

Revenue per Trading Day = DIVIDE ( [Total Revenue], [Trading Days] )

Revenue per Customer = DIVIDE ( [Total Revenue], [Total Customers] )
```

### Time intelligence

```dax
Revenue PY =
CALCULATE ( [Total Revenue], SAMEPERIODLASTYEAR ( DateTable[Date] ) )

Revenue YoY % =
VAR Prev = [Revenue PY]
RETURN DIVIDE ( [Total Revenue] - Prev, Prev )

Revenue MoM % =
VAR Prev = CALCULATE ( [Total Revenue], DATEADD ( DateTable[Date], -1, MONTH ) )
RETURN DIVIDE ( [Total Revenue] - Prev, Prev )

Revenue 7d Rolling Avg =
AVERAGEX (
    DATESINPERIOD ( DateTable[Date], MAX ( DateTable[Date] ), -7, DAY ),
    [Total Revenue]
)

Revenue 28d Rolling Avg =
AVERAGEX (
    DATESINPERIOD ( DateTable[Date], MAX ( DateTable[Date] ), -28, DAY ),
    [Total Revenue]
)

Revenue YTD = TOTALYTD ( [Total Revenue], DateTable[Date] )

Revenue MTD = TOTALMTD ( [Total Revenue], DateTable[Date] )
```

### Profit (estimated — see the caveat on page 8)

```dax
Estimated Profit = SUM ( daily_sales_features[Profit] )

Profit Margin % = DIVIDE ( [Estimated Profit], [Total Revenue] )
```

### Product and country

For these, `Sales` is the line-item table exported in Step 0.

```dax
Revenue by Product = SUM ( Sales[Revenue] )

Product Rank =
RANKX ( ALLSELECTED ( Sales[Description] ), [Revenue by Product], , DESC, DENSE )

Top 10 Product Revenue =
CALCULATE ( [Revenue by Product], KEEPFILTERS ( TOPN ( 10, ALL ( Sales[Description] ), [Revenue by Product], DESC ) ) )

Product Revenue % of Total =
DIVIDE ( [Revenue by Product], CALCULATE ( [Revenue by Product], ALL ( Sales[Description] ) ) )

Country Revenue = SUM ( Sales[Revenue] )

Country Rank =
RANKX ( ALLSELECTED ( Sales[Country] ), [Country Revenue], , DESC, DENSE )

Customer Concentration =
VAR TotalRev = CALCULATE ( SUM ( Sales[Revenue] ), ALL ( Sales ) )
VAR Top10 = CALCULATE ( SUM ( Sales[Revenue] ),
              KEEPFILTERS ( TOPN ( 10, VALUES ( Sales[CustomerID] ), [Country Revenue], DESC ) ) )
RETURN DIVIDE ( Top10, TotalRev )
```

### Customer analytics (from `customer_segments.csv`)

```dax
Customers = COUNTROWS ( customer_segments )

Segment Customers = COUNTROWS ( customer_segments )

Segment Revenue = SUM ( customer_segments[Monetary] )

Revenue Share % =
DIVIDE ( [Segment Revenue], CALCULATE ( [Segment Revenue], ALL ( customer_segments[Segment] ) ) )

Customer Share % =
DIVIDE ( [Customers], CALCULATE ( [Customers], ALL ( customer_segments[Segment] ) ) )

Avg Recency = AVERAGE ( customer_segments[Recency] )

Avg Frequency = AVERAGE ( customer_segments[Frequency] )

Avg Monetary = AVERAGE ( customer_segments[Monetary] )

Champions =
CALCULATE ( [Customers], customer_segments[Segment] = "Champions" )

Revenue Concentration =
VAR Ranked =
    ADDCOLUMNS (
        VALUES ( customer_segments[CustomerID] ),
        "@Rev", CALCULATE ( SUM ( customer_segments[Monetary] ) )
    )
VAR TotalRev = SUMX ( Ranked, [@Rev] )
VAR Top20 = SUMX ( TOPN ( ROUND ( COUNTROWS ( Ranked ) * 0.2, 0 ), Ranked, [@Rev], DESC ), [@Rev] )
RETURN DIVIDE ( Top20, TotalRev )
```

A Pareto visual: put `Monetary` on the X axis, customer rank on the Y axis, and
this measure as a line — the classic 80/20 curve.

### Churn (from `customer_churn.csv`)

```dax
Churn Customers = COUNTROWS ( customer_churn )

High Risk Customers =
CALCULATE ( [Churn Customers], customer_churn[RiskTier] = "High Risk" )

High Risk % =
DIVIDE ( [High Risk Customers], [Churn Customers] )

Revenue at Risk =
CALCULATE ( SUM ( customer_churn[Monetary] ), customer_churn[RiskTier] = "High Risk" )

Avg Churn Probability = AVERAGE ( customer_churn[ChurnProbability] )

Actual Churn Rate = AVERAGE ( customer_churn[Churn] )
```

Read the AUC dynamically rather than hard-coding it, so it tracks the pipeline:

```dax
Model AUC =
LOOKUPVALUE (
    churn_model_comparison[auc],
    churn_model_comparison[model], "Gradient Boosting"
)

Model Lift =
LOOKUPVALUE (
    churn_model_comparison[lift_top10],
    churn_model_comparison[model], "Gradient Boosting"
)
```

> Reference run values: AUC **0.814**, lift **1.64x**, 2,162 high-risk customers
> holding **£1.13M** of historic revenue. Use these to sanity-check your visuals.

### Inventory (from `inventory_policy.csv`)

```dax
SKU Count = COUNTROWS ( inventory_policy )

Total EOQ = SUM ( inventory_policy[EOQ] )

Total Safety Stock = SUM ( inventory_policy[SafetyStock] )

Avg Reorder Point = AVERAGE ( inventory_policy[ReorderPoint] )

Annual Inventory Cost = SUM ( inventory_policy[AnnualTotalCost] )

Class A SKUs =
CALCULATE ( [SKU Count], inventory_policy[ABC] = "A" )

Class A Revenue Share =
DIVIDE (
    CALCULATE ( SUM ( inventory_policy[TotalRevenue] ), inventory_policy[ABC] = "A" ),
    SUM ( inventory_policy[TotalRevenue] )
)

SKUs Needing Tight Control =
CALCULATE ( [SKU Count], inventory_policy[ABC] = "A", inventory_policy[XYZ] = "X" )
```

### Forecast (from `forecast_daily.csv`)

```dax
Forecast Revenue = SUM ( forecast_daily[Forecast] )

Forecast Lower = SUM ( forecast_daily[Lower90] )

Forecast Upper = SUM ( forecast_daily[Upper90] )

Forecast Interval Width = [Forecast Upper] - [Forecast Lower]

Forecast MAPE =
LOOKUPVALUE (
    model_comparison_daily[mape],
    model_comparison_daily[model], "Holt-Winters (damped)"
)
```

> Reference run: **16.2% MAPE** on the weekly grain (the planning grain) and
> **33.6%** on the daily grain. The 90-day projection is £3.29M. The model name
> must match `model_comparison_daily.csv` exactly, or the LOOKUPVALUE returns
> blank — check that file after any re-run.

> Update the model name in that measure to whichever model wins your run —
> check `model_comparison_daily.csv`.

### Monitoring (from `drift_report.csv`)

```dax
Features Monitored = COUNTROWS ( drift_report )

Drifted Features =
CALCULATE ( [Features Monitored], drift_report[psi] >= 0.25 )

Max PSI = MAX ( drift_report[psi] )

Retrain Status =
IF ( [Drifted Features] > 0, "RETRAIN REQUIRED", "MODEL HEALTHY" )
```

---

## Step 4 — Colour theme

Import a theme JSON (View → Themes → Browse for themes) so the report matches
the Streamlit app:

```json
{
  "name": "RetailPulse",
  "dataColors": ["#2563EB","#0891B2","#7C3AED","#16A34A","#F59E0B","#DC2626","#0F2B4C","#DB2777"],
  "background": "#EEF2F7",
  "foreground": "#0F172A",
  "tableAccent": "#2563EB",
  "textClasses": {
    "title":    { "fontFace": "Segoe UI Semibold", "fontSize": 14, "color": "#0F172A" },
    "header":   { "fontFace": "Segoe UI Semibold", "fontSize": 12, "color": "#0F172A" },
    "label":    { "fontFace": "Segoe UI",          "fontSize": 10, "color": "#475569" }
  }
}
```

Use a **card visual** with a transparent background, `Segoe UI Semibold` at
28pt for the number, and the label in `#475569` uppercase — this reproduces the
KPI cards used in the Streamlit dashboard.

---

## Step 5 — The 12 pages

### Page 1 — Executive Summary
* KPI cards: Total Revenue, Total Orders, Total Customers, AOV, Revenue YoY %.
* Line: `Revenue` by `DateTable[Date]` with the 28-day rolling average.
* Donut: revenue share by segment.
* Card: `Revenue Concentration` (top-20% customers) — the headline insight.
* Slicers: Year, Country.
* **Insight box:** "Champions are ~22% of customers but drive ~68% of revenue."

### Page 2 — Sales Trend Analysis
* Line + area: daily revenue, with a 7-day moving average line.
* Column: revenue by month, sorted by a numeric month key.
* Line: `Revenue YoY %` by month.
* Small multiples: revenue by `DayName` (weekday pattern).
* Slicers: date range, Year.

### Page 3 — Product Performance
* Table: Description, Revenue, Units, `Product Rank`, `Product Revenue % of Total`.
* Bar: Top 10 products by revenue.
* Treemap: product revenue by Description.
* Scatter: units sold vs revenue per unit (finds high-volume low-margin items).
* **Insight box:** name the product driving the largest single share.

### Page 4 — Customer Analytics
* Cards: Customers, Avg Recency, Avg Frequency, Avg Monetary.
* Bar: customers per segment vs revenue share per segment (dual axis).
* Scatter: Recency vs Monetary, coloured by Segment.
* Table: `RecommendedAction` per segment — this turns analysis into a to-do list.

### Page 5 — Region / Country Sales
* Filled map: `Country Revenue` (set Data Category → Country).
* Bar: Top 10 countries by revenue.
* Card: % of revenue from outside the UK.
* Table: Revenue, Orders, AOV by country.

### Page 6 — Monthly & Seasonal Sales
* Matrix: Year × Month with revenue as values, plus conditional formatting
  (colour scale) — the seasonality heatmap.
* Line: revenue by `WeekNo` to expose the Q4 ramp.
* Column: average revenue by `DayName`.
* **Insight box:** "UK retail peaks in November; this dataset peaks in
  October–November, consistent with pre-Christmas wholesale buying."

### Page 7 — Customer Behaviour
* Histogram: order frequency distribution.
* Scatter: Frequency vs Monetary (log scale on both if possible).
* Bar: revenue by `DayOfWeek` and by hour band (uses the `Hour` column).
* Card: repeat-purchase rate.

### Page 8 — Profit & Revenue Analysis
* Cards: Total Revenue, Estimated Profit, `Profit Margin %`.
* Waterfall: revenue → estimated cost → estimated profit.
* Bar: margin % by ABC class.
* **Prominent caveat:** the source data has no cost price, so profit uses a
  configurable 35% margin. Label this clearly — it is the honest thing to do.

### Page 9 — Inventory Risk
* Cards: SKU Count, Total EOQ, Total Safety Stock, Annual Inventory Cost.
* Matrix: ABC × XYZ with SKU counts (the control matrix).
* Table: SKUs where `ReorderPoint` is highest — the priority reorder list.
* Bar: `Class A Revenue Share`.
* Slicers: ABC, XYZ.

### Page 10 — Order & Transaction
* Cards: Orders, AOV, Units per Order, Line Items.
* Line: orders per day.
* Column: average order value by month.
* Decomposition tree: Revenue → Country → Segment → Product.

### Page 11 — Advanced Analytics
* Cards: `Model AUC`, `Forecast MAPE`, `Fill rate`, `Model AUC` vs baseline.
* Line: forecast vs actual from the backtest.
* Bar: model comparison by MAPE with a 10% target reference line.
* Bar: feature importance for churn.
* Table: `inventory_metrics` (naive vs optimised).

### Page 12 — Interactive Filter Dashboard
* **Slicers:** Date range, Year, Quarter, Country, Segment, ABC, XYZ, RiskTier.
* **Buttons:** bookmark navigation to each page; "Reset all filters" bookmark.
* **Tooltip pages:** create two report pages and enable *Tooltip* in page
  settings — one showing product detail, one showing customer detail. Attach
  them to the product and customer visuals.
* **Drill-through:** create a hidden *Customer Detail* page and add
  `CustomerID` to the drill-through well. Right-click any customer to drill.
* **Bookmarks:** Current view = "All", plus saved views for "Q4 only",
  "High-risk customers", "Class A SKUs".

---

## Step 6 — Performance checklist

Before publishing:

- [ ] **Reduce the line-item table.** Import only the columns you visualise; a
      narrow table is dramatically faster.
- [ ] **Turn off Auto date/time.** Options → Data load → uncheck *Auto date/time*.
      You have a proper date table; Power BI's hidden ones bloat the model.
- [ ] **Check the relationship direction.** Single direction, DateTable → facts.
- [ ] **Aggregate where you can.** Prefer `daily_sales_features` (739 rows) to
      the 777k-row line-item table for trend pages.
- [ ] **Use measures, not calculated columns**, for anything aggregated.
- [ ] **Performance Analyzer** (View → Performance Analyzer → Start recording)
      to find any visual over ~1 second.

---

## Step 7 — Business insights to write up

The report should end with a page of findings, not just charts. Fill these in
from your run:

1. **Revenue concentration** — what share of revenue comes from the top 20% of
   customers, and what does that imply for retention spend?
2. **Champion economics** — Champions' share of customers vs share of revenue.
3. **Seasonality** — which month peaks, and how far in advance should stock be
   committed given a 7-day lead time?
4. **Churn value** — total historic revenue held by high-risk customers, i.e.
   the size of the prize for a win-back campaign.
5. **Assortment focus** — how many class A SKUs carry the majority of revenue?
6. **Stockout cost** — difference in lost units between the naive and optimised
   reorder policies, converted to revenue.
7. **Model reliability** — forecast MAPE, and whether the drift report calls for
   retraining.

---

## Appendix — M query for the sales fact table

Paste into Advanced Editor to load and type the daily table in one step:

```
let
    Source = Csv.Document(
        File.Contents("C:\path\to\retailpulse\data\processed\daily_sales_features.csv"),
        [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]
    ),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    Typed = Table.TransformColumnTypes(Promoted, {
        {"Date", type date},
        {"Revenue", type number},
        {"Units", Int64.Type},
        {"Orders", Int64.Type},
        {"Customers", Int64.Type},
        {"Profit", type number},
        {"IsTradingDay", type logical}
    }),
    RemovedOthers = Table.SelectColumns(Typed, {
        "Date", "Revenue", "Units", "Orders", "Customers", "Profit",
        "IsTradingDay", "AvgOrderValue", "Year", "Month", "DayOfWeek"
    })
in
    RemovedOthers
```

Dropping the ~30 rolling/lag columns keeps the model small — Power BI does not
need them, because time intelligence is handled by DAX.
