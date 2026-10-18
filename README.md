# RetailPulse — AI-Powered Customer Analytics & Demand Forecasting

An end-to-end data science platform for retail: **demand forecasting, customer
segmentation, churn prediction and inventory optimisation**, built on the UCI
*Online Retail II* dataset (1.03M real transactions, Dec 2009 – Dec 2011).

Built for the **Zidio Development — Data Science & Analytics** internship.

---

## Results at a glance

| Capability | Method | Headline result (reference run) |
|---|---|---|
| Customer segmentation | RFM quintiles + K-Means (k by silhouette) | 6 tiers; **Champions = 22% of customers but 68% of revenue** |
| Churn prediction | Gradient boosting on 12 behavioural features | **ROC AUC 0.814**, top-decile lift **1.64x** |
| Demand forecasting | Rolling-origin backtest, 4 model families | **16.2% MAPE weekly**, **33.6% daily** (damped Holt-Winters) |
| Inventory optimisation | Order-up-to policy vs naive, demand replay | Fill rate **90.6% → 98.5%**; units lost **down 84%** |
| Monitoring | PSI + Kolmogorov–Smirnov drift tests | 7 features, **all stable** (max PSI 0.056) |
| Data pipeline | 7 documented quality rules | 1,033,036 → **776,594 clean rows** in ~26s |

> Every number above is produced by the pipeline and written to
> `data/processed/`. Nothing is hard-coded. Re-run the pipeline and the numbers
> in the README can be checked line by line against the CSVs.

---

## Quick start

```bash
# 1. install dependencies (see "Runs with just pandas + NumPy" below)
pip install -r requirements.txt

# 2. build everything: download -> clean -> segment -> forecast -> churn -> inventory
python -m src.run_pipeline

# 3. launch the dashboard
streamlit run dashboard/app.py
```

### Useful pipeline options

```bash
python -m src.run_pipeline --list                  # show the stage order
python -m src.run_pipeline --only churn            # run one stage
python -m src.run_pipeline --from forecasting      # resume from a stage
python -m src.run_pipeline --skip download         # reuse the cached raw data
python -m src.run_pipeline --dry-run               # show the plan
```

Each stage writes to `data/processed/` and is safe to re-run. Expensive inputs
are cached, so a second run is much faster than the first.

---

## Runs with just pandas + NumPy

The pipeline is designed so a missing wheel never blocks a build. Everything
outside the core data stack is **optional**, and the package probes for it at
startup:

```
$ python -m src.optional_deps
Optional dependencies:
  scikit-learn     not installed (using built-in fallback)
  matplotlib       not installed (using built-in fallback)
  prophet          not installed (using built-in fallback)
  ...
```

| Optional package | Used for | Built-in fallback |
|---|---|---|
| scikit-learn | K-Means, Random Forest, Gradient Boosting | `src/models/numpy_ml.py` — k-means++, CART trees, boosting |
| matplotlib / seaborn | PNG charts | `src/viz/svg_charts.py` — dependency-free SVG charts |
| prophet | an extra forecasting model | automatically skipped from the model list |
| joblib | model persistence | standard-library `pickle` |
| mlflow | experiment tracking | skipped; metrics still written to CSV |

Force the fallback path at any time (useful for testing):

```bash
RETAILPULSE_FORCE_FALLBACK=1 python -m src.run_pipeline     # bash
$env:RETAILPULSE_FORCE_FALLBACK='1'; python -m src.run_pipeline   # PowerShell
```

---

## Architecture

```
Raw UCI workbooks (2 sheets, 1,067,371 rows)
        │
        ▼  Stage 1  ingestion            → data/interim/transactions.*
   normalise schema, drop 34,335 duplicate Dec-2010 rows
        │
        ▼  Stage 2  cleaning + features  → daily_sales_features.csv (739 days x 42 cols)
   7 documented quality rules             weekly_sales_features.csv (106 weeks)
   + lag/rolling/seasonal features        customer_rfm.csv (5,852 customers)
        │
        ├─▶ Stage 3  segmentation        → customer_segments.csv, segment_profiles.csv
        │            RFM tiers + K-Means (k chosen by silhouette)
        │
        ├─▶ Stage 4  forecasting         → forecast_daily.csv, model_comparison_daily.csv
        │            rolling-origin backtest, 4 model families
        │
        ├─▶ Stage 5  churn               → customer_churn.csv, churn_feature_importance.csv
        │            observation/prediction window split, 3 classifiers
        │
        ├─▶ Stage 6  inventory           → inventory_policy.csv, inventory_metrics.csv
        │            EOQ + safety stock + reorder point + policy simulation
        │
        └─▶ Stage 7  drift & tracking    → drift_report.csv, mlruns/
                     PSI + KS tests, MLflow run
        │
        ▼
   Streamlit dashboard (6 pages)  +  Power BI guidance (reports/)
```

Full detail: [`reports/ARCHITECTURE.md`](reports/ARCHITECTURE.md)

---

## Project structure

```
retailpulse/
├── src/
│   ├── config.py                  # every path, threshold and business rule
│   ├── optional_deps.py           # what is installed, and what to fall back to
│   ├── run_pipeline.py            # orchestrator: runs all stages in order
│   ├── ingestion/
│   │   ├── download_data.py       # reproducible UCI download
│   │   └── load_data.py           # schema normalisation + quality report
│   ├── features/
│   │   └── build_features.py      # cleaning funnel, daily/weekly/RFM features
│   ├── segmentation/
│   │   └── segment_customers.py   # RFM tiers, K-Means, silhouette, k selection
│   ├── forecasting/
│   │   └── forecast_demand.py     # 4 models, rolling backtest, forward forecast
│   ├── churn/
│   │   └── predict_churn.py       # window design, 3 classifiers, worklist
│   ├── inventory/
│   │   └── optimize_inventory.py  # ABC/XYZ, EOQ, safety stock, simulation
│   ├── evaluation/
│   │   └── metrics.py             # MAPE, RMSE, AUC, KS, lift — all from scratch
│   ├── mlops/
│   │   ├── drift.py               # PSI + KS drift detection
│   │   └── track_experiments.py   # MLflow logging (optional)
│   ├── models/
│   │   ├── numpy_ml.py            # fallback K-Means, trees, boosting
│   │   └── persistence.py         # joblib or pickle
│   └── viz/
│       └── svg_charts.py          # fallback charting
├── dashboard/
│   ├── app.py                     # Streamlit entry point
│   ├── design.py                  # palette, KPI cards, Plotly theme
│   ├── data.py                    # cached loaders with friendly failure
│   └── views/                     # sales, customers, forecast, inventory, status
├── tests/
│   └── test_retailpulse.py        # 40+ unit tests on the pure logic
├── reports/                       # methodology, interpretation, figures
├── data/{raw,interim,processed}/
├── models/
├── k8s/deployment.yaml
├── Dockerfile
├── .github/workflows/ci.yml
└── requirements.txt
```

---

## Methodology notes (the parts that matter)

**Revenue is the demand proxy.** The dataset has no separate
quantity-demanded field, and cancellations are removed, so forecasts represent
*expected sales*, not unconstrained demand. This is stated wherever a forecast
appears rather than buried.

**Churn uses an observation window, not a leaky label.** Features come strictly
from before a cutoff date; the label comes from the 90 days after it. A model
that saw the future would score better and be worthless.

**Accuracy comes from rolling-origin backtests.** A single train/test split can
flatter a model by luck. Each model is retrained and re-scored across multiple
folds, and the reported spread (`mape_std`) matters as much as the mean.

**Metrics are implemented, not imported.** MAPE, SMAPE, WAPE, RMSE, ROC AUC,
KS and lift are written out in `src/evaluation/metrics.py`, including the
tie-handling and near-zero-guard details that quietly change results.

**Variability includes zero-demand days.** A SKU that sells on 30 of 739 days is
far more erratic than one that sells a little every day. Safety stock is
computed over the full calendar, otherwise buffers are badly understated.

**Profit is estimated.** The source has no cost price, so a configurable 35%
gross margin is applied (`DEFAULT_MARGIN` in `src/config.py`). Profit figures
are directional until a real cost file is supplied.

---

## Dashboard

Six pages, all reading the pipeline outputs:

| Page | Answers |
|---|---|
| Overview | What does the platform do, and what is the measured impact? |
| Sales Analytics | Revenue, trend, seasonality, top products and countries |
| Customer Intelligence | RFM tiers, K-Means clusters, churn risk, retention worklist |
| Demand Forecasting | Forecast with interval, what-if scenario, backtest accuracy |
| Inventory Optimisation | EOQ, safety stock, reorder worklist, policy simulation |
| Data Quality & Monitoring | Cleaning funnel, drift report, data dictionary |

---

## Power BI

The brief specifies a 12-page Power BI report alongside the Streamlit app.
Step-by-step instructions, the full DAX measure library and the page-by-page
build order are in [`reports/POWERBI_GUIDE.md`](reports/POWERBI_GUIDE.md).

---

## Testing

```bash
pytest tests -v            # if pytest is installed
python tests/run_tests.py  # dependency-free runner, same suite
```

**37 tests, no dataset required** — they exercise the pure logic in seconds:

- metric correctness, including AUC tie handling and hand-computed values
- every RFM segment rule, checked across all 125 R/F/M score combinations
- EOQ optimality (total cost is genuinely minimised) and safety-stock monotonicity
- PSI/KS behaviour on identical vs shifted distributions
- a feature-leakage contract: rolling averages must never include the current day

---

## Deployment

```bash
docker build -t retailpulse .
docker run -p 8501:8501 retailpulse

kubectl apply -f k8s/deployment.yaml
```

CI runs lint, tests and a Docker build on every push
(`.github/workflows/ci.yml`).

---

## Data source

| Property | Value |
|---|---|
| Dataset | Online Retail II (UCI ML Repository, id 502) |
| Mirror | The Kaggle "Online Retail II" dataset is a copy of this |
| Period | 2009-12-01 to 2011-12-09 |
| Raw rows | 1,067,371 across two sheets |
| After de-duplication | 1,033,036 |
| After quality rules | 776,594 |
| Customers | 5,852 |
| Products | 4,620 |
| Countries | 41 |

`src/ingestion/download_data.py` fetches it directly from UCI, so the project
bootstraps without a Kaggle API key.

---

## Honest limitations

- No delivery dates, so revenue is attributed to invoice date.
- No cost column, so margin is an assumption.
- No inventory snapshots, so stock levels are simulated rather than observed.
- No marketing spend, so uplift scenarios are illustrative, not causal.
- 2011 ends on 9 December, so the final weeks are a partial year — relevant
  when comparing 2010 and 2011 annual totals.

*Built with Python, pandas, NumPy, scikit-learn, Prophet, Streamlit and MLflow.*
