# RetailPulse — Architecture

Five layers, matching the project brief: **data → features → models → serving →
monitoring**.

```
┌──────────────────────────────────────────────────────────────────────────┐
│ 1. DATA LAYER                                                            │
│                                                                          │
│   UCI Online Retail II (2 sheets, 1,067,371 rows)                        │
│        │                                                                 │
│        ├─ ingestion/download_data.py   direct download, no API key       │
│        └─ ingestion/load_data.py       schema normalisation              │
│                                        + 34,335 duplicate Dec-2010 rows  │
│                                        removed by exact-match dedup      │
│                     ▼                                                    │
│   data/interim/transactions.*          1,033,036 rows                    │
│        │                                                                 │
│        └─ 7 documented quality rules                                     │
│                     ▼                                                    │
│   data/interim/transactions_clean.*    776,594 rows                      │
└──────────────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼────────────────────────────────────────┐
│ 2. FEATURE ENGINEERING LAYER                                             │
│                                                                          │
│   Time series                 │  Customer                                │
│   ─────────────               │  ────────                                │
│   daily grain (739 days)      │  RFM per customer (5,852)                │
│   weekly grain (106 weeks)    │    Recency / Frequency / Monetary        │
│   lags 1,7,14,28,364          │    R/F/M quintile scores 1-5             │
│   rolling mean/std 7..90      │    RFM_Score 3-15, RFM_Cell              │
│   Fourier seasonality         │    tenure, purchase rate, AOV            │
│   holiday distance            │                                          │
│   ALL rolling features are    │                                          │
│   shifted by one day to       │                                          │
│   prevent target leakage      │                                          │
└─────────────────────────────────┬────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼────────────────────────────────────────┐
│ 3. MODEL LAYER                                                           │
│                                                                          │
│  SEGMENTATION        FORECASTING            CHURN          INVENTORY     │
│  ────────────        ───────────            ─────          ─────────     │
│  RFM rule tiers      Seasonal naive         Logistic Reg.  ABC/XYZ       │
│  K-Means (k by       Linear trend+seas.     Random Forest  EOQ           │
│   silhouette)        Holt-Winters           Grad Boosting  Safety stock  │
│  silhouette, CH,     Prophet (optional)                    Reorder point │
│  Davies-Bouldin      rolling-origin                        policy sim    │
│                      backtest, 4 folds                                   │
│                                                                          │
│  Artefacts → models/    Run metrics → reports/run_metrics.csv (MLflow)   │
└─────────────────────────────────┬────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼────────────────────────────────────────┐
│ 4. SERVING LAYER                                                         │
│                                                                          │
│  Streamlit dashboard (6 pages)     Power BI report (12 pages)            │
│  dashboard/app.py                  reports/POWERBI_GUIDE.md              │
│   ├─ Overview                       ├─ Executive Summary                 │
│   ├─ Sales Analytics                ├─ Sales Trend                       │
│   ├─ Customer Intelligence          ├─ Product Performance               │
│   ├─ Demand Forecasting             ├─ Customer Analytics                │
│   ├─ Inventory Optimisation         ├─ Region / Country                  │
│   └─ Data Quality & Monitoring      ├─ Monthly & Seasonal                │
│                                     ├─ Customer Behaviour                │
│  Docker → container image           ├─ Profit & Revenue                  │
│  Kubernetes → Deployment+Service    ├─ Inventory Risk                    │
│              +HPA                   ├─ Order & Transaction               │
│                                     ├─ Advanced Analytics                │
│                                     └─ Interactive Filters               │
└─────────────────────────────────┬────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼────────────────────────────────────────┐
│ 5. MONITORING LAYER                                                      │
│                                                                          │
│  PSI (Population Stability Index)  reference half vs current half        │
│     < 0.10 stable | 0.10-0.25 moderate | > 0.25 retrain                 │
│  KS two-sample test with asymptotic p-value                              │
│  MLflow run: params, metrics, figures, report artefacts                  │
│  reports/run_metrics.csv: machine-readable run record                    │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Why each design decision was made

### Ingestion

**Two sheets, one table.** Online Retail II ships as two workbooks whose date
ranges overlap across December 2010. Loading only one sheet — a common mistake —
silently discards half the history. We concatenate and drop exact duplicates,
recovering 1,033,036 unique rows and removing 34,335 double-counted ones.

**Cached as Parquet.** Reading the Excel workbooks takes about 70 seconds. The
first run writes an intermediate cache; every later run reads it in under a
second. Where a Parquet engine is unavailable the cache transparently becomes
gzip-compressed CSV, so the pipeline still runs on a bare pandas install.

### Feature engineering

**Shift-before-roll.** Every rolling statistic uses `shift(1)` first. Without
it, "the 7-day average revenue" for a given day includes that day's own revenue,
which is target leakage and produces optimistic backtests that do not survive
contact with production.

**An explicit zero calendar.** Trading days with no sales are inserted as real
rows with zero revenue, because a time series with holes forecasts badly and a
SKU's variability is understated if unsold days are ignored.

### Segmentation

**Two views on purpose.** The rule-based RFM tiers are what a marketing team can
act on ("these 1,284 Champions get early access"). The K-Means clusters are a
check on whether the rules match the data's own structure. When they disagree,
that is interesting rather than wrong — and the k-scan output makes the choice
of *k* auditable instead of a guess.

**Cluster numbering is value-ranked.** "Cluster 1" always means the most
valuable group, so the labels stay meaningful across reruns.

### Forecasting

**Backtest, don't split.** A single train/test split can flatter a model by
luck. Rolling-origin backtesting retrains and rescoring across folds gives a
mean *and* a standard deviation, and the spread is often the more useful number.

**A seasonal-naive baseline is mandatory.** If a sophisticated model cannot beat
"repeat last week", it should not ship. It is included in every comparison.

**Fourier terms with a fixed calendar origin.** Deriving the annual angle from
`dayofyear` rather than from an index inside the training window means late
November maps to the same phase in every year, so seasonality is learned across
years.

**Prediction intervals widen with horizon.** Forecast uncertainty genuinely
grows; a flat band implies a confidence that does not exist.

### Churn

**Observation window, not a leaky label.** Features come strictly from before a
cutoff; the label comes from the 90 days after it. Getting this wrong is the
most common way a churn model reports an AUC of 0.99 and is useless.

**Lift, not just AUC.** A retention team can call perhaps 200 customers. What
matters is how many churners are in that top slice, which `lift_at_k` measures
directly.

### Inventory

**Variability over the whole calendar.** Safety stock depends on demand
variability, and a SKU that sells on 30 of 739 days is far more erratic than one
that sells a little daily. Measuring standard deviation only on selling days
understates the buffer needed — a mistake that shows up as stockouts.

**Simulate the policy, don't just calculate it.** EOQ and reorder points are
formulas; whether they actually reduce stockouts depends on lead times, review
periods and demand randomness. Replaying real demand through the policy is the
only way to claim a fill-rate improvement honestly.

### Monitoring

**Drift is measured, not assumed.** PSI and KS are computed per feature between
the reference and current halves of history. The report states plainly whether
retraining is needed rather than leaving a dashboard green by default.

---

## Data contracts

| File | Grain | Rows | Key columns |
|---|---|---|---|
| `transactions_clean.*` | line item | 776,594 | Invoice, StockCode, Quantity, Date, Revenue |
| `daily_sales_features.csv` | day | 739 | Date, Revenue, Units, Orders, lag/rolling features |
| `weekly_sales_features.csv` | week | 106 | Date, Revenue, Orders, TradingDays |
| `customer_rfm.csv` | customer | 5,852 | CustomerID, Recency, Frequency, Monetary, R/F/M_Score |
| `customer_segments.csv` | customer | 5,852 | + Segment, Cluster, ClusterName, RecommendedAction |
| `customer_churn.csv` | customer | 5,256 | + ChurnProbability, RiskTier |
| `inventory_policy.csv` | SKU | 4,620 | StockCode, ABC, XYZ, EOQ, SafetyStock, ReorderPoint |
| `forecast_daily.csv` | future day | 90 | Date, Forecast, Lower90, Upper90 |
| `drift_report.csv` | feature | 7 | feature, psi, ks_statistic, ks_p_value |

Row counts are from the reference run and will vary slightly if the source or
quality rules change.

---

## Deployment topology

```
                    ┌─────────────────┐
   GitHub push ───▶ │ GitHub Actions  │  lint → test → docker build
                    └────────┬────────┘
                             │ image
                    ┌────────▼────────┐
                    │ Container       │  multi-stage, non-root,
                    │ registry        │  healthcheck on /_stcore/health
                    └────────┬────────┘
                             │
                    ┌────────▼──────────────────────────┐
                    │ Kubernetes                        │
                    │  Deployment (2+ replicas)         │
                    │  Service (ClusterIP :80 → :8501)  │
                    │  HPA (2→6 pods at 70% CPU)        │
                    │  PVC  (read-only processed data)  │
                    └───────────────────────────────────┘
```

The container runs as a non-root user, mounts processed data read-only, and
exposes an HTTP health endpoint that both the Docker `HEALTHCHECK` and the
Kubernetes readiness/liveness probes use.
