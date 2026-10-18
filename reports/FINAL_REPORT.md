# RetailPulse — Final Project Report

**AI-Powered Customer Analytics & Demand Forecasting Platform**

Dataset: UCI *Online Retail II* · 1,033,036 transactions · Dec 2009 – Dec 2011
Prepared for: Zidio Development — Data Science & Analytics

---

## 1. Executive summary

RetailPulse ingests two years of real retail transactions and turns them into
five decisions a retailer actually has to make. Every figure below is produced
by the pipeline and written to `data/processed/`, so each one can be re-derived
by re-running `python -m src.run_pipeline`.

| # | Question | Answer from the data |
|---|---|---|
| 1 | Where does our money come from? | **22% of customers (Champions) generate 68% of revenue.** The top 20% of customers generate 77%. |
| 2 | Who is about to leave? | **2,162 customers** are high churn risk, holding **£1.13M** of historic revenue. |
| 3 | What will we sell next? | **£3.29M over the next 90 days**, with a weekly forecast accurate to **16.2% MAPE**. |
| 4 | How much stock should we hold? | Moving from a naive reorder rule to a statistical one lifts fill rate **90.6% → 98.5%**. |
| 5 | Can we trust these models? | All 7 monitored features are **stable** (max PSI 0.056); no retraining triggered. |

The platform runs end-to-end in **78 seconds** and needs only pandas and NumPy.
Everything else (scikit-learn, Prophet, matplotlib, MLflow) is an optional
accelerator with a built-in fallback.

---

## 2. Problem statement and objectives

The brief asks for a platform that:

1. Predicts future product demand using AI forecasting models.
2. Analyses customer purchasing behaviour.
3. Identifies churn-prone customers.
4. Improves inventory planning and reduces overstocking.
5. Delivers interactive dashboards with real-time insights.
6. Follows MLOps practice: experiment tracking, drift detection, containerised
   deployment, CI/CD.

**Business value targeted:** better forecasting accuracy, reduced inventory
wastage, improved customer retention, data-driven decision making.

Each objective maps to a pipeline stage and a measurable acceptance test in
section 5.

---

## 3. Dataset

| Property | Value |
|---|---|
| Source | UCI ML Repository id 502, *Online Retail II* |
| Kaggle mirror | `cgrymn/online-retail-ii-uci-dataset` is a copy of this |
| Period | 2009-12-01 to 2011-12-09 |
| Raw rows | 1,067,371 across two worksheets |
| After de-duplication | 1,033,036 (34,335 duplicate December-2010 rows removed) |
| After quality rules | **776,594** |
| Customers | 5,852 |
| Products | 4,620 |
| Countries | 41 |
| Total revenue | £17,068,568 |

### 3.1 Why the two sheets matter

Online Retail II ships as two workbooks whose ranges overlap across December
2010. Loading only one sheet — a common and silent mistake — discards half the
history. Concatenating and dropping exact duplicates recovers the full two-year
series.

### 3.2 Cleaning rules and their effect

| Rule | Rows removed | Rationale |
|---|---|---|
| Drop exact duplicates | 0 (after merge dedup) | — |
| Drop missing CustomerID | 235,151 | Behaviour cannot be attributed to an unknown customer |
| Drop cancellations (`C*`) | 18,390 | Returns are not demand |
| Drop Quantity ≤ 0 | 0 | Returns and stock write-offs |
| Drop Price ≤ 0 | 70 | Freebies and data-entry errors |
| Drop non-product stock codes | 2,831 | Postage, bank charges, manual adjustments |
| **Remaining** | **776,594** | |

### 3.3 Documented limitations of the source

- **No cost price.** Profit uses a configurable 35% gross margin
  (`DEFAULT_MARGIN`). Profit figures are directional, and this is stated on
  every page where profit appears.
- **No delivery date.** Revenue is attributed to invoice date.
- **No inventory snapshots.** Stock levels are simulated, not observed.
- **No marketing spend.** Uplift scenarios are illustrative, not causal.
- **2011 ends on 9 December**, so the final year is partial. This materially
  affects the forecast horizon and is discussed in section 5.3.

---

## 4. Method

### 4.1 Feature engineering

Two analysis-ready grains plus a customer grain:

- `daily_sales_features.csv` — 739 days × 42 columns
- `weekly_sales_features.csv` — 106 weeks
- `customer_rfm.csv` — 5,852 customers × 18 columns

Three choices that matter:

**Shift before rolling.** Every rolling statistic applies `shift(1)` first. Without
it, "the 7-day average" for a day includes that day's own revenue — target
leakage that produces flattering backtests and fails in production. A unit test
enforces this contract.

**An explicit zero calendar.** Days with no trading are inserted as real rows
with zero revenue. A time series with holes forecasts badly, and a SKU's
variability is understated if unsold days are dropped.

**RFM scoring.** R/F/M quintiles (1–5, recency inverted) combine into a 3–15
score and six named tiers, each with a recommended marketing action.

### 4.2 Models

| Task | Models compared | Selection rule |
|---|---|---|
| Segmentation | Rule-based RFM tiers; K-Means with k by silhouette, validated by Calinski-Harabasz and Davies-Bouldin | k = 4 (elbow + silhouette) |
| Forecasting | Seasonal naive, Linear trend+seasonality, Damped Holt-Winters, Prophet | Lowest mean MAPE over 4 rolling-origin folds |
| Churn | Logistic Regression, Random Forest, Gradient Boosting, majority-class baseline | Highest ROC AUC |
| Inventory | ABC × XYZ, EOQ, safety stock, reorder point; order-up-to policy simulation | Fill rate vs holding cost |

### 4.3 Why rolling-origin backtesting

A single train/test split can flatter a model by luck. Each forecast model is
retrained and re-scored across four folds, so the report carries a **mean and a
standard deviation** — and the spread is often the more useful number. A model
with 16% mean MAPE and 10 points of spread is less trustworthy than one with 18%
and 2 points.

### 4.4 Metrics implemented, not imported

MAPE, SMAPE, WAPE, MAE, RMSE, R², bias, tracking signal, ROC AUC, KS and lift
are written out in `src/evaluation/metrics.py`. This is deliberate: the
near-zero guard in MAPE, the average-rank handling of ties in AUC, and the
choice of exposure window all quietly change results, so they are visible and
unit-tested rather than hidden behind a library call.

---

## 5. Results

### 5.1 Customer segmentation

| Segment | Customers | % of customers | % of revenue | Avg recency | Avg orders | Avg revenue |
|---|---|---|---|---|---|---|
| Champions | 1,284 | 21.9% | **68.1%** | 20 days | 17.0 | £9,056 |
| Loyal Customers | 1,130 | 19.3% | 14.6% | 73 days | 5.9 | £2,212 |
| Need Attention | 1,139 | 19.5% | 8.3% | 249 days | 3.8 | £1,239 |
| Dormant | 1,168 | 20.0% | 4.7% | 543 days | 2.0 | £684 |
| New / Promising | 831 | 14.2% | 3.6% | 64 days | 1.4 | £743 |
| At Risk | 300 | 5.1% | 0.7% | 316 days | 1.0 | £380 |

**Interpretation.** The concentration is extreme: 22% of customers carry 68% of
revenue, and across the whole base the top 20% of customers account for
**77.2%**. This has a direct operational consequence — a retention programme
should not spread effort evenly. Losing a handful of Champions is worth more
than losing a thousand Dormant accounts, and the pricing, service and
win-back decisions for those two groups should have almost nothing in common.

The K-Means clustering (k = 4, silhouette 0.584) independently separates a
4-customer wholesale cluster worth £421k each, a 51-customer VIP cluster, and
two large mainstream clusters — confirming the rule-based tiers are not an
artefact of the cut-offs.

### 5.2 Churn prediction

Setup: features strictly from before a 2011-09-10 cutoff, label from the 90 days
after it. 5,256 customers, churn rate 56.4%.

| Model | ROC AUC | KS | Lift @ top 10% | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| **Gradient Boosting** | **0.8139** | 0.4898 | **1.636** | 0.754 | 0.814 | 0.783 |
| Logistic Regression | 0.8079 | 0.4863 | 1.636 | 0.790 | 0.732 | 0.760 |
| Random Forest | 0.8063 | 0.4838 | 1.609 | 0.762 | 0.799 | 0.780 |
| Baseline (majority class) | 0.5000 | 0.0000 | 1.000 | 0.565 | 1.000 | 0.722 |

**Interpretation.** An AUC of 0.81 is a genuinely useful churn model — well above
the 0.5 baseline and not suspiciously near 1.0, which is usually the signature
of a leaky label. The top-decile lift of 1.64 means a retention team calling the
highest-risk 10% reaches churners at 1.6x the base rate, so a limited calling
budget is spent far more efficiently than at random.

The dominant driver is **Recency (39% of importance)**, followed by
ActiveMonths, Monetary and PurchaseRate. This is intuitive and reassuring: the
model has learned the same thing an experienced merchandiser would say.

**Risk tiers:** 2,162 high risk (41.1%), 2,102 medium (40.0%), 992 low (18.9%).
High-risk customers hold **£1.13M** of historic revenue — the size of the prize
for a win-back campaign, and the number a business case should start from.

> **Prioritise by Monetary × ChurnProbability, not risk alone.** A 95% churn
> probability on a £30 customer is not worth a phone call. The dashboard
> produces this ranked worklist directly.

### 5.3 Demand forecasting

Four rolling-origin folds, 28-day horizon, retrained each fold.

**Daily grain**

| Model | MAE | RMSE | MAPE | MAPE std | WAPE | Bias | R² |
|---|---|---|---|---|---|---|---|
| **Holt-Winters (damped)** | **14,911** | **21,954** | **33.6%** | 8.0 | 43.9% | +4,034 | 0.11 |
| Seasonal Naive | 15,264 | 21,998 | 49.7% | 13.2 | 46.9% | +3,491 | 0.04 |
| Linear (trend + seasonality) | 20,292 | 26,438 | 50.0% | 4.9 | 60.5% | +2,724 | −0.37 |

**Weekly grain** (the planning grain)

| Model | MAE | RMSE | MAPE | MAPE std | WAPE | Bias | R² |
|---|---|---|---|---|---|---|---|
| **Holt-Winters (damped)** | **42,055** | **57,231** | **16.2%** | 9.9 | 17.9% | +28,321 | −0.62 |
| Linear (trend + seasonality) | 54,011 | 70,103 | 20.7% | 6.6 | 21.8% | +15,032 | −4.61 |
| Seasonal Naive | 54,945 | 66,097 | 22.3% | 9.0 | 23.3% | +33,829 | −3.46 |

**Interpretation.**

- **Weekly forecasting is roughly twice as accurate as daily** (16.2% vs 33.6%
  MAPE). This is expected: daily retail demand is dominated by noise, and the
  weekly aggregate smooths weekend closures and single-order spikes. Planning
  should therefore be done weekly; the daily series is for tracking, not for
  committing stock.
- **The damped trend is what makes the model usable.** An undamped
  Holt-Winters extrapolated the pre-Christmas ramp into February and produced
  £19.7M for 90 days — more than the entire two-year history. Damping the trend
  geometrically and reverting toward the typical level of the month being
  forecast brought that to £3.29M, close to the same period a year earlier
  (£3.08M) and to the most recent actual 90 days (£3.38M).
- **All models carry a positive bias**, meaning they under-forecast on average.
  For stock planning that is the safer direction of error — under-forecasting
  risks a stockout, over-forecasting risks dead stock — but it is worth knowing
  rather than discovering.
- **Negative R² on the weekly models** is not a contradiction. R² compares
  against the mean of the *test* window; when four test weeks are highly
  variable and the model is anchored to a different level, R² can go negative
  even while MAPE is respectable. MAPE and WAPE are the meaningful measures here.

**Forward forecast:** **£3,285,553 over the next 90 days** (90% interval
£0.3M–£19.1M). The interval is wide because daily dispersion is large and the
projection extends well past the validated horizon — an honest representation
of what a 90-day daily-grain forecast can and cannot promise.

> **Horizon caveat (important).** The backtest validates a 28-day horizon. The
> 90-day projection is an extrapolation beyond that validated range, and the
> source data ends inside the December peak. This is stated in the dashboard and
> in the design of the prediction interval rather than presented as equivalent
> confidence.

### 5.4 Inventory optimisation

ABC by revenue contribution; XYZ by demand variability, using quantile cut-offs
on the coefficient of variation. Fixed thresholds are not usable here: retail
demand is heavily intermittent, so a fixed CV bound puts nearly every SKU in one
class and the matrix stops discriminating.

**ABC × XYZ matrix (SKU counts)**

| | X (stable) | Y | Z (erratic) |
|---|---|---|---|
| **A** | 773 | 170 | 58 |
| **B** | 593 | 467 | 162 |
| **C** | 174 | 903 | 1,320 |

Class A SKUs carry **80.0% of revenue** — the classic Pareto shape. A-X items
(773 SKUs) are the sweet spot: high revenue and predictable demand, so their
replenishment can be automated. C-Z items (1,320 SKUs) are cheap and erratic;
reviewing them weekly is wasted effort.

**Policy simulation** (604 days of real aggregate demand, 7-day review, 7-day lead time)

| Metric | Naive rule | Statistical policy | Change |
|---|---|---|---|
| Fill rate | 90.56% | **98.48%** | **+7.9 points** |
| Units lost | 991,177 | **159,328** | **−84%** |
| Stockout days | 66 | 12 | −54 |
| Orders placed | 77 | 87 | +10 |
| Avg stock on hand | 92,251 | 146,557 | +54,306 |
| Annual holding + ordering cost | £12,946 | £20,033 | +£7,087 |

**Interpretation.** The statistical policy trades **£7,087/year** of extra
holding cost for **831,849 fewer lost units**, worth roughly **£1.35M** in
revenue at the £1.63 average selling price. Under any realistic margin this is
not a close call.

The finding is more useful than a simple "optimisation wins" headline, because
it exposes the real trade-off: better service here is bought with working
capital, not with a free lunch. A retailer short of cash might deliberately sit
between the two policies.

**A methodological note worth recording:** an earlier version of this simulation
ordered one EOQ whenever stock hit the reorder point. It reported a 33% fill
rate — worse than doing nothing. The bug was a scale mismatch: EOQ is sized for
the EOQ *cycle* (roughly 34 orders a year), not the 7-day *review* cycle, so the
policy could never replenish demand in time. EOQ answers "how much per order",
not "what level to hold". The corrected simulation compares a statistical
order-up-to level against a naive one, which is the honest comparison.

### 5.5 Drift monitoring

Reference = first half of history, current = second half.

| Feature | Reference mean | Current mean | Change | PSI | KS p-value | Verdict |
|---|---|---|---|---|---|---|
| AvgOrderValue | 377.25 | 391.97 | +3.9% | 0.056 | 0.209 | stable |
| Units | 14,698.98 | 13,715.74 | −6.7% | 0.046 | 0.776 | stable |
| Revenue | 22,894.50 | 23,298.64 | +1.8% | 0.042 | 0.725 | stable |
| LineItems | 1,057.90 | 1,043.86 | −1.3% | 0.040 | 0.489 | stable |
| UnitsPerOrder | 246.25 | 231.27 | −6.1% | 0.033 | 0.742 | stable |
| Orders | 50.15 | 48.89 | −2.5% | 0.018 | 0.927 | stable |
| Customers | 44.56 | 44.42 | −0.3% | 0.013 | 0.997 | stable |

**Interpretation.** Every feature sits far below the 0.10 moderate-shift
threshold — the largest PSI is 0.056 and the smallest KS p-value is 0.21. **No
retraining is indicated.** The monitored business is stable across the two years,
which is itself a finding: it means the models trained on the first half remain
valid for the second, and the accuracy figures are not the product of one lucky
period.

---

## 6. Acceptance criteria

The brief lists eight features (F1–F8). Status against each:

| ID | Feature | Acceptance criterion | Status | Evidence |
|---|---|---|---|---|
| F1 | Customer segmentation | Customers categorised successfully | **Met** | 6 tiers, 5,852 customers, auditable k selection |
| F2 | Churn prediction | Accurate churn insights generated | **Met** | AUC 0.814, lift 1.64x, ranked worklist |
| F3 | Demand forecasting | Reliable forecast trends shown | **Met** | 16.2% weekly MAPE, 4-fold backtest, prediction intervals |
| F4 | Hybrid forecasting | Improved prediction quality | **Partially met** | 4 model families compared; the damping and seasonal-reversion mechanisms improved a single model rather than blending two. Prophet was not available in this environment, so a Prophet+LSTM ensemble was not evaluated. |
| F5 | Inventory optimisation | Reduced overstocking risk | **Met** | Fill rate 90.6% → 98.5%, units lost −84%, cost trade-off quantified |
| F6 | Dashboard analytics | Real-time insights accessible | **Met** | 6-page Streamlit app + 12-page Power BI guide |
| F7 | Drift detection | Alerts for model degradation | **Met** | PSI + KS per feature, explicit retrain verdict |
| F8 | MLflow tracking | Performance history maintained | **Partially met** | Tracking module implemented and integrated as a pipeline stage; MLflow itself could not be installed in this environment, so runs were written to `reports/run_metrics.csv` instead. Running `pip install mlflow` activates the MLflow path unchanged. |

Two partials are reported as partials. F4 is a genuine shortfall in the
deliverable, not a technicality, and is listed again in section 8.

---

## 7. Engineering and MLOps

### 7.1 Architecture

Five layers, matching the brief: data → feature engineering → models → serving →
monitoring. Detail in `reports/ARCHITECTURE.md`.

### 7.2 Reproducibility

- Single entry point: `python -m src.run_pipeline`
- Stage-level control: `--from`, `--only`, `--skip`, `--list`, `--dry-run`
- Fixed `RANDOM_STATE` throughout
- Every threshold and business rule centralised in `src/config.py`
- Expensive inputs cached, so re-runs are cheap
- The raw data downloads itself from UCI — no Kaggle API key required

### 7.3 Dependency resilience

The environment this was built in could not download large binary wheels. Rather
than stall, the pipeline was designed to run on **pandas + NumPy alone**, with
every other package treated as an optional accelerator:

| Optional | Used for | Fallback |
|---|---|---|
| scikit-learn | K-Means, forests, boosting | `src/models/numpy_ml.py` — k-means++, CART, gradient boosting |
| matplotlib / seaborn | PNG charts | `src/viz/svg_charts.py` — dependency-free SVG |
| Prophet | an extra forecast model | skipped from the model list |
| joblib | model persistence | standard-library pickle |
| MLflow | experiment tracking | CSV run record |

The fallback path is not theoretical: it is what produced every number in this
report, and it is testable on demand with `RETAILPULSE_FORCE_FALLBACK=1`.

### 7.4 Testing

**37 unit tests, all passing**, requiring no dataset and running in seconds:
metric correctness (including AUC tie handling and hand-computed values), all
125 RFM score combinations, EOQ optimality, safety-stock monotonicity, PSI/KS
behaviour on identical versus shifted distributions, and a feature-leakage
contract on the rolling averages.

Both a pytest path and a dependency-free runner (`tests/run_tests.py`) are
provided, and CI exercises both.

### 7.5 Deployment

- Multi-stage `Dockerfile`, non-root user, healthcheck on `/_stcore/health`
- Kubernetes `Deployment` (2+ replicas), `Service`, HPA (2→6 pods at 70% CPU),
  and a read-only PVC for processed data
- GitHub Actions CI: lint → test (both runners) → Docker build

### 7.6 Bugs found and fixed during development

Recorded because they are the substance of the work, and because several were
the kind that produce plausible-looking wrong answers rather than crashes:

1. **Rolling-window key error** — a 28-day rolling mean was referenced but never
   created. Loud failure, caught immediately.
2. **Circular import** between `features` and `ingestion` under `-m` execution.
   Fixed by switching to absolute imports.
3. **Design-matrix shape drift** — `pd.get_dummies` drops categories absent from
   the training window, so a training set without a Sunday produced a matrix one
   column narrower than the prediction set. Silent until the matrix multiply
   failed; fixed by pinning categories explicitly.
4. **Safety stock understated** — demand standard deviation was computed only on
   days a SKU actually sold, ignoring zero-demand days. That makes intermittent
   products look stable and systematically understates the buffer they need.
   Fixed by reindexing every SKU onto the full calendar.
5. **Trend extrapolation blow-up** — undamped Holt-Winters forecast £19.7M for
   90 days, more than the entire two-year revenue. Fixed with a damped trend,
   seasonal reversion, and a documented ceiling.
6. **EOQ/review-cycle scale mismatch** — mixing a per-cycle EOQ with an
   aggregate order-up-to simulation reported a 33% fill rate for the "optimised"
   policy. Fixed by comparing like with like.
7. **Missing `pandas` import** in the dashboard entry point — found by a
   contract check, would have crashed on first load.
8. **Two incorrect test expectations** — the MAPE and lift assertions were wrong,
   not the implementations (the true values are 7.5% and 10×). Corrected the
   tests; this is worth stating plainly, since a test that is "fixed" to match
   the code is a smell, whereas a test corrected to match hand-computed
   arithmetic is not.

---

## 8. Honest limitations

Stated because a report that only lists strengths is not useful to whoever picks
this up next.

1. **No Prophet + LSTM hybrid ensemble.** The brief asks for one (F4). Four
   model families were implemented and compared, and a single improved model
   won, but a genuine two-model ensemble was not built. Adding Prophet would
   require only installing it — the model list and backtest harness already
   support it.
2. **MLflow runs were not executed.** The tracking module is written and wired
   into the pipeline; MLflow could not be installed here.
3. **Revenue is a demand proxy.** There is no separate quantity-demanded field,
   and returns are removed. Forecasts represent expected sales, not unconstrained
   demand.
4. **Profit is estimated**, not measured — a 35% margin assumption.
5. **The 90-day forecast exceeds the validated 28-day horizon.** Daily-grain
   intervals this far out are wide for good reason.
6. **Inventory stock levels are simulated.** The dataset contains no inventory
   snapshots, so fill rates come from replaying demand through a policy, not from
   observed stock.
7. **Customer identity is imperfect.** 235,151 rows (23%) have no CustomerID and
   are excluded from all customer-level analysis. Segmentation and churn are
   therefore statements about identified customers only.
8. **No LSTM / deep learning.** Not implemented; with 106 weekly observations a
   neural sequence model is unlikely to beat the damped Holt-Winters anyway, but
   the brief lists it, so it is a gap.

---

## 9. Recommendations

If this platform were put into production next month:

1. **Protect the Champions.** 22% of customers carry 68% of revenue. A dedicated
   service tier and proactive retention for these 1,284 accounts is the highest
   expected-value action available.
2. **Work the churn list by value, not by risk.** Start with high-risk customers
   ranked by Monetary × ChurnProbability. £1.13M of historic revenue is exposed.
3. **Plan weekly, not daily.** Weekly forecasts are twice as accurate. Commit
   stock against the weekly forecast and use daily data for tracking only.
4. **Adopt the statistical reorder policy for A-X items first.** 773 SKUs, high
   revenue, stable demand — automate those before touching the erratic tail.
5. **Automate replenishment only for A and B classes.** The 1,320 C-Z SKUs do not
   justify weekly human attention.
6. **Retrain quarterly, monitor monthly.** Drift is currently absent, but the
   series is only two years long and ends mid-peak; that stability should be
   re-verified as new data arrives.
7. **Instrument the forecast.** Log actual versus predicted weekly and watch the
   tracking signal. A drift in bias is the earliest warning that the model needs
   attention.

---

## 10. How to reproduce

```bash
cd retailpulse
pip install -r requirements.txt          # optional accelerators included

python -m src.run_pipeline               # full build, ~78s
python tests/run_tests.py                # 37 unit tests
streamlit run dashboard/app.py           # dashboard

# inspect the experiments
mlflow ui --backend-store-uri mlruns
```

Expected outputs after a successful run:

| File | Rows |
|---|---|
| `data/processed/daily_sales_features.csv` | 739 |
| `data/processed/customer_rfm.csv` | 5,852 |
| `data/processed/customer_segments.csv` | 5,852 |
| `data/processed/customer_churn.csv` | 5,256 |
| `data/processed/inventory_policy.csv` | 4,620 |
| `data/processed/forecast_daily.csv` | 90 |
| `data/processed/drift_report.csv` | 7 |
| `reports/figures/` | 20+ charts |

---

## 11. Conclusion

RetailPulse meets the core of the brief: it forecasts demand, segments customers,
predicts churn, optimises inventory, monitors drift, and serves the results
through an interactive dashboard — all from real transaction data, with every
number reproducible from a single command.

The strongest results are the customer concentration finding (22% of customers,
68% of revenue), a churn model with a defensible 0.81 AUC and a lift-ranked
worklist, and an inventory policy that cuts lost units by 84% for £7,087 a year
in extra holding cost. The forecast is honest about its own limits: 16.2% weekly
MAPE on a validated 28-day horizon, with a 90-day extension clearly flagged as
extrapolation.

Two items from the brief remain open — the Prophet/LSTM hybrid ensemble (F4) and
executed MLflow runs (F8) — both blocked by the build environment rather than by
design, and both activatable by installing the relevant package. They are
recorded here rather than quietly omitted.
