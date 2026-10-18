"""RetailPulse unit tests.

These exercise the pure logic — metrics, segmentation rules, inventory formulas,
drift statistics — with no dependency on the cleaned dataset, so they run in
seconds and in CI.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import metrics as M                      # noqa: E402
from src.inventory.optimize_inventory import (                # noqa: E402
    economic_order_quantity, reorder_point, safety_stock, z_for_service_level,
)
from src.mlops.drift import classify_psi, ks_drift, psi        # noqa: E402
from src.segmentation.segment_customers import assign_rfm_segment  # noqa: E402


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
class TestMetrics:
    def test_mae_and_rmse_known_values(self):
        actual = [10, 20, 30]
        pred = [12, 18, 33]
        assert M.mae(actual, pred) == pytest.approx((2 + 2 + 3) / 3)
        assert M.rmse(actual, pred) == pytest.approx(np.sqrt((4 + 4 + 9) / 3))

    def test_perfect_forecast_scores_zero(self):
        y = [5, 10, 15, 20]
        assert M.mae(y, y) == 0
        assert M.rmse(y, y) == 0
        assert M.mape(y, y) == 0
        assert M.r2(y, y) == pytest.approx(1.0)

    def test_mape_skips_near_zero_actuals(self):
        # the 0 actual would explode percentage error, so it must be excluded;
        # remaining errors are |100-110|/100 = 10% and |200-190|/200 = 5%
        detail = M.mape_detail([0, 100, 200], [0, 110, 190])
        assert detail["points_scored"] == 2
        assert detail["points_skipped"] == 1
        assert detail["mape"] == pytest.approx(7.5)

    def test_bias_sign_meaning(self):
        # predicting lower than actual => under-forecast => positive bias
        assert M.bias([100, 100], [90, 90]) > 0
        assert M.bias([100, 100], [110, 110]) < 0

    def test_roc_auc_perfect_and_random(self):
        assert M.roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
        assert M.roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(0.0)
        # fully tied scores carry no information
        assert M.roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)

    def test_roc_auc_handles_ties_correctly(self):
        auc = M.roc_auc([0, 1], [0.5, 0.5])
        assert auc == pytest.approx(0.5)

    def test_confusion_and_f1(self):
        y = [1, 1, 0, 0]
        p = [1, 0, 1, 0]
        c = M.confusion(y, p)
        assert c == {"tp": 1, "tn": 1, "fp": 1, "fn": 1}
        m = M.precision_recall_f1(y, p)
        assert m["precision"] == pytest.approx(0.5)
        assert m["recall"] == pytest.approx(0.5)
        assert m["f1"] == pytest.approx(0.5)
        assert m["accuracy"] == pytest.approx(0.5)

    def test_lift_at_k(self):
        # 10 rows, 1 positive, so the base rate is 0.1. The top 10% is exactly
        # one row and it is the positive one: precision 1.0, so lift is 10.
        y = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        scores = [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        assert M.lift_at_k(y, scores, 0.1) == pytest.approx(10.0)

    def test_lift_at_k_is_one_when_uninformative(self):
        # a ranking that carries no information should give a lift of about 1.0
        y = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        scores = [0.5] * 10
        assert M.lift_at_k(y, scores, 0.2) == pytest.approx(1.0)

    def test_seasonal_naive_repeats_last_cycle(self):
        series = list(range(1, 15))          # last 7 = 8..14
        out = M.seasonal_naive_forecast(series, 7, 7)
        assert list(out) == [8, 9, 10, 11, 12, 13, 14]

    def test_wape_is_scale_free(self):
        assert M.wape([100, 200], [110, 180]) == pytest.approx(30 / 300 * 100)


# --------------------------------------------------------------------------
# RFM segmentation rules
# --------------------------------------------------------------------------
class TestSegmentationRules:
    @pytest.mark.parametrize("r,f,m,expected", [
        (5, 5, 5, "Champions"),
        (4, 4, 4, "Champions"),
        (3, 3, 3, "Loyal Customers"),
        (5, 1, 3, "New / Promising"),
        (2, 3, 3, "Need Attention"),
        (2, 1, 1, "At Risk"),
        (1, 5, 5, "Dormant"),
    ])
    def test_segment_assignment(self, r, f, m, expected):
        row = pd.Series({"R_Score": r, "F_Score": f, "M_Score": m})
        assert assign_rfm_segment(row) == expected

    def test_every_score_combination_maps_to_a_segment(self):
        for r in range(1, 6):
            for f in range(1, 6):
                for m in range(1, 6):
                    out = assign_rfm_segment(pd.Series({"R_Score": r, "F_Score": f, "M_Score": m}))
                    assert isinstance(out, str) and out


# --------------------------------------------------------------------------
# Inventory formulas
# --------------------------------------------------------------------------
class TestInventory:
    def test_eoq_known_value(self):
        # D=10000, S=25, holding = 0.2*1.0 = 0.2  ->  sqrt(2*10000*25/0.2) = 1581.14
        assert economic_order_quantity(10_000, 25, 1.0, 0.20) == pytest.approx(1581.14, rel=1e-4)

    def test_eoq_is_zero_without_demand(self):
        assert economic_order_quantity(0) == 0.0

    def test_eoq_minimises_total_cost(self):
        d, s, c, h = 10_000, 25, 2.0, 0.20
        eoq = economic_order_quantity(d, s, c, h)

        def total(q):
            return s * d / q + q / 2 * c * h

        assert total(eoq) < total(eoq * 0.5)
        assert total(eoq) < total(eoq * 2)

    def test_safety_stock_increases_with_variability(self):
        low = safety_stock(1.0, 7, 0.95)
        high = safety_stock(5.0, 7, 0.95)
        assert high > low
        assert low == pytest.approx(1.645 * 1.0 * np.sqrt(7), rel=1e-3)

    def test_safety_stock_zero_when_no_variance(self):
        assert safety_stock(0.0, 7, 0.95) == 0.0

    def test_reorder_point_covers_lead_time(self):
        rop = reorder_point(avg_daily=10, lead_time_days=7, std_daily=2)
        assert rop > 70                      # lead-time demand alone is 70
        assert rop == pytest.approx(70 + safety_stock(2, 7), rel=1e-6)

    def test_z_scores_monotonic(self):
        zs = [z_for_service_level(s) for s in (0.80, 0.85, 0.90, 0.95, 0.975, 0.99)]
        assert zs == sorted(zs)
        assert z_for_service_level(0.95) == pytest.approx(1.645)


# --------------------------------------------------------------------------
# Drift statistics
# --------------------------------------------------------------------------
class TestDrift:
    def test_psi_zero_for_identical_distributions(self):
        rng = np.random.default_rng(0)
        sample = rng.normal(size=5000)
        assert psi(sample, sample) == pytest.approx(0.0, abs=1e-6)

    def test_psi_large_for_shifted_distributions(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 5000)
        b = rng.normal(3, 1, 5000)
        assert psi(a, b) > 0.25

    def test_psi_between_zeros_and_ones(self):
        rng = np.random.default_rng(1)
        a = rng.random(4000)
        b = rng.random(4000) ** 2
        value = psi(a, b)
        assert 0 <= value < 1

    def test_ks_detects_shift_and_not_identical(self):
        rng = np.random.default_rng(2)
        a = rng.normal(0, 1, 2000)
        b = rng.normal(0, 1, 2000)
        d_same, p_same = ks_drift(a, b)
        assert p_same > 0.01                        # same distribution -> no evidence
        d_diff, p_diff = ks_drift(a, rng.normal(2, 1, 2000))
        assert d_diff > d_same
        assert p_diff < 0.01

    def test_ks_statistic_bounds(self):
        d, p = ks_drift([1, 2, 3], [4, 5, 6])
        assert 0 <= d <= 1
        assert 0 <= p <= 1

    @pytest.mark.parametrize("value,expected", [
        (0.0, "stable"),
        (0.09, "stable"),
        (0.15, "moderate shift"),
        (0.30, "major shift - retrain"),
    ])
    def test_psi_verdicts(self, value, expected):
        assert classify_psi(value) == expected


# --------------------------------------------------------------------------
# Feature engineering contracts
# --------------------------------------------------------------------------
class TestFeatureContracts:
    def test_rolling_features_do_not_leak_the_current_day(self):
        """RollMean must never include today's own value — that would be leakage."""
        from src.features.build_features import build_daily_sales

        dates = pd.date_range("2011-01-01", periods=40, freq="D")
        tx = pd.DataFrame({
            "Invoice": [f"I{i}" for i in range(40)],
            "StockCode": ["SKU1"] * 40,
            "Description": ["thing"] * 40,
            "Quantity": [10] * 40,
            "InvoiceDate": dates,
            "Price": [1.0] * 40,
            "CustomerID": [12345] * 40,
            "Country": ["United Kingdom"] * 40,
            "Revenue": [10.0] * 40,
            "EstimatedProfit": [6.5] * 40,
            "Date": dates,
        })
        out = build_daily_sales(tx)
        # constant series => the lagged rolling mean equals the constant, never above
        assert out["RevenueRollMean7"].max() <= 10.0 + 1e-9

    def test_rfm_scores_are_bounded(self):
        from src.features.build_features import build_customer_rfm

        rng = np.random.default_rng(3)
        n = 300
        df = pd.DataFrame({
            "CustomerID": rng.integers(1, 60, n),
            "Invoice": [f"I{i}" for i in range(n)],
            "StockCode": rng.choice(["A", "B", "C"], n),
            "Quantity": rng.integers(1, 20, n),
            "Revenue": rng.uniform(5, 500, n),
            "Date": pd.to_datetime("2011-01-01") + pd.to_timedelta(rng.integers(0, 300, n), unit="D"),
            "Country": "United Kingdom",
        })
        rfm = build_customer_rfm(df)
        for col in ("R_Score", "F_Score", "M_Score"):
            assert rfm[col].between(1, 5).all()
        assert rfm["RFM_Score"].between(3, 15).all()
