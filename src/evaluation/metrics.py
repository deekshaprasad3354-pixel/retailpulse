"""Shared evaluation metrics.

Kept dependency-free (NumPy only) so every stage can score models the same way,
and so the definitions are explicit rather than hidden behind a library call.
"""
from __future__ import annotations

import numpy as np


def _clean(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def mae(y_true, y_pred) -> float:
    """Mean Absolute Error — average size of the miss, in the target's units."""
    a, b = _clean(y_true, y_pred)
    return float(np.mean(np.abs(a - b))) if len(a) else float("nan")


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error — like MAE but punishes big misses harder."""
    a, b = _clean(y_true, y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2))) if len(a) else float("nan")


def mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error, in percent.

    Points with |actual| < 1 are skipped because percentage error explodes near
    zero; the count of skipped points is reported by :func:`mape_detail`.
    """
    a, b = _clean(y_true, y_pred)
    mask = np.abs(a) >= 1
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((a[mask] - b[mask]) / a[mask])) * 100)


def mape_detail(y_true, y_pred) -> dict:
    a, b = _clean(y_true, y_pred)
    mask = np.abs(a) >= 1
    return {
        "mape": mape(a, b),
        "points_total": int(len(a)),
        "points_scored": int(mask.sum()),
        "points_skipped": int((~mask).sum()),
    }


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE — bounded 0-200%, safe when actuals approach zero."""
    a, b = _clean(y_true, y_pred)
    denom = (np.abs(a) + np.abs(b)) / 2
    mask = denom > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(a[mask] - b[mask]) / denom[mask]) * 100)


def r2(y_true, y_pred) -> float:
    """Coefficient of determination — share of variance explained."""
    a, b = _clean(y_true, y_pred)
    if len(a) < 2:
        return float("nan")
    ss_res = float(np.sum((a - b) ** 2))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def bias(y_true, y_pred) -> float:
    """Mean signed error. Positive => the model under-forecasts on average."""
    a, b = _clean(y_true, y_pred)
    return float(np.mean(a - b)) if len(a) else float("nan")


def tracking_signal(y_true, y_pred) -> float:
    """Forecast bias relative to error — a classic demand-planning control chart.

    Values inside +/-4 are normally considered 'in control'.
    """
    m = mae(y_true, y_pred)
    if not np.isfinite(m) or m == 0:
        return float("nan")
    return float(bias(y_true, y_pred) / m)


def naive_forecast(series, horizon: int) -> np.ndarray:
    """Seasonal-naive baseline: repeat the last observed week."""
    s = np.asarray(series, dtype=float)
    if len(s) < 7:
        return np.repeat(s[-1], horizon)
    return np.array([s[-7 + (i % 7)] for i in range(horizon)])


def seasonal_naive_forecast(series, horizon: int, season: int = 7) -> np.ndarray:
    """Repeat the last full seasonal cycle — the benchmark every model must beat."""
    s = np.asarray(series, dtype=float)
    if len(s) < season:
        return np.repeat(s[-1] if len(s) else 0.0, horizon)
    tail = s[-season:]
    return np.array([tail[i % season] for i in range(horizon)])


def wape(y_true, y_pred) -> float:
    """Weighted Absolute Percentage Error — robust for intermittent demand."""
    a, b = _clean(y_true, y_pred)
    denom = float(np.sum(np.abs(a)))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(a - b)) / denom * 100)


def evaluate(y_true, y_pred) -> dict:
    """One call, every regression metric we report."""
    return {
        "mae": round(mae(y_true, y_pred), 2),
        "rmse": round(rmse(y_true, y_pred), 2),
        "mape": round(mape(y_true, y_pred), 3),
        "smape": round(smape(y_true, y_pred), 3),
        "wape": round(wape(y_true, y_pred), 3),
        "r2": round(r2(y_true, y_pred), 4),
        "bias": round(bias(y_true, y_pred), 2),
    }


# --------------------------------------------------------------------------
# Classification metrics (churn)
# --------------------------------------------------------------------------
def confusion(y_true, y_pred) -> dict:
    a = np.asarray(y_true).astype(int)
    b = np.asarray(y_pred).astype(int)
    return {
        "tp": int(((a == 1) & (b == 1)).sum()),
        "tn": int(((a == 0) & (b == 0)).sum()),
        "fp": int(((a == 0) & (b == 1)).sum()),
        "fn": int(((a == 1) & (b == 0)).sum()),
    }


def precision_recall_f1(y_true, y_pred) -> dict:
    c = confusion(y_true, y_pred)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }


def roc_auc(y_true, y_score) -> float:
    """ROC AUC via the rank (Mann-Whitney U) formula — no sklearn needed.

    AUC = P(a random positive scores higher than a random negative), and 0.5
    means the model is no better than a coin flip.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=float)
    pos, neg = y == 1, y == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)

    # average ranks for ties, otherwise AUC is biased
    sorted_s = s[order]
    i = 0
    while i < len(sorted_s):
        j = i
        while j + 1 < len(sorted_s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1

    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def ks_statistic(y_true, y_score) -> float:
    """Kolmogorov-Smirnov separation between the two score distributions."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    grid = np.unique(s)
    cdf_pos = np.searchsorted(np.sort(pos), grid, side="right") / len(pos)
    cdf_neg = np.searchsorted(np.sort(neg), grid, side="right") / len(neg)
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def lift_at_k(y_true, y_score, fraction: float = 0.1) -> float:
    """How much better than random we are at catching churners in the top k%.

    A churn team can only call a limited number of customers, so this is often
    a more useful number than AUC.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=float)
    n = len(y)
    k = max(int(n * fraction), 1)
    top = np.argsort(-s)[:k]
    base = y.mean()
    if base == 0:
        return float("nan")
    return float(y[top].mean() / base)
