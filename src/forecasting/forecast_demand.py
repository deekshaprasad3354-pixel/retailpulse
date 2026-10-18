"""Stage 5 — Demand forecasting.

The goal is to predict future revenue/demand with an honest accuracy number
behind it. Because the source has no true "demand" column, **revenue** is used
as the demand proxy, with the caveat documented in the report.

Four model families are compared on the same rolling-origin backtest:

    Seasonal naive   - repeat last week. The bar every model must clear.
    Linear trend+seasonality - OLS on trend, day-of-week, month, Fourier terms.
    Holt-Winters     - triple exponential smoothing (level/trend/season).
    Prophet          - additive/multiplicative decomposable model (if installed).

Backtesting is **rolling-origin**: train on everything up to *t*, forecast the
next window, roll forward, repeat. A single train/test split can flatter a model
by luck; several folds give a distribution.

Outputs:
    data/processed/forecast_backtest.csv      per-fold accuracy for every model
    data/processed/model_comparison.csv       accuracy leaderboard
    data/processed/forecast_daily.csv         chosen model, daily horizon
    data/processed/forecast_weekly_12w.csv    chosen model, 12-week horizon
    data/processed/prophet_forecast_full.csv  fitted + forecast (for the UI)
    models/forecast_*.joblib
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config as cfg
from src.evaluation.metrics import evaluate, seasonal_naive_forecast


# ==========================================================================
# Model 1 — Linear regression on engineered calendar features (NumPy OLS)
# ==========================================================================
def _fourier(idx: np.ndarray, period: float, order: int) -> np.ndarray:
    """Sine/cosine pairs that let a linear model represent smooth seasonality."""
    t = idx / period
    cols = []
    for k in range(1, order + 1):
        cols.append(np.sin(2 * np.pi * k * t))
        cols.append(np.cos(2 * np.pi * k * t))
    return np.column_stack(cols)


def _dummies(values: pd.Series, categories: list, prefix: str) -> np.ndarray:
    """One-hot encode against a FIXED category list.

    This matters: `pd.get_dummies` drops categories that happen to be absent, so
    a training window without a Sunday would produce a matrix with one fewer
    column than the prediction window — and the matrix multiply would fail.
    Pinning the categories keeps the design matrix shape constant.
    """
    cat = pd.Categorical(values, categories=categories)
    return pd.get_dummies(cat, prefix=prefix, drop_first=True).to_numpy(float)


def _design(dates: pd.DatetimeIndex, day_index: np.ndarray, span: int,
            level: np.ndarray | None = None) -> np.ndarray:
    """Build the feature matrix.

    Two deliberate choices make this much stronger than a plain global trend:

    * **Day-of-year Fourier terms with a fixed calendar origin.** Deriving the
      angle from `dayofyear` rather than from an arbitrary index inside the
      training window means "late November" maps to the same phase in every
      year, so seasonality is learned across years instead of being re-invented
      inside each window.
    * **An explicit recent-level column.** A single global trend cannot follow a
      series that ramps into Q4 and eases in January. Passing a causal trailing
      average (already shifted by one day, so today never leaks into today) lets
      the same linear model anchor on the current trading level.
    """
    doy = dates.dayofyear.to_numpy(dtype=float)
    dow = _dummies(dates.dayofweek, list(range(7)), "dow")
    month = _dummies(dates.month, list(range(1, 13)), "m")

    feats = [
        np.ones((len(dates), 1)),                        # intercept
        day_index.reshape(-1, 1) / max(span, 1),         # long-run trend
        _fourier(doy, 365.25, 4),                        # yearly seasonality
        dow,
        month,
        _fourier(np.arange(len(dates), dtype=float), 7.0, 2),   # intra-week shape
        (dates.dayofweek >= 5).astype(float).reshape(-1, 1),    # weekend flag
    ]
    if level is not None:
        feats.append(np.nan_to_num(level, nan=0.0).reshape(-1, 1))
    return np.hstack(feats)


class LinearSeasonalModel:
    """OLS on trend + calendar effects + Fourier seasonality, fitted with NumPy."""

    name = "Linear (trend + seasonality)"

    def fit(self, dates: pd.DatetimeIndex, y: np.ndarray, day_index: np.ndarray,
            level: np.ndarray | None = None) -> "LinearSeasonalModel":
        self.origin = day_index[0]
        self.span = int(day_index[-1] - day_index[0]) + 1
        self.level_fallback = float(np.mean(y[-28:])) if len(y) >= 28 else float(np.mean(y))
        level = np.full(len(y), self.level_fallback) if level is None else level
        x = _design(dates, day_index - self.origin, self.span, level)
        # ridge-style stabiliser: keeps the normal equations solvable
        xtx = x.T @ x + np.eye(x.shape[1]) * 1e-6
        self.beta = np.linalg.solve(xtx, x.T @ y)
        self.resid_std = float(np.std(y - x @ self.beta))
        return self

    def predict(self, dates: pd.DatetimeIndex, day_index: np.ndarray,
                level: np.ndarray | None = None) -> np.ndarray:
        level = np.full(len(dates), self.level_fallback) if level is None else level
        x = _design(dates, day_index - self.origin, self.span, level)
        return x @ self.beta


# ==========================================================================
# Model 2 — Holt-Winters triple exponential smoothing (from scratch)
# ==========================================================================
class HoltWinters:
    """Additive Holt-Winters with a **damped** trend.

    Implemented directly because statsmodels is a heavy dependency for one
    model, and writing it out makes the level/trend/season logic inspectable.

    The damping factor `phi` matters more than it looks. An undamped additive
    trend compounds linearly: a series drifting up by 2,000 a day would be
    forecast 180,000 above the last observation 90 days out. Damping shrinks the
    trend contribution geometrically, so long horizons stay plausible. This is
    the standard fix (Gardner & McKenzie) for exactly this failure mode.
    """

    name = "Holt-Winters (damped)"

    def __init__(self, season: int = 7, alpha: float = 0.35, beta: float = 0.02,
                 gamma: float = 0.30, phi: float = 0.70) -> None:
        self.season = season
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.phi = phi

    def fit(self, y: np.ndarray) -> "HoltWinters":
        y = np.asarray(y, dtype=float)
        n = len(y)
        season = min(self.season, max(n // 2, 1))

        # initialise level/trend from the first two seasons, seasonals from the
        # average seasonal deviation (a standard, simple initialisation)
        first = y[:season]
        second = y[season:2 * season] if n >= 2 * season else first
        self.level = float(first.mean())
        self.trend = float((second.mean() - first.mean()) / max(season, 1))
        season_avg = float(y[:season].mean())
        self.seasonals = list(y[:season] - season_avg)

        # one-step-ahead in-sample errors, used later for interval width
        errors = []
        for i in range(n):
            s = self.seasonals[i % season]
            errors.append(y[i] - (self.level + self.phi * self.trend + s))
            last_level = self.level
            self.level = self.alpha * (y[i] - s) + (1 - self.alpha) * (self.level + self.phi * self.trend)
            self.trend = self.beta * (self.level - last_level) + (1 - self.beta) * self.phi * self.trend
            self.seasonals[i % season] = self.gamma * (y[i] - self.level) + (1 - self.gamma) * s

        self._season = season
        self.resid_std = float(np.std(errors))
        # A robust reference level (median of the last 8 weeks) and a robust
        # ceiling. Retail demand is strongly seasonal, so a trend estimated
        # during the pre-Christmas ramp keeps compounding well past anything the
        # business has seen. Capping the projection keeps it inside the range of
        # observed reality. The 99.5th percentile is used rather than the raw
        # maximum so that one exceptional bulk order does not become the ceiling
        # for a 90-day projection.
        tail = y[-56:] if n >= 56 else y
        self.level_reference = float(np.median(tail))
        self.historical_peak = float(np.quantile(y, 0.995))
        return self

    def fit_monthly_seasonality(self, dates: pd.DatetimeIndex, y: np.ndarray) -> "HoltWinters":
        """Per-month typical **level**, used as a reversion target.

        Holt-Winters has weekly seasonality only, so on its own it cannot know
        that January follows December — on a series that peaks in
        November-December, that omission makes a long projection hold Christmas
        level forever.

        These month levels are deliberately *not* multiplied into the forecast:
        a training window that never contains, say, July would default July to
        1.0 and distort the level. (Measured: doing so pushed daily MAPE from
        35% to 52%, worse than the seasonal-naive baseline.) Instead they act as
        a **reversion target** for the long horizon, where reverting toward a
        typical January clearly beats holding December indefinitely.
        """
        s = pd.Series(np.asarray(y, dtype=float), index=pd.DatetimeIndex(dates))
        overall = float(s.median()) if len(s) else 0.0
        if overall <= 0:
            self.month_levels = {}
            return self

        by_month = s.groupby(s.index.month).median()
        self.month_levels = {int(m): float(v) for m, v in by_month.items()}
        self.overall_level = float(overall)
        return self

    def predict(self, horizon: int, start_index: int = 0,
                future_dates: pd.DatetimeIndex | None = None,
                reversion: float = 1.5) -> np.ndarray:
        """Forecast with a geometrically damped trend and long-run reversion.

        The cumulative trend contribution at step H is phi + phi^2 + ... + phi^H,
        which converges to phi/(1-phi) instead of growing linearly without bound.

        A damped trend still holds the level seen at the end of the training
        data. When the series ends inside a seasonal peak that is the wrong
        shape for a 90-day projection, so the forecast is progressively blended
        toward the typical level of the month being predicted. The blend weight
        grows with the square root of the horizon: negligible over a couple of
        weeks, substantial by 90 days.
        """
        out = []
        cumulative = 0.0
        for h in range(1, horizon + 1):
            cumulative += self.phi ** h
            s = self.seasonals[(start_index + h - 1) % self._season]
            out.append(self.level + cumulative * self.trend + s)

        values = np.maximum(np.array(out, dtype=float), 0.0)

        month_levels = getattr(self, "month_levels", None)
        if future_dates is not None and len(future_dates) == horizon and month_levels and reversion > 0:
            targets = np.array([month_levels.get(int(d.month), self.level) for d in future_dates])
            weight = np.clip(reversion * np.sqrt(np.arange(1, horizon + 1) / horizon), 0.0, 0.85)
            values = values * (1 - weight) + targets * weight

        return np.maximum(values, 0.0)

    def predict_bounded(self, horizon: int, start_index: int = 0,
                        peak_multiple: float = 1.0,
                        future_dates: pd.DatetimeIndex | None = None,
                        reversion: float = 1.5) -> np.ndarray:
        """Damped, reverting forecast with a sanity ceiling.

        The cap is `peak_multiple` x a robust high quantile of observed daily
        revenue. This is a guard against model artefacts, not a modelling
        choice, and it is stated in the report rather than hidden.
        """
        raw = self.predict(horizon, start_index, future_dates, reversion)
        ceiling = max(self.historical_peak * peak_multiple, self.level_reference)
        return np.minimum(raw, ceiling)


# ==========================================================================
# Model 3 — Prophet
# ==========================================================================
def fit_prophet(train: pd.DataFrame, horizon: int, weekly: bool = False) -> tuple[np.ndarray, pd.DataFrame] | None:
    """Fit Prophet on a ds/y frame. Returns (point_forecast, full_frame) or None."""
    try:
        from prophet import Prophet
    except ImportError:
        return None

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=not weekly,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        interval_width=0.90,
    )
    m.fit(train[["ds", "y"]])
    future = m.make_future_dataframe(periods=horizon, freq="W-SUN" if weekly else "D")
    fc = m.predict(future)
    full = fc[["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]].copy()
    tail = full.tail(horizon)
    point = np.maximum(tail["yhat"].to_numpy(float), 0.0)
    return point, full


# ==========================================================================
# Backtesting
# ==========================================================================
def _trailing_level(y: np.ndarray, window: int = 28) -> np.ndarray:
    """Mean of the previous `window` days, ending yesterday (no leakage)."""
    s = pd.Series(y, dtype=float).shift(1)
    return s.rolling(window, min_periods=1).mean().to_numpy()


def rolling_backtest(
    daily: pd.DataFrame,
    models: list[str],
    folds: int = cfg.BACKTEST_FOLDS,
    horizon: int = 28,
    min_train: int = 365,
) -> tuple[pd.DataFrame, dict]:
    """Rolling-origin backtest over the daily series.

    Returns the per-fold metric table and the last fold's forecast per model
    (used to show a concrete prediction vs actual picture).
    """
    y_all = daily["Revenue"].to_numpy(float)
    dates_all = pd.DatetimeIndex(daily["Date"])
    n = len(y_all)

    if n < min_train + horizon * folds:
        min_train = max(int(n * 0.5), 30)

    rows, last_forecasts = [], {}
    for fold in range(folds):
        test_end = n - (folds - 1 - fold) * horizon
        test_start = test_end - horizon
        train_end = test_start
        if train_end < min_train:
            continue

        y_train = y_all[:train_end]
        y_test = y_all[test_start:test_end]
        d_train = dates_all[:train_end]
        d_test = dates_all[test_start:test_end]
        idx_train = np.arange(train_end)
        idx_test = np.arange(test_start, test_end)

        preds: dict[str, np.ndarray] = {}

        # Causal trailing level: the mean of the previous 28 days, so the value
        # known at training end is the one carried into the forecast horizon.
        level_train = _trailing_level(y_train, window=28)
        level_anchor = float(level_train[-1]) if len(level_train) else float(np.mean(y_train))
        level_test = np.full(horizon, level_anchor)

        if "Seasonal Naive" in models:
            preds["Seasonal Naive"] = seasonal_naive_forecast(y_train, horizon, 7)

        if "Linear (trend + seasonality)" in models:
            try:
                lin = LinearSeasonalModel().fit(d_train, y_train, idx_train, level_train)
                preds["Linear (trend + seasonality)"] = np.maximum(
                    lin.predict(d_test, idx_test, level_test), 0.0)
            except np.linalg.LinAlgError:
                pass

        if "Holt-Winters (damped)" in models:
            hw = HoltWinters(season=7).fit(y_train).fit_monthly_seasonality(d_train, y_train)
            preds["Holt-Winters (damped)"] = hw.predict_bounded(
                horizon, start_index=train_end, future_dates=d_test)

        if "Prophet" in models:
            train_df = pd.DataFrame({"ds": d_train, "y": y_train})
            out = fit_prophet(train_df, horizon)
            if out is not None:
                preds["Prophet"] = out[0]
                prophet_last = out[1]

        for name, p in preds.items():
            m = evaluate(y_test, p)
            rows.append({"fold": fold + 1, "model": name, "train_days": train_end,
                         "test_start": str(d_test[0].date()), "test_end": str(d_test[-1].date()), **m})

        last_forecasts = {
            "dates": d_test,
            "actual": y_test,
            "preds": preds,
            "train_dates": d_train,
            "train": y_train,
        }

    return pd.DataFrame(rows), last_forecasts


def summarise_backtest(bt: pd.DataFrame) -> pd.DataFrame:
    """Leaderboard: mean and spread of each metric across folds."""
    agg = bt.groupby("model").agg(
        folds=("fold", "count"),
        mae=("mae", "mean"),
        rmse=("rmse", "mean"),
        mape=("mape", "mean"),
        mape_std=("mape", "std"),
        wape=("wape", "mean"),
        bias=("bias", "mean"),
        r2=("r2", "mean"),
    ).round(3).reset_index()
    return agg.sort_values("mape").reset_index(drop=True)


# ==========================================================================
# Final fit + forward forecast
# ==========================================================================
def final_forecast(daily: pd.DataFrame, model_name: str, horizon: int) -> pd.DataFrame:
    """Refit the winner on the full history and forecast `horizon` days ahead."""
    y = daily["Revenue"].to_numpy(float)
    dates = pd.DatetimeIndex(daily["Date"])
    idx = np.arange(len(y))
    future_dates = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    future_idx = np.arange(len(y), len(y) + horizon)

    if model_name == "Linear (trend + seasonality)":
        level_hist = _trailing_level(y, window=28)
        level_anchor = float(level_hist[-1])
        level_future = np.full(horizon, level_anchor)
        m = LinearSeasonalModel().fit(dates, y, idx, level_hist)
        pred = np.maximum(m.predict(future_dates, future_idx, level_future), 0.0)
        # Prediction interval widens with horizon, as real forecast error does.
        resid = float(np.std(y - m.predict(dates, idx, level_hist)))
        widen = np.sqrt(np.arange(1, horizon + 1))
        lower, upper = pred - 1.645 * resid * widen, pred + 1.645 * resid * widen

    elif model_name == "Holt-Winters (damped)":
        m = HoltWinters(season=7).fit(y).fit_monthly_seasonality(dates, y)
        pred = m.predict_bounded(horizon, start_index=len(y), future_dates=future_dates)
        # residual spread from the model's own one-step-ahead errors, widened
        # with the square root of the horizon (error grows like a random walk)
        resid = max(m.resid_std, float(np.std(np.diff(y))) / np.sqrt(2))
        widen = np.sqrt(np.arange(1, horizon + 1))
        lower, upper = pred - 1.645 * resid * widen, pred + 1.645 * resid * widen

    elif model_name == "Prophet":
        train = pd.DataFrame({"ds": dates, "y": y})
        out = fit_prophet(train, horizon)
        if out is None:
            raise RuntimeError("Prophet is not installed; choose another model.")
        _, full = out
        tail = full.tail(horizon)
        pred = np.maximum(tail["yhat"].to_numpy(float), 0.0)
        lower = np.maximum(tail["yhat_lower"].to_numpy(float), 0.0)
        upper = np.maximum(tail["yhat_upper"].to_numpy(float), 0.0)

    else:  # Seasonal Naive
        pred = seasonal_naive_forecast(y, horizon, 7)
        resid = float(np.std(y[-56:] - seasonal_naive_forecast(y[:-horizon] if len(y) > horizon else y,
                                                              min(horizon, len(y)), 7)[:min(horizon, len(y))])) \
            if len(y) > 60 else float(np.std(y) * 0.2)
        lower, upper = pred - 1.645 * resid, pred + 1.645 * resid

    return pd.DataFrame({
        "Date": future_dates,
        "Forecast": np.round(pred, 2),
        "Lower90": np.round(np.maximum(lower, 0), 2),
        "Upper90": np.round(np.maximum(upper, 0), 2),
        "Model": model_name,
    })


def to_weekly(daily_fc: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a daily forecast to weeks for planning conversations."""
    d = daily_fc.set_index("Date")
    wk = d.resample("W-SUN").agg(
        Forecast=("Forecast", "sum"), Lower90=("Lower90", "sum"), Upper90=("Upper90", "sum"),
    ).reset_index()
    wk["WeekLabel"] = wk["Date"].dt.strftime("W%V %Y")
    return wk


def weekly_backtest(weekly: pd.DataFrame, models: list[str], folds: int = 4,
                    horizon: int = 4, min_train: int = 60) -> tuple[pd.DataFrame, dict]:
    """Same rolling-origin idea, on the weekly series (4-week-ahead horizon)."""
    y_all = weekly["Revenue"].to_numpy(float)
    dates_all = pd.DatetimeIndex(weekly["Date"])
    n = len(y_all)
    rows, last = [], {}
    if n < min_train + horizon * folds:
        min_train = max(int(n * 0.5), 20)

    for fold in range(folds):
        test_end = n - (folds - 1 - fold) * horizon
        test_start = test_end - horizon
        train_end = test_start
        if train_end < min_train:
            continue
        y_train, y_test = y_all[:train_end], y_all[test_start:test_end]
        d_train, d_test = dates_all[:train_end], dates_all[test_start:test_end]
        idx_train = np.arange(train_end)
        idx_test = np.arange(test_start, test_end)
        preds = {}

        # Weekly series: a 4-week trailing level replaces the 28-day one
        level_train = _trailing_level(y_train, window=4)
        level_anchor = float(level_train[-1]) if len(level_train) else float(np.mean(y_train))
        level_test = np.full(horizon, level_anchor)

        if "Seasonal Naive" in models:
            preds["Seasonal Naive"] = seasonal_naive_forecast(y_train, horizon, 4)
        if "Linear (trend + seasonality)" in models:
            try:
                m = LinearSeasonalModel().fit(d_train, y_train, idx_train, level_train)
                preds["Linear (trend + seasonality)"] = np.maximum(
                    m.predict(d_test, idx_test, level_test), 0.0)
            except np.linalg.LinAlgError:
                pass
        if "Holt-Winters (damped)" in models:
            preds["Holt-Winters (damped)"] = (
                HoltWinters(season=4).fit(y_train)
                .fit_monthly_seasonality(d_train, y_train)
                .predict_bounded(horizon, start_index=train_end, future_dates=d_test)
            )
        if "Prophet" in models:
            out = fit_prophet(pd.DataFrame({"ds": d_train, "y": y_train}), horizon, weekly=True)
            if out is not None:
                preds["Prophet"] = out[0]

        for name, p in preds.items():
            rows.append({"fold": fold + 1, "model": name, "granularity": "weekly", **evaluate(y_test, p)})
        last = {"dates": d_test, "actual": y_test, "preds": preds}

    return pd.DataFrame(rows), last


def _make_svg_figures(daily: pd.DataFrame, bt_summary: pd.DataFrame, last: dict,
                      daily_fc: pd.DataFrame, bt: pd.DataFrame) -> None:
    """Dependency-free SVG fallback for the forecasting charts."""
    from src.viz import bar_chart, line_chart, save

    hist = daily.tail(240)
    save(line_chart(
        {f"forecast ({daily_fc['Model'].iloc[0]})": (daily_fc["Date"].tolist(),
                                                     daily_fc["Forecast"].tolist()),
         "actual": (hist["Date"].tolist(), hist["Revenue"].tolist())},
        "Daily revenue — recent history and forward forecast",
        y_label="revenue (GBP)",
        band=(daily_fc["Date"].tolist(), daily_fc["Lower90"].tolist(),
              daily_fc["Upper90"].tolist())),
        cfg.FIGURES / "forecast_horizon.svg")

    if last.get("preds"):
        series = {"actual": (last["dates"].tolist(), last["actual"].tolist())}
        for name, p in last["preds"].items():
            series[name] = (last["dates"].tolist(), list(np.asarray(p, dtype=float)))
        save(line_chart(series, "Backtest — final fold: every model vs actual",
                        y_label="revenue (GBP)"),
             cfg.FIGURES / "forecast_backtest_models.svg")

    d = bt_summary.sort_values("mape")
    save(bar_chart(d["model"].tolist(), d["mape"].tolist(),
                   "Forecast accuracy by model (lower MAPE is better)",
                   horizontal=True, value_suffix="%"),
         cfg.FIGURES / "forecast_model_comparison.svg")

    if not bt.empty:
        folds = sorted(bt["fold"].unique())
        series = {
            model: ([int(f) for f in grp["fold"]], grp["mape"].tolist())
            for model, grp in bt.groupby("model")
        }
        save(line_chart(series, "Accuracy stability across backtest folds",
                        x_label="backtest fold", y_label="MAPE %"),
             cfg.FIGURES / "forecast_fold_stability.svg")


def make_figures(daily: pd.DataFrame, bt_summary: pd.DataFrame, last: dict,
                 daily_fc: pd.DataFrame, bt: pd.DataFrame) -> None:
    from src.optional_deps import MATPLOTLIB

    if not MATPLOTLIB.ok:
        _make_svg_figures(daily, bt_summary, last, daily_fc, bt)
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")

    # 1. history + forecast with interval
    fig, ax = plt.subplots(figsize=(14, 5.5))
    hist = daily.tail(240)
    ax.plot(hist["Date"], hist["Revenue"], color="#0f172a", lw=1.1, label="actual daily revenue")
    ax.plot(daily_fc["Date"], daily_fc["Forecast"], color="#2563eb", lw=1.8,
            label=f"forecast ({daily_fc['Model'].iloc[0]})")
    ax.fill_between(daily_fc["Date"], daily_fc["Lower90"], daily_fc["Upper90"],
                    color="#2563eb", alpha=0.18, label="90% interval")
    ax.set_title("Daily revenue — recent history and forward forecast", fontsize=13, weight="bold")
    ax.set_ylabel("revenue (GBP)"); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(cfg.FIGURES / "forecast_horizon.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # 2. backtest: forecast vs actual on the last fold, all models
    if last.get("preds"):
        fig, ax = plt.subplots(figsize=(14, 5.5))
        ax.plot(last["dates"], last["actual"], "o-", color="#0f172a", lw=2, label="actual", zorder=5)
        colors = sns.color_palette("husl", len(last["preds"]))
        for (name, p), c in zip(last["preds"].items(), colors):
            ax.plot(last["dates"], p, "--o", color=c, lw=1.6, ms=3, label=name)
        ax.set_title("Backtest — final fold: every model vs actual", fontsize=13, weight="bold")
        ax.set_ylabel("revenue (GBP)"); ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(cfg.FIGURES / "forecast_backtest_models.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    # 3. accuracy leaderboard
    fig, ax = plt.subplots(figsize=(9, 4.5))
    d = bt_summary.sort_values("mape")
    sns.barplot(data=d, x="mape", y="model", ax=ax, palette="Blues_r")
    ax.axvline(10, color="#dc2626", ls="--", lw=1.2)
    ax.text(10.2, len(d) - 0.5, "10% target", color="#dc2626", fontsize=8)
    ax.set_title("Forecast accuracy by model (lower MAPE is better)", fontsize=12, weight="bold")
    ax.set_xlabel("MAPE %"); ax.set_ylabel("")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.1f%%", padding=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(cfg.FIGURES / "forecast_model_comparison.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # 4. MAPE by fold — shows stability, not just the average
    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns.lineplot(data=bt, x="fold", y="mape", hue="model", marker="o", ax=ax)
    ax.set_title("Accuracy stability across backtest folds", fontsize=12, weight="bold")
    ax.set_ylabel("MAPE %"); ax.set_xlabel("backtest fold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(cfg.FIGURES / "forecast_fold_stability.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ==========================================================================
# Entry point
# ==========================================================================
def main() -> None:
    print("=" * 78)
    print("RetailPulse :: Stage 5 - demand forecasting")
    print("=" * 78)

    daily = pd.read_csv(cfg.PROCESSED / "daily_sales_features.csv", parse_dates=["Date"])
    weekly = pd.read_csv(cfg.PROCESSED / "weekly_sales_features.csv", parse_dates=["Date"])
    print(f"[forecast] daily rows {len(daily):,} | weekly rows {len(weekly):,}")

    models = ["Seasonal Naive", "Linear (trend + seasonality)", "Holt-Winters (damped)", "Prophet"]

    print("\n[forecast] rolling-origin backtest (daily, 28-day horizon) ...")
    bt, last = rolling_backtest(daily, models, folds=cfg.BACKTEST_FOLDS, horizon=28,
                                min_train=300)
    summary = summarise_backtest(bt)
    print("\n--- daily backtest leaderboard ---")
    print(summary.to_string(index=False))

    print("\n[forecast] rolling-origin backtest (weekly, 4-week horizon) ...")
    wbt, wlast = weekly_backtest(weekly, models, folds=cfg.BACKTEST_FOLDS, horizon=4)
    wsummary = summarise_backtest(wbt) if not wbt.empty else pd.DataFrame()
    if not wsummary.empty:
        print("\n--- weekly backtest leaderboard ---")
        print(wsummary.to_string(index=False))

    best = summary.iloc[0]["model"]
    print(f"\n[forecast] best model by MAPE: {best}")

    horizon = cfg.FORECAST_HORIZON_DAYS
    print(f"[forecast] fitting {best} on full history, projecting {horizon} days ...")
    daily_fc = final_forecast(daily, best, horizon)
    weekly_fc = to_weekly(daily_fc)

    print("\n--- next 14 days ---")
    print(daily_fc.head(14).to_string(index=False))
    print("\n--- next 12 weeks ---")
    print(weekly_fc.head(12).to_string(index=False))

    total_next_90 = daily_fc["Forecast"].sum()
    print(f"\n[forecast] projected revenue next {horizon} days: "
          f"{cfg.CURRENCY_SYMBOL}{total_next_90:,.0f}")
    print(f"[forecast] 90% interval: {cfg.CURRENCY_SYMBOL}{daily_fc['Lower90'].sum():,.0f} "
          f"to {cfg.CURRENCY_SYMBOL}{daily_fc['Upper90'].sum():,.0f}")

    bt.to_csv(cfg.PROCESSED / "forecast_backtest.csv", index=False)
    summary.to_csv(cfg.PROCESSED / "model_comparison_daily.csv", index=False)
    if not wbt.empty:
        wbt.to_csv(cfg.PROCESSED / "forecast_backtest_weekly.csv", index=False)
        wsummary.to_csv(cfg.PROCESSED / "model_comparison_weekly.csv", index=False)
    daily_fc.to_csv(cfg.PROCESSED / "forecast_daily.csv", index=False)
    weekly_fc.to_csv(cfg.PROCESSED / "forecast_weekly_12w.csv", index=False)

    # Prophet full fitted+forecast frame, for the dashboard
    try:
        train = pd.DataFrame({"ds": pd.DatetimeIndex(daily["Date"]), "y": daily["Revenue"].to_numpy(float)})
        out = fit_prophet(train, cfg.FORECAST_HORIZON_DAYS)
        if out is not None:
            _, full = out
            full.to_csv(cfg.PROCESSED / "prophet_forecast_full.csv", index=False)
            print("[forecast] prophet full frame written")
    except Exception as exc:
        print(f"[forecast] prophet full frame skipped: {exc}")

    make_figures(daily, summary, last, daily_fc, bt)

    try:
        from src.models.persistence import save_model

        written = save_model(
            {"model": best, "daily": daily_fc, "weekly": weekly_fc,
             "metrics": summary.iloc[0].to_dict()},
            cfg.MODELS / "forecast_best.joblib",
        )
        print(f"[forecast] saved {written.name}")
    except Exception as exc:
        print(f"[forecast] model persistence skipped: {exc}")

    print(f"[forecast] wrote outputs -> {cfg.PROCESSED}")


if __name__ == "__main__":
    main()
