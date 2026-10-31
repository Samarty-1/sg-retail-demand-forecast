"""The four models in docs/PROTOCOL.md. Each forecasts horizons 1..12 from an origin,
using only data up to and including the origin month."""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .calendar import cny_holiday_share, cny_run_up_share

H = 12
MODELS = ["seasonal_naive", "naive_plus_growth", "ets", "lightgbm"]
LABELS = {
    "seasonal_naive": "Same month last year",
    "naive_plus_growth": "Last year + recent growth",
    "ets": "Exponential smoothing (ETS)",
    "lightgbm": "LightGBM with CNY calendar",
}


def seasonal_naive(y: np.ndarray, horizon: int = H) -> np.ndarray:
    if len(y) < 12:
        raise ValueError("need at least 12 months")
    last_year = y[-12:]
    return np.array([last_year[(h - 1) % 12] for h in range(1, horizon + 1)], dtype=float)


def naive_plus_growth(y: np.ndarray, horizon: int = H) -> np.ndarray:
    if len(y) < 24:
        raise ValueError("need at least 24 months")
    growth = y[-12:].sum() / y[-24:-12].sum()
    return seasonal_naive(y, horizon) * growth


def ets(y: np.ndarray, horizon: int = H, window: int = 120) -> np.ndarray:
    """Holt-Winters on log sales: damped trend, additive (i.e. multiplicative on sales)
    seasonality. Falls back to seasonal naive if the optimiser fails outright."""
    z = np.log(y[-window:])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ExponentialSmoothing(z, trend="add", damped_trend=True, seasonal="add",
                                       seasonal_periods=12, initialization_method="estimated").fit()
        out = np.exp(fit.forecast(horizon))
        if not np.all(np.isfinite(out)):
            raise FloatingPointError("non-finite forecast")
        return out
    except Exception:  # noqa: BLE001 - any fitting failure falls back to the baseline
        return seasonal_naive(y, horizon)


# ---------------------------------------------------------------- LightGBM (global)

FEATURES = ["h", "target_month", "growth_1m", "growth_3m", "growth_12m", "momentum",
            "cny_run_up_shift", "cny_holiday_shift", "series_code", "is_fnb"]
GBM_PARAMS = dict(objective="l1", n_estimators=400, learning_rate=0.03, num_leaves=15,
                  min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  random_state=7, verbose=-1)


def _cny_tables(months: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    run = np.array([cny_run_up_share(m) for m in months])
    hol = np.array([cny_holiday_share(m) for m in months])
    return run, hol


class GlobalGBM:
    """One model across every series. Target: log change of the target month against the
    same month a year earlier, so the model only learns the *correction* to seasonal naive."""

    def __init__(self, panel: pd.DataFrame):
        self.panel = panel  # index: month (DatetimeIndex), columns: series
        self.months = panel.index
        self.logy = np.log(panel.to_numpy(dtype=float))
        self.cny_run, self.cny_hol = _cny_tables(self.months)
        self.series = list(panel.columns)
        self.model: lgb.LGBMRegressor | None = None

    def _rows(self, s: int, t: int, hs: range) -> list[dict] | None:
        """Feature rows for series s at origin index t. None if history is too short."""
        ly = self.logy[:, s]
        if t < 24 or np.isnan(ly[t - 24]) or np.isnan(ly[t]):
            return None
        rows = []
        for h in hs:
            target = t + h
            base = target - 12  # same month last year: always <= t because h <= 12
            if target < len(self.months):
                month = self.months[target]
                run_now = self.cny_run[target]
                hol_now = self.cny_hol[target]
            else:  # forecasting past the end of the data
                month = self.months[t] + pd.DateOffset(months=h)
                run_now, hol_now = cny_run_up_share(month), cny_holiday_share(month)
            rows.append({
                "series_idx": s, "t": t, "h": h, "target": target,
                "target_month": month.month,
                "growth_1m": ly[t] - ly[t - 12],
                "growth_3m": np.log(np.exp(ly[t - 2:t + 1]).sum() / np.exp(ly[t - 14:t - 11]).sum()),
                "growth_12m": np.log(np.exp(ly[t - 11:t + 1]).sum() / np.exp(ly[t - 23:t - 11]).sum()),
                "momentum": (ly[t] - ly[t - 1]) - (ly[t - 12] - ly[t - 13]),
                "cny_run_up_shift": run_now - self.cny_run[base],
                "cny_holiday_shift": hol_now - self.cny_hol[base],
                "series_code": s,
                "is_fnb": int(self.series[s].startswith("Food")),
                "base_log": ly[base],
            })
        return rows

    def fit(self, cutoff: int) -> "GlobalGBM":
        """Train on every (origin, horizon) whose target month is at or before `cutoff`."""
        rows = []
        for s in range(len(self.series)):
            for t in range(24, cutoff):
                r = self._rows(s, t, range(1, min(H, cutoff - t) + 1))
                if r:
                    rows.extend(r)
        df = pd.DataFrame(rows)
        y = np.array([self.logy[r.target, r.series_idx] for r in df.itertuples()]) - df["base_log"].to_numpy()
        keep = np.isfinite(y)
        X = df.loc[keep, FEATURES].astype({"series_code": "category"})
        X["series_code"] = X["series_code"].cat.set_categories(range(len(self.series)))
        self.model = lgb.LGBMRegressor(**GBM_PARAMS).fit(X, y[keep])
        return self

    def predict(self, s: int, t: int, horizon: int = H) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit first")
        rows = self._rows(s, t, range(1, horizon + 1))
        if rows is None:
            raise ValueError("not enough history")
        df = pd.DataFrame(rows)
        X = df[FEATURES].astype({"series_code": "category"})
        X["series_code"] = X["series_code"].cat.set_categories(range(len(self.series)))
        return np.exp(df["base_log"].to_numpy() + self.model.predict(X))
