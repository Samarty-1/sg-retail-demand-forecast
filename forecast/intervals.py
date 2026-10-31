"""80% ranges from a model's own backtest errors, and the calibration check."""
from __future__ import annotations

import numpy as np
import pandas as pd

LEVEL = 0.8


def log_errors(bt: pd.DataFrame, model: str) -> pd.Series:
    return np.log(bt["actual"] / bt[model])


def quantiles_by_horizon(bt: pd.DataFrame, model: str, level: float = LEVEL) -> pd.DataFrame:
    """Lower and upper multipliers per horizon: actual / forecast fell between them in
    `level` of past (non-COVID) forecasts."""
    ok = bt[~bt["covid"]]
    e = log_errors(ok, model)
    q = e.groupby(ok["h"]).quantile([(1 - level) / 2, (1 + level) / 2]).unstack()
    q.columns = ["low", "high"]
    return np.exp(q)


def calibration(bt: pd.DataFrame, model: str, fit_until: str = "2019-12-01", test_from: str = "2022-01-01",
                level: float = LEVEL) -> dict:
    """Build ranges from errors on targets up to `fit_until`, then count how often later
    actuals land inside them."""
    past = bt[(bt["target"] <= fit_until) & ~bt["covid"]]
    later = bt[bt["target"] >= test_from].copy()
    q = quantiles_by_horizon(past, model, level)
    later = later.join(q, on="h")
    inside = (later["actual"] >= later[model] * later["low"]) & (later["actual"] <= later[model] * later["high"])
    by_h = inside.groupby(later["h"]).mean()
    return {"nominal": level, "coverage": float(inside.mean()), "by_horizon": by_h, "n": int(len(later))}
