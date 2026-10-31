"""Rolling-origin backtest exactly as specified in docs/PROTOCOL.md."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .models import H, MODELS, GlobalGBM, ets, naive_plus_growth, seasonal_naive

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
FIRST_ORIGIN = pd.Timestamp("2012-12-01")
MIN_HISTORY = 60  # months of data a series needs before it is forecast
PANEL_START = pd.Timestamp("1991-01-01")  # the Chinese New Year table starts in 1990
COVID = (pd.Timestamp("2020-02-01"), pd.Timestamp("2021-12-01"))


def load_panel(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or ROOT / "data" / "series.csv", parse_dates=["month"])
    panel = df.pivot(index="month", columns="series", values="index").sort_index()
    panel = panel[panel.index >= PANEL_START]
    full = pd.date_range(panel.index.min(), panel.index.max(), freq="MS")
    if len(full) != len(panel):
        raise ValueError("panel has missing months")
    return panel


def run(panel: pd.DataFrame, first_origin: pd.Timestamp = FIRST_ORIGIN,
        last_origin: pd.Timestamp | None = None) -> pd.DataFrame:
    months = panel.index
    values = panel.to_numpy(dtype=float)
    last_origin = last_origin or months[-2]
    origins = [i for i, m in enumerate(months) if first_origin <= m <= last_origin]
    gbm = GlobalGBM(panel)
    fitted_at = None
    rows = []
    for t in origins:
        # Refit once a year (each December) and at the first origin, on data up to t.
        if fitted_at is None or months[t].month == 12:
            gbm.fit(cutoff=t)
            fitted_at = t
            log.info("LightGBM refitted at %s", months[t].date())
        horizon = min(H, len(months) - 1 - t)
        for s, name in enumerate(panel.columns):
            hist = values[: t + 1, s]
            hist = hist[~np.isnan(hist)]
            if len(hist) < MIN_HISTORY or np.isnan(values[t, s]):
                continue
            preds = {
                "seasonal_naive": seasonal_naive(hist, horizon),
                "naive_plus_growth": naive_plus_growth(hist, horizon),
                "ets": ets(hist, horizon),
                "lightgbm": gbm.predict(s, t, horizon),
            }
            for h in range(1, horizon + 1):
                actual = values[t + h, s]
                if np.isnan(actual):
                    continue
                rows.append({"series": name, "origin": months[t], "h": h, "target": months[t + h],
                             "actual": actual, **{m: preds[m][h - 1] for m in MODELS}})
    out = pd.DataFrame(rows)
    out["covid"] = out["target"].between(*COVID)
    return out


def ape(bt: pd.DataFrame) -> pd.DataFrame:
    """Absolute percentage error of every forecast, per model."""
    return pd.DataFrame({m: (bt[m] - bt["actual"]).abs() / bt["actual"] for m in MODELS}, index=bt.index)


def summarise(bt: pd.DataFrame) -> dict[str, pd.DataFrame]:
    errors = ape(bt)
    normal = ~bt["covid"]

    headline = pd.DataFrame({
        "error_excl_covid": errors[normal].mean(),
        "error_covid_targets": errors[~normal].mean(),
        "error_all": errors.mean(),
    })
    per_series = errors[normal].groupby(bt.loc[normal, "series"]).mean()
    skill = 1 - per_series.div(per_series["seasonal_naive"], axis=0)
    headline["median_skill_vs_naive"] = skill.median()
    headline["series_beating_naive"] = (per_series.lt(per_series["seasonal_naive"], axis=0)).sum()
    headline.loc["seasonal_naive", "series_beating_naive"] = np.nan
    headline["beats_baseline"] = (headline["error_excl_covid"] < headline.loc["seasonal_naive", "error_excl_covid"]) \
        & (headline["series_beating_naive"] > len(per_series) / 2)
    headline.loc["seasonal_naive", "beats_baseline"] = False

    by_h = errors[normal].groupby(bt.loc[normal, "h"]).mean()
    # Error in the months Chinese New Year can land in, against all other months.
    jan_feb = bt["target"].dt.month.isin([1, 2])
    by_season = pd.DataFrame({
        "jan_feb": errors[normal & jan_feb].mean(),
        "other_months": errors[normal & ~jan_feb].mean(),
    })
    return {"headline": headline, "by_horizon": by_h, "per_series": per_series, "by_season": by_season}
