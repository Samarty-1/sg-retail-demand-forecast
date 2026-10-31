from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecast import backtest, calendar, intervals
from forecast.models import MODELS, GlobalGBM, ets, naive_plus_growth, seasonal_naive


# ------------------------------------------------------------------ calendar

@pytest.mark.parametrize("month,run_up,holiday", [
    ("2026-02-01", 1.0, 1.0),      # CNY 17 Feb 2026: run-up 3-16 Feb
    ("2026-01-01", 0.0, 0.0),
    ("2025-01-01", 1.0, 1.0),      # CNY 29 Jan 2025: run-up 15-28 Jan, holiday 29-30 Jan
    ("2024-01-01", 5 / 14, 0.0),   # CNY 10 Feb 2024: run-up 27 Jan - 9 Feb
    ("2024-02-01", 9 / 14, 1.0),
    ("2023-03-01", 0.0, 0.0),
])
def test_cny_shares(month, run_up, holiday):
    m = pd.Timestamp(month)
    assert calendar.cny_run_up_share(m) == pytest.approx(run_up)
    assert calendar.cny_holiday_share(m) == pytest.approx(holiday)


def test_cny_shares_sum_to_one_across_a_year():
    for year in range(1991, 2028):
        months = pd.date_range(f"{year}-01-01", periods=12, freq="MS")
        assert sum(calendar.cny_run_up_share(m) for m in months) == pytest.approx(1.0)
        assert sum(calendar.cny_holiday_share(m) for m in months) == pytest.approx(1.0)


def test_every_cny_date_is_in_late_january_or_february():
    for d in calendar.CNY.values():
        assert (d.month == 1 and d.day >= 21) or (d.month == 2 and d.day <= 20)


# ------------------------------------------------------------------ simple models

def seasonal_series(n=180, growth=0.003, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 100 * np.exp(growth * t) * (1 + 0.1 * np.sin(2 * np.pi * t / 12)) * np.exp(rng.normal(0, 0.005, n))


def test_seasonal_naive_repeats_last_year():
    y = np.arange(1.0, 25.0)
    assert list(seasonal_naive(y)) == list(y[-12:])
    assert list(seasonal_naive(y, 3)) == list(y[-12:-9])


def test_naive_plus_growth_scales_by_annual_growth():
    y = np.concatenate([np.full(12, 100.0), np.full(12, 110.0)])
    assert naive_plus_growth(y) == pytest.approx(np.full(12, 121.0))


def test_short_history_is_rejected():
    with pytest.raises(ValueError):
        seasonal_naive(np.ones(11))
    with pytest.raises(ValueError):
        naive_plus_growth(np.ones(23))


def test_ets_tracks_a_clean_seasonal_trend():
    y = seasonal_series()
    truth = seasonal_series(n=192)[-12:]
    assert np.mean(np.abs(ets(y) / truth - 1)) < 0.02


# ------------------------------------------------------------------ no look-ahead

def panel_of(n_series=3, n=200, start="2000-01-01"):
    idx = pd.date_range(start, periods=n, freq="MS")
    return pd.DataFrame({f"Retail: S{i}": seasonal_series(n, seed=i) for i in range(n_series)}, index=idx)


def test_gbm_forecast_ignores_data_after_the_origin():
    panel = panel_of()
    t = 150
    a = GlobalGBM(panel).fit(cutoff=t).predict(1, t)
    corrupted = panel.copy()
    corrupted.iloc[t + 1:] = corrupted.iloc[t + 1:] * 50  # wreck the future
    b = GlobalGBM(corrupted).fit(cutoff=t).predict(1, t)
    np.testing.assert_allclose(a, b)


def test_backtest_forecasts_ignore_data_after_the_origin():
    panel = panel_of(n=170)
    origin = panel.index[140]
    a = backtest.run(panel, first_origin=origin, last_origin=origin)
    corrupted = panel.copy()
    corrupted.iloc[141:] = corrupted.iloc[141:] * 50
    b = backtest.run(corrupted, first_origin=origin, last_origin=origin)
    pd.testing.assert_frame_equal(a[MODELS], b[MODELS])
    assert (a["target"] > a["origin"]).all() and set(a["h"]) == set(range(1, 13))


def test_series_too_young_are_not_forecast():
    panel = panel_of(n=170)
    panel.iloc[:120, 0] = np.nan  # series 0 starts 50 months before the origin
    bt = backtest.run(panel, first_origin=panel.index[169 - 12], last_origin=panel.index[169 - 12])
    assert "Retail: S0" not in set(bt["series"])


# ------------------------------------------------------------------ scoring

def fake_backtest(errors: dict[str, float], n_series=4, covid_error=0.5):
    rows = []
    for s in range(n_series):
        for month in pd.date_range("2018-01-01", "2021-06-01", freq="MS"):
            covid = backtest.COVID[0] <= month <= backtest.COVID[1]
            row = {"series": f"S{s}", "origin": month - pd.DateOffset(months=1), "h": 1, "target": month,
                   "actual": 100.0, "covid": covid}
            for m, e in errors.items():
                row[m] = 100.0 * (1 + (covid_error if covid else e))
            rows.append(row)
    return pd.DataFrame(rows)


def test_summary_applies_the_protocol_rule():
    bt = fake_backtest({"seasonal_naive": 0.05, "naive_plus_growth": 0.06, "ets": 0.04, "lightgbm": 0.03})
    head = backtest.summarise(bt)["headline"]
    assert head.loc["lightgbm", "error_excl_covid"] == pytest.approx(0.03)
    assert head.loc["seasonal_naive", "error_covid_targets"] == pytest.approx(0.5)  # kept out of the headline
    assert bool(head.loc["lightgbm", "beats_baseline"]) and bool(head.loc["ets", "beats_baseline"])
    assert not bool(head.loc["naive_plus_growth", "beats_baseline"])
    assert head.loc["lightgbm", "median_skill_vs_naive"] == pytest.approx(0.4)


def test_ranges_cover_what_they_claim():
    rng = np.random.default_rng(1)
    n = 6000
    target = pd.Timestamp("2010-01-01") + pd.to_timedelta(rng.integers(0, 365 * 15, n), unit="D")
    actual = np.full(n, 100.0)
    bt = pd.DataFrame({"h": rng.integers(1, 13, n), "target": target.normalize(), "actual": actual,
                       "seasonal_naive": actual * np.exp(rng.normal(0, 0.05, n)), "covid": False})
    c = intervals.calibration(bt, "seasonal_naive")
    assert 0.75 < c["coverage"] < 0.85
