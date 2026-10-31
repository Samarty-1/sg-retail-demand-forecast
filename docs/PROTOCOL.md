# Evaluation protocol

Written before any model was run on the data. Results that come later are judged
against this, not the other way round.

## Question

For a Singapore shop or F&B owner planning stock, staff or cash for the next 1 to 12
months, does any model forecast their sector's sales better than **"same month last
year"**? That is the rule of thumb a small business already uses, so it is the
baseline everything must beat.

## Data

SingStat Retail Sales Index (M602121) and Food & Beverage Services Index (M602131),
monthly, current prices: 22 series (16 retail, 6 F&B), up to August 2026. The
snapshot used is committed as `data/series.csv`.

## Models (four, fixed in advance)

1. **Seasonal naive**: same month last year.
2. **Seasonal naive + growth**: same month last year, times the growth of the last
   12 months over the 12 before.
3. **ETS**: Holt-Winters exponential smoothing on log sales (damped trend, additive
   seasonality on the log scale), fitted per series on the last 10 years before each
   forecast date.
4. **LightGBM, one model across all series**: predicts the year-on-year change of
   the target month from recent growth, momentum, horizon, calendar month, the
   series, and **where Chinese New Year falls** (it moves between January and
   February, which the seasonal naive cannot know). Refitted once a year on data
   available at the time.

No other models will be added after seeing results. Hyperparameters are set once
(in `forecast/models.py`) and not tuned on the test period.

## Backtest

- Rolling origin: a forecast is made at the end of every month from December 2012
  to July 2026, for horizons 1 to 12, using only data up to that month.
- A series is forecast only once it has 5 years of history.
- **Primary metric**: mean absolute percentage error (sales indices are always
  positive and far from zero, so percentage errors are well behaved), averaged
  over series, horizons and forecast dates.
- **Skill vs seasonal naive** = 1 - error(model) / error(seasonal naive), per series,
  then the median across series.
- **COVID** (targets from February 2020 to December 2021) is reported separately
  and excluded from the headline, because no model could have forecast circuit
  breaker closures, and averaging them in would rank models by luck in 2020.

A model "beats the baseline" only if its headline error is lower **and** it beats
seasonal naive on more than half of the 22 series.

## Uncertainty ranges

80% ranges come from the model's own backtest errors at each horizon.
Calibration check: ranges built only from errors on targets up to December 2019
must contain 70% to 90% of actuals on targets from January 2022 onward. If they
don't, the README says so.
