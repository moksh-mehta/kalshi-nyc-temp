# Kalshi NYC Temperature Contract Model

A probabilistic forecasting model for Kalshi's daily high-temperature contracts
for New York City (NWS Central Park station, `GHCND:USW00094728`), built to
price and backtest binary/bracket temperature contracts against real NWS
observations.

## What this does

1. **`src/fetch_data.py`** pulls daily max/min temperature history for
   Central Park from NOAA's public API (`ncei.noaa.gov`), caches it locally
   as CSV.
2. **`src/features.py`** builds a feature set per calendar day: day-of-year
   harmonic terms, trailing N-day anomaly, ENSO-adjustable trend term,
   lagged autocorrelation features.
3. **`src/model.py`** fits a day-of-year seasonal baseline (harmonic
   regression) plus a residual model (Gaussian or skew-normal residual
   distribution, fit per rolling window) to produce a full predictive
   distribution for a given target date, not just a point forecast. 
2. **`src/kalshi_pricing.py`** converts the predictive distribution into
   implied probabilities for arbitrary Kalshi strike brackets, and compares
   against a naive climatological-only baseline.
3. **`src/backtest.py`** walk-forward backtest: for each historical date,
   fit only on data available before that date, generate a forecast
   distribution, and score it (Brier score, log loss, calibration) against
   what actually happened.
4. **`notebooks/`** exploratory analysis and result plots.

## Status

The full pipeline has been run end to end against real NWS Central Park
data (2000 to 2025). Walk forward backtest results, scored against a naive
climatology only baseline using proper scoring rules:

| Metric | Model | Climatology baseline |
|---|---|---|
| Mean Brier score | 0.0883 | 0.0888 |
| Mean log loss | 0.3168 | 0.3191 |

Test window: 2015 01 01 to 2024 12 31 (3,653 test dates), refit every 30 days.

Next steps: test whether the edge is robust to different residual
windows and refit frequencies, extend to hourly temperature contracts,
and evaluate whether the small Brier and log loss improvement translates
to a viable edge after Kalshi's fees and spread on real contract prices.
