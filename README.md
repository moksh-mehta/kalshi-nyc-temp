# Kalshi NYC Temperature Contract Model

A probabilistic forecasting model for Kalshi's daily high-temperature contracts
for New York City (NWS Central Park station, `GHCND:USW00094728`), built to
price and backtest binary/bracket temperature contracts against real NWS
observations.

## Why Central Park

Kalshi's NYC temperature markets (`KXHIGHNY` and similar tickers) settle
against the NWS Central Park station reading. That station has one of the
longest continuous, high-quality daily records in the NOAA GHCN-Daily
network, which makes it a good fit for both climatological baselines and
short-window model fitting.

## What this does

1. **`src/fetch_data.py`** — pulls daily max/min temperature history for
   Central Park from NOAA's public API (`ncei.noaa.gov`), caches it locally
   as CSV.
2. **`src/features.py`** — builds a feature set per calendar day: day-of-year
   harmonic terms, trailing N-day anomaly, ENSO-adjustable trend term,
   lagged autocorrelation features.
3. **`src/model.py`** — fits a day-of-year seasonal baseline (harmonic
   regression) plus a residual model (Gaussian or skew-normal residual
   distribution, fit per rolling window) to produce a full predictive
   distribution for a given target date, not just a point forecast. This
   distribution is what actually prices a Kalshi bracket contract — you need
   P(T > threshold), not just an expected value.
2. **`src/kalshi_pricing.py`** — converts the predictive distribution into
   implied probabilities for arbitrary Kalshi strike brackets, and compares
   against a naive climatological-only baseline.
3. **`src/backtest.py`** — walk-forward backtest: for each historical date,
   fit only on data available before that date, generate a forecast
   distribution, and score it (Brier score, log loss, calibration) against
   what actually happened.
4. **`notebooks/`** — exploratory analysis and result plots.

## Status

**The code in this repo is complete and runs against live NOAA data, but has
not yet been executed end-to-end in this environment** — the sandbox this
was built in only has network access to package registries (PyPI/npm/GitHub),
not to `ncei.noaa.gov`. Run `src/fetch_data.py` from a machine with normal
internet access to pull the data, then run `src/backtest.py`; results
(Brier score, calibration plot, PnL under a simple strategy) will populate
`notebooks/results.md`.

## Honesty note

This is a from-scratch project built to genuinely learn the space, not a
polished production system. The model is intentionally simple (harmonic
seasonal regression + residual distribution) — a defensible, explainable
starting point rather than an overfit black box on a short data history.

## Tests

`tests/test_pipeline.py` verifies the pipeline end-to-end on synthetic
in-memory data (never written to disk or committed) so correctness can be
checked without network access to NOAA. Run with:

```bash
pip install pytest
python -m pytest tests/ -v
```

Note: `backtest.py`'s `--refit-every` controls how often the model
re-fits during the walk-forward loop. Refitting every single day is most
correct but slow; refitting every N days reuses the same fit for N
consecutive test dates as a speed/accuracy tradeoff. Set it to 1 for a
maximally rigorous (but slower) backtest.

## Setup

```bash
pip install -r requirements.txt
python src/fetch_data.py --station GHCND:USW00094728 --start 2000-01-01
python src/backtest.py --target-month 7  # backtest July contracts, e.g.
```
