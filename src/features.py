"""
Feature engineering for the NYC daily-high-temperature model.

Design principle: Kalshi's contract resolves on a single day's recorded
high, so the model needs a full predictive distribution for that specific
calendar date, conditioned on:
  - seasonal position (day-of-year, via harmonics rather than raw month/day
    dummies, to avoid a discontinuity at year boundaries and to fit smoothly
    with limited data per calendar day)
  - recent anomaly (has it been a warm or cold stretch relative to normal?)
  - long-run trend (warming trend over the sample period)
"""
import numpy as np
import pandas as pd


def add_harmonic_terms(df: pd.DataFrame, date_col: str = "DATE", n_harmonics: int = 3) -> pd.DataFrame:
    """Add sin/cos day-of-year harmonics, the standard way to encode a
    smooth annual seasonal cycle without discontinuities at Dec 31 -> Jan 1."""
    doy = df[date_col].dt.dayofyear
    year_len = 365.25
    for k in range(1, n_harmonics + 1):
        df[f"sin_{k}"] = np.sin(2 * np.pi * k * doy / year_len)
        df[f"cos_{k}"] = np.cos(2 * np.pi * k * doy / year_len)
    return df


def add_trend_term(df: pd.DataFrame, date_col: str = "DATE") -> pd.DataFrame:
    """Linear trend in years since sample start, to capture long-run
    warming rather than forcing the seasonal fit to absorb it."""
    df["years_since_start"] = (
        df[date_col] - df[date_col].min()
    ).dt.days / 365.25
    return df


def add_trailing_anomaly(df: pd.DataFrame, temp_col: str = "TMAX", window: int = 7) -> pd.DataFrame:
    """Trailing N-day anomaly vs. a same-calendar-day climatological norm.

    Deliberately does NOT compare against a whole-year expanding/rolling
    mean of TMAX — that conflates seasonal position with anomaly (e.g. every
    December day would look like a huge negative anomaly relative to a
    mean dominated by summer months). Instead, for each day, subtract the
    historical mean TMAX for that same day-of-year (+/- a small window)
    computed only from prior years, then take a trailing rolling mean of
    that de-seasonalized residual.

    If 'seasonal_baseline' (the harmonic regression fit) is already present
    on df, that is used instead, since it's a smoother, better estimate."""
    if "seasonal_baseline" in df.columns:
        df["resid"] = df[temp_col] - df["seasonal_baseline"]
    else:
        df = _add_day_of_year_climatology_resid(df, temp_col=temp_col)

    df[f"trailing_anomaly_{window}d"] = (
        df["resid"].rolling(window, min_periods=max(2, window // 2)).mean().shift(1)
    )
    return df


def _add_day_of_year_climatology_resid(df: pd.DataFrame, temp_col: str = "TMAX",
                                        doy_window: int = 10) -> pd.DataFrame:
    """Fallback de-seasonalization when no fitted seasonal_baseline exists
    yet: for each row, compute the mean TMAX across all *prior* years within
    +/- doy_window calendar days of that row's day-of-year, and use that as
    the seasonal norm. This avoids lookahead (only uses years strictly
    before the current row's year) and avoids the whole-year-mean bug.

    Vectorized via a (n_rows x 366) day-of-year bucket table rather than a
    per-row Python loop, since the naive O(n^2) version is too slow past a
    few thousand rows (this dataset has ~9-25k rows depending on history
    length)."""
    df = df.copy()
    doy = df["DATE"].dt.dayofyear.to_numpy()
    year = df["DATE"].dt.year.to_numpy()
    vals = df[temp_col].to_numpy()

    years_sorted = np.unique(year)
    # sum/count of TMAX per (year, day-of-year), 1-indexed doy up to 366
    max_doy = 366
    year_index = {y: i for i, y in enumerate(years_sorted)}
    sums = np.zeros((len(years_sorted), max_doy + 1))
    counts = np.zeros((len(years_sorted), max_doy + 1))
    for y, d, v in zip(year, doy, vals):
        if pd.isna(v):
            continue
        yi = year_index[y]
        sums[yi, d] += v
        counts[yi, d] += 1

    # circular day-of-year window: smear each day's sum/count into neighboring
    # days +/- doy_window using convolution-style accumulation
    window_days = np.arange(-doy_window, doy_window + 1)
    smeared_sums = np.zeros_like(sums)
    smeared_counts = np.zeros_like(counts)
    for w in window_days:
        shifted_days = ((np.arange(1, max_doy + 1) - 1 + w) % max_doy) + 1
        smeared_sums[:, 1:] += sums[:, shifted_days]
        smeared_counts[:, 1:] += counts[:, shifted_days]

    # prior-years-only cumulative sum/count (exclusive of current year)
    cum_sums = np.cumsum(smeared_sums, axis=0) - smeared_sums
    cum_counts = np.cumsum(smeared_counts, axis=0) - smeared_counts

    resid = np.full(len(df), np.nan)
    for i in range(len(df)):
        yi = year_index[year[i]]
        c = cum_counts[yi, doy[i]]
        if c > 0:
            norm = cum_sums[yi, doy[i]] / c
            resid[i] = vals[i] - norm
    df["resid"] = resid
    return df


def add_lag_features(df: pd.DataFrame, temp_col: str = "TMAX", lags=(1, 2, 3)) -> pd.DataFrame:
    for lag in lags:
        df[f"{temp_col}_lag{lag}"] = df[temp_col].shift(lag)
    return df


def build_feature_frame(df: pd.DataFrame, temp_col: str = "TMAX") -> pd.DataFrame:
    """Full feature pipeline. Expects df sorted by DATE ascending, with no
    duplicate dates (dedupe upstream if a station has overlapping records)."""
    df = df.sort_values("DATE").reset_index(drop=True).copy()
    df = add_harmonic_terms(df)
    df = add_trend_term(df)
    df = add_lag_features(df, temp_col=temp_col)
    df = add_trailing_anomaly(df, temp_col=temp_col)
    return df
