"""
Predictive distribution model for NYC daily high temperature.

Two-stage approach:
  1. Harmonic regression gives E[TMAX | day-of-year, trend, recent anomaly] —
     a smooth seasonal-plus-trend point forecast.
  2. Residuals (actual - predicted) are modeled with a skew-normal
     distribution fit on a rolling window of the same calendar-day
     neighborhood, since temperature residuals are not symmetric
     (cold snaps are typically sharper than warm excursions in NYC winter,
     and the reverse can hold in summer) — a plain Gaussian residual
     understates tail probability on one side.

The output of `predict_distribution` is a scipy.stats frozen distribution,
which is what you actually need to price a Kalshi bracket: P(low < TMAX < high)
for arbitrary strikes, not just a point estimate.
"""
import numpy as np
import pandas as pd
from scipy import stats
from numpy.linalg import lstsq


FEATURE_COLS_DEFAULT = ["sin_1", "cos_1", "sin_2", "cos_2", "sin_3", "cos_3", "years_since_start"]


def fit_harmonic_regression(train_df: pd.DataFrame, temp_col: str = "TMAX",
                             feature_cols=None) -> np.ndarray:
    """OLS fit of TMAX on harmonic + trend features. Returns coefficient
    vector (no intercept term needed — cos_0 term already centers it, but
    we add an explicit intercept column for clarity)."""
    feature_cols = feature_cols or FEATURE_COLS_DEFAULT
    X = train_df[feature_cols].to_numpy()
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    y = train_df[temp_col].to_numpy()
    valid = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
    coef, *_ = lstsq(X[valid], y[valid], rcond=None)
    return coef


def predict_point(df: pd.DataFrame, coef: np.ndarray, feature_cols=None) -> np.ndarray:
    feature_cols = feature_cols or FEATURE_COLS_DEFAULT
    X = df[feature_cols].to_numpy()
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    return X @ coef


def fit_residual_distribution(residuals: np.ndarray):
    """
    Fit a skew-normal to residuals. Falls back to a plain Gaussian if the
    skew-normal fit is degenerate (e.g. too few residuals in a window) or
    if the fitted skew parameter is implausibly extreme, which usually
    signals a small-sample artifact rather than genuine skew.
    """
    residuals = residuals[~np.isnan(residuals)]
    if len(residuals) < 20:
        raise ValueError("Need at least 20 residuals to fit a distribution reliably")
    try:
        a, loc, scale = stats.skewnorm.fit(residuals)
        if abs(a) > 20 or scale <= 0:
            raise ValueError("degenerate skewnorm fit")
        return stats.skewnorm(a, loc=loc, scale=scale)
    except Exception:
        mu, sigma = np.nanmean(residuals), np.nanstd(residuals)
        return stats.norm(loc=mu, scale=sigma)


def predict_distribution(train_df: pd.DataFrame, target_row: pd.Series,
                          temp_col: str = "TMAX", residual_window_days: int = 21,
                          feature_cols=None):
    """
    Fit the harmonic regression on train_df, then build a residual
    distribution using only residuals from calendar days within
    +/- residual_window_days of the target's day-of-year (across all
    years in train_df) — this captures season-specific residual behavior
    (e.g. wider variance in winter cold snaps) rather than pooling
    residuals across the whole year.

    Returns: (point_forecast, frozen_scipy_distribution_of_residuals)
    Combine as: actual_temp ~ point_forecast + residual_distribution
    """
    feature_cols = feature_cols or FEATURE_COLS_DEFAULT
    coef = fit_harmonic_regression(train_df, temp_col=temp_col, feature_cols=feature_cols)
    train_df = train_df.copy()
    train_df["seasonal_baseline"] = predict_point(train_df, coef, feature_cols)
    train_df["resid"] = train_df[temp_col] - train_df["seasonal_baseline"]

    target_doy = pd.Timestamp(target_row["DATE"]).dayofyear
    doy_all = train_df["DATE"].dt.dayofyear
    # circular distance in day-of-year space, so e.g. Dec 28 is "close to" Jan 3
    diff = np.minimum(
        np.abs(doy_all - target_doy),
        365 - np.abs(doy_all - target_doy),
    )
    window_mask = diff <= residual_window_days
    residuals = train_df.loc[window_mask, "resid"].to_numpy()

    dist = fit_residual_distribution(residuals)
    point_forecast = float(predict_point(target_row.to_frame().T, coef, feature_cols)[0])
    return point_forecast, dist


def contract_probability(point_forecast: float, resid_dist, low: float = None, high: float = None) -> float:
    """
    P(low < TMAX < high) for a Kalshi bracket contract, given the fitted
    point forecast and residual distribution. Pass only `low` or only
    `high` for a one-sided ("above X" / "below X") contract.
    """
    if low is None and high is None:
        raise ValueError("Provide at least one of low/high")
    if low is not None and high is not None:
        return float(resid_dist.cdf(high - point_forecast) - resid_dist.cdf(low - point_forecast))
    if high is not None:
        return float(resid_dist.cdf(high - point_forecast))
    return float(1 - resid_dist.cdf(low - point_forecast))
