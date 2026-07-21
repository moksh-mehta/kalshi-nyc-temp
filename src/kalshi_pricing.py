"""
Translate the model's predictive distribution into Kalshi-style bracket
prices, and compare against a naive climatological-only baseline (i.e.
"what would you price this contract at using only the historical average
for this calendar day, ignoring recent conditions and trend").

Kalshi NYC daily-high-temperature markets typically list a ladder of
2-degree-wide brackets (e.g. "72-73", "74-75", ...) plus open-ended
top/bottom contracts. This module prices an arbitrary ladder.
"""
import numpy as np
import pandas as pd

from model import contract_probability


def price_ladder(point_forecast: float, resid_dist, brackets: list) -> pd.DataFrame:
    """
    brackets: list of (low, high) tuples, in Fahrenheit, matching Kalshi's
    listed strikes for the target date. Use None for open-ended ends,
    e.g. (None, 60) for "59 or below", (90, None) for "90 or above".

    Returns a DataFrame with model-implied probability per bracket,
    normalized to sum to 1 (small numerical drift can occur at the tails
    since brackets should partition the real line exactly).
    """
    rows = []
    for low, high in brackets:
        prob = contract_probability(point_forecast, resid_dist, low=low, high=high)
        rows.append({"low": low, "high": high, "model_prob": prob})
    out = pd.DataFrame(rows)
    out["model_prob"] = out["model_prob"] / out["model_prob"].sum()
    out["model_price_cents"] = (out["model_prob"] * 100).round(1)
    return out


def climatology_baseline_distribution(train_df: pd.DataFrame, target_doy: int,
                                       temp_col: str = "TMAX", window_days: int = 10):
    """
    Naive baseline: empirical distribution of TMAX on calendar days within
    +/- window_days of the target day-of-year, across all history, with no
    trend adjustment and no conditioning on recent anomaly. This is the
    bar the harmonic+residual model needs to beat in the backtest — if it
    doesn't outperform this, the extra machinery isn't earning its keep.
    """
    doy_all = train_df["DATE"].dt.dayofyear
    diff = np.minimum(np.abs(doy_all - target_doy), 365 - np.abs(doy_all - target_doy))
    sample = train_df.loc[diff <= window_days, temp_col].dropna().to_numpy()
    return sample  # empirical sample; use np.mean/np.std or ECDF directly


def edge_vs_baseline(model_ladder: pd.DataFrame, baseline_sample: np.ndarray,
                      brackets: list) -> pd.DataFrame:
    """Compare model-implied bracket probabilities to the empirical
    climatology baseline, to see where the model disagrees most —
    those are the brackets where, if the model is right, there's edge."""
    baseline_probs = []
    n = len(baseline_sample)
    for low, high in brackets:
        lo = -np.inf if low is None else low
        hi = np.inf if high is None else high
        baseline_probs.append(np.mean((baseline_sample >= lo) & (baseline_sample < hi)) if n else np.nan)
    out = model_ladder.copy()
    out["baseline_prob"] = baseline_probs
    out["edge"] = out["model_prob"] - out["baseline_prob"]
    return out
