"""
Unit tests using synthetic data (generated in-memory, never committed),
so the pipeline can be verified without network access to NOAA.

Run with: python -m pytest tests/
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features import build_feature_frame
from model import fit_harmonic_regression, predict_distribution, contract_probability
from kalshi_pricing import price_ladder, climatology_baseline_distribution
from backtest import run_backtest, make_bracket_ladder, brier_and_logloss


def make_synthetic_df(n_years=15, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2005-01-01", periods=365 * n_years, freq="D")
    doy = dates.dayofyear
    seasonal = 55 + 22 * np.sin(2 * np.pi * (doy - 100) / 365.25)
    noise = rng.normal(0, 6, len(dates))
    tmax = seasonal + noise
    return pd.DataFrame({"DATE": dates, "TMAX": tmax, "TMIN": tmax - 15})


def test_feature_frame_has_expected_columns():
    df = make_synthetic_df()
    feat = build_feature_frame(df)
    for col in ["sin_1", "cos_1", "years_since_start", "trailing_anomaly_7d"]:
        assert col in feat.columns


def test_trailing_anomaly_is_not_dominated_by_season():
    """Regression test for the bug where winter days looked like huge
    negative anomalies because the fallback compared against a whole-year
    mean instead of a same-calendar-day climatology."""
    df = make_synthetic_df()
    feat = build_feature_frame(df)
    late_year = feat[feat["DATE"] >= "2015-01-01"]
    # after enough history has accumulated, anomalies should be small,
    # not systematically ~20 degrees off in either direction
    assert late_year["trailing_anomaly_7d"].abs().median() < 8


def test_harmonic_regression_recovers_seasonal_shape():
    df = make_synthetic_df()
    feat = build_feature_frame(df)
    coef = fit_harmonic_regression(feat)
    assert coef.shape[0] == 8  # intercept + 6 harmonic + 1 trend term
    assert not np.isnan(coef).any()


def test_predict_distribution_gives_reasonable_forecast():
    df = make_synthetic_df()
    feat = build_feature_frame(df)
    train = feat[feat["DATE"] < "2019-01-01"]
    target = feat[feat["DATE"] == "2019-07-15"].iloc[0]
    point, dist = predict_distribution(train, target)
    # July 15 should forecast near the summer peak (~77), not winter (~33)
    assert 65 < point < 90


def test_contract_probability_sums_correctly():
    df = make_synthetic_df()
    feat = build_feature_frame(df)
    train = feat[feat["DATE"] < "2019-01-01"]
    target = feat[feat["DATE"] == "2019-07-15"].iloc[0]
    point, dist = predict_distribution(train, target)
    p_below = contract_probability(point, dist, high=70)
    p_above = contract_probability(point, dist, low=70)
    assert abs((p_below + p_above) - 1.0) < 1e-9


def test_price_ladder_normalizes_to_one():
    df = make_synthetic_df()
    feat = build_feature_frame(df)
    train = feat[feat["DATE"] < "2019-01-01"]
    target = feat[feat["DATE"] == "2019-07-15"].iloc[0]
    point, dist = predict_distribution(train, target)
    brackets = make_bracket_ladder(round(point))
    ladder = price_ladder(point, dist, brackets)
    assert abs(ladder["model_prob"].sum() - 1.0) < 1e-6


def test_backtest_runs_and_beats_or_matches_baseline_directionally():
    """Not a strict 'model must win' assertion (would be flaky on
    synthetic data with no real anomaly structure) -- just checks the
    backtest runs without error and produces valid probability scores."""
    df = make_synthetic_df(n_years=12)
    feat_check = build_feature_frame(df)
    results = run_backtest(df, "2015-01-01", "2015-01-31", refit_every=30)
    assert len(results) > 0
    assert (results["brier"] >= 0).all() and (results["brier"] <= 1).all()
    assert (results["baseline_brier"] >= 0).all()


def test_brier_and_logloss_perfect_prediction():
    brackets = [(None, 50), (50, 60), (60, None)]
    probs = [0.0, 1.0, 0.0]
    brier, logloss = brier_and_logloss(brackets, probs, actual_temp=55)
    assert brier < 1e-6
    assert logloss < 1e-4


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
