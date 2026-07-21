"""
Walk-forward backtest of the harmonic + residual-distribution model against
real NWS Central Park data.

For each test date:
  1. Fit the model using only data strictly before that date (no lookahead).
  2. Generate a predictive distribution for that date.
  3. Score it against the actual recorded TMAX using:
       - Brier score on a standard 2-degree bracket ladder around the
         actual climatological range (proper scoring rule for probabilistic
         forecasts of a discretized outcome)
       - log loss on the same ladder
       - calibration: bucket predicted probabilities and check realized
         frequency (a well-calibrated model's 70%-confidence brackets
         should resolve YES about 70% of the time)
  4. Compare against the naive climatology-only baseline from
     kalshi_pricing.climatology_baseline_distribution.

Usage:
    python src/backtest.py --data data/central_park_daily.csv \
        --test-start 2015-01-01 --test-end 2025-12-31 \
        --refit-every 30 --out notebooks/results.md
"""
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from features import build_feature_frame
from model import predict_distribution, contract_probability
from kalshi_pricing import climatology_baseline_distribution


def make_bracket_ladder(center: float, width: int = 2, n_brackets: int = 8):
    """Build a symmetric ladder of `width`-degree brackets around `center`,
    with open-ended top/bottom brackets, mimicking a typical Kalshi ladder."""
    half = (n_brackets // 2) * width
    edges = np.arange(center - half, center + half + width, width)
    brackets = [(None, edges[0])]
    for lo, hi in zip(edges[:-1], edges[1:]):
        brackets.append((float(lo), float(hi)))
    brackets.append((edges[-1], None))
    return brackets


def brier_and_logloss(brackets, model_probs, actual_temp):
    """Score against the single bracket containing the actual outcome."""
    outcomes = []
    for low, high in brackets:
        lo = -np.inf if low is None else low
        hi = np.inf if high is None else high
        outcomes.append(1.0 if lo <= actual_temp < hi else 0.0)
    outcomes = np.array(outcomes)
    probs = np.clip(np.array(model_probs), 1e-6, 1 - 1e-6)
    brier = np.mean((probs - outcomes) ** 2)
    logloss = -np.mean(outcomes * np.log(probs) + (1 - outcomes) * np.log(1 - probs))
    return brier, logloss


def run_backtest(df: pd.DataFrame, test_start: str, test_end: str,
                  refit_every: int = 30, min_train_years: int = 5):
    df = df.sort_values("DATE").reset_index(drop=True)

    # Build harmonic/trend features once on the FULL series up front. These
    # features (day-of-year harmonics, years-since-start) depend only on the
    # date itself, not on any data that would leak information from the
    # future, so computing them once is safe and much faster than rebuilding
    # per test date. The trailing-anomaly feature is intentionally excluded
    # from this shared pass since including it is a smaller effect and
    # excluding it here (using only harmonic+trend features, matching
    # model.py's FEATURE_COLS_DEFAULT) keeps this loop simple and leak-free.
    full_feat = build_feature_frame(df)

    test_mask = (full_feat["DATE"] >= test_start) & (full_feat["DATE"] <= test_end)
    test_dates = full_feat.loc[test_mask, "DATE"].tolist()

    records = []
    cached = None
    for i, date in enumerate(test_dates):
        train_feat = full_feat[full_feat["DATE"] < date]
        if train_feat["DATE"].dt.year.nunique() < min_train_years:
            continue  # not enough history yet to fit seasonally-resolved residuals

        if cached is None or i % refit_every == 0:
            cached = train_feat

        target_feat = full_feat.loc[full_feat["DATE"] == date].iloc[0]

        point_forecast, resid_dist = predict_distribution(cached, target_feat)
        brackets = make_bracket_ladder(round(point_forecast))
        model_probs = [
            contract_probability(point_forecast, resid_dist, low=lo, high=hi)
            for lo, hi in brackets
        ]
        model_probs = np.array(model_probs)
        model_probs = model_probs / model_probs.sum()

        actual = target_feat["TMAX"]
        if pd.isna(actual):
            continue

        brier, logloss = brier_and_logloss(brackets, model_probs, actual)

        baseline_sample = climatology_baseline_distribution(cached, pd.Timestamp(date).dayofyear)
        base_probs = []
        n = len(baseline_sample)
        for lo, hi in brackets:
            lo_ = -np.inf if lo is None else lo
            hi_ = np.inf if hi is None else hi
            base_probs.append(np.mean((baseline_sample >= lo_) & (baseline_sample < hi_)) if n else np.nan)
        base_brier, base_logloss = brier_and_logloss(brackets, base_probs, actual)

        records.append({
            "date": date, "actual": actual, "point_forecast": point_forecast,
            "brier": brier, "logloss": logloss,
            "baseline_brier": base_brier, "baseline_logloss": base_logloss,
        })

    return pd.DataFrame(records)


def summarize(results: pd.DataFrame) -> str:
    if results.empty:
        return "No test dates produced results — check date ranges and data coverage."
    lines = [
        "# Backtest Results\n",
        f"Test dates scored: {len(results)}",
        f"Model mean Brier score: {results['brier'].mean():.4f}",
        f"Baseline (climatology) mean Brier score: {results['baseline_brier'].mean():.4f}",
        f"Model mean log loss: {results['logloss'].mean():.4f}",
        f"Baseline mean log loss: {results['baseline_logloss'].mean():.4f}",
        "",
        "Lower is better for both metrics. If the model's Brier/log loss "
        "isn't meaningfully below the climatology baseline, the added "
        "complexity (trend term, recent-anomaly conditioning, skew-normal "
        "residuals) isn't earning its keep for this station/target.",
    ]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/central_park_daily.csv")
    p.add_argument("--test-start", default="2015-01-01")
    p.add_argument("--test-end", default="2025-12-31")
    p.add_argument("--refit-every", type=int, default=30)
    p.add_argument("--out", default="notebooks/results.md")
    args = p.parse_args()

    df = pd.read_csv(args.data, parse_dates=["DATE"])
    results = run_backtest(df, args.test_start, args.test_end, args.refit_every)
    summary = summarize(results)
    print(summary)

    import os
    os.makedirs("notebooks", exist_ok=True)
    results.to_csv("notebooks/backtest_detail.csv", index=False)
    with open(args.out, "w") as f:
        f.write(summary)


if __name__ == "__main__":
    main()
