"""
Seasonal breakdown of backtest performance.

The top-line Brier/log-loss averages hide whether the model's edge over
climatology is spread evenly across the year or concentrated in specific
months (e.g. shoulder seasons, where "recent conditions" carry more
information than in a stable mid-summer stretch).

Run after backtest.py has produced notebooks/backtest_detail.csv:

    python src/seasonal_breakdown.py
"""
import pandas as pd

df = pd.read_csv("notebooks/backtest_detail.csv", parse_dates=["date"])
df["month"] = df["date"].dt.month

by_month = df.groupby("month").agg(
    n=("brier", "size"),
    model_brier=("brier", "mean"),
    baseline_brier=("baseline_brier", "mean"),
    model_logloss=("logloss", "mean"),
    baseline_logloss=("baseline_logloss", "mean"),
)
by_month["brier_edge_pct"] = (
    (by_month["baseline_brier"] - by_month["model_brier"]) / by_month["baseline_brier"] * 100
)
by_month["logloss_edge_pct"] = (
    (by_month["baseline_logloss"] - by_month["model_logloss"]) / by_month["baseline_logloss"] * 100
)

by_month = by_month.round(4)
print(by_month.to_string())

by_month.to_csv("notebooks/seasonal_breakdown.csv")
print("\nWrote notebooks/seasonal_breakdown.csv")
print(f"\nBest month by Brier edge: {by_month['brier_edge_pct'].idxmax()} "
      f"({by_month['brier_edge_pct'].max():.2f}% improvement)")
print(f"Worst month by Brier edge: {by_month['brier_edge_pct'].idxmin()} "
      f"({by_month['brier_edge_pct'].min():.2f}% improvement)")
