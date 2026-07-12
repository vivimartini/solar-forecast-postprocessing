# scripts/01_eda.py
"""Exploratory diagnostics: understand the data and lock the baseline.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import load_config, build_dataset


def rmse(x):
    return float(np.sqrt(np.mean(np.square(x))))


def main():
    cfg = load_config()
    df = build_dataset(cfg)
    day = df[df["is_day"]].copy()

    print("=== coverage ===")
    print("rows total:", len(df), "| daytime:", len(day))
    print("valid-time range:", df.step.min(), "->", df.step.max())
    print("op_lead_h range:", round(df.op_lead_h.min(), 2), "->", round(df.op_lead_h.max(), 2))

    print("\n=== generation upper envelope by year (99.9th pct actual, MW) ===")
    yr = day.assign(year=day.step.dt.year)
    print(yr.groupby("year")["actual_mw"].quantile(0.999).round(0).to_dict())

    print("\n=== baseline daytime RMSE by operational-lead band (MW) ===")
    for lo, hi in [(0, 6), (6, 12), (12, 24), (24, 36)]:
        s = day[(day.op_lead_h > lo) & (day.op_lead_h <= hi)]
        print(f"  {lo:2d}-{hi:2d}h  n={len(s):6d}  RMSE={rmse(s.residual_mw):7.1f}  bias={s.residual_mw.mean():7.1f}")
    print(f"  overall 0-36h  RMSE={rmse(day.residual_mw):.1f}")

    day["hour"] = day.step.dt.hour
    day["month"] = day.step.dt.month
    by_hour = day.groupby("hour")["residual_mw"].apply(rmse)
    by_month = day.groupby("month")["residual_mw"].apply(rmse)

    Path("outputs").mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(by_hour.index, by_hour.values, color="#c9736f")
    ax[0].set(title="Baseline RMSE by hour (UTC)", xlabel="hour", ylabel="MW")
    ax[1].bar(by_month.index, by_month.values, color="#6b8cce")
    ax[1].set(title="Baseline RMSE by month", xlabel="month", ylabel="MW")
    fig.tight_layout()
    fig.savefig("outputs/eda_diagnostics.png", dpi=110)
    print("\nsaved outputs/eda_diagnostics.png")
    print("worst hours (RMSE):", by_hour.sort_values(ascending=False).head(4).round(0).to_dict())


if __name__ == "__main__":
    main()