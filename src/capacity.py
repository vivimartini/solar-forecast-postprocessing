"""Fleet cap proxy from trailing daily forecast peaks."""
import pandas as pd


def add_capacity_proxy(forecasts, window_days=60, out_col="cap_mw"):
    f = forecasts.copy()
    day = f["issued_at"].dt.floor("D")
    daily_peak = f.assign(_d=day).groupby("_d")["fc_mw"].max().sort_index()
    cap = (daily_peak.rolling(f"{window_days}D", min_periods=1).max()
                     .shift(1).ffill().bfill())  # shift(1): today's peak doesn't set today's cap
    f[out_col] = day.map(cap)
    return f
