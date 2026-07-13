# src/capacity.py
"""Leak-safe fleet-size proxy = the recent peak of the forecast.
The biggest forecast over recent weeks tracks how much solar is installed, is always
known at issue time, and lets us express errors as a FRACTION of the fleet (which stays
stationary as the fleet grows).
"""
import pandas as pd


def add_capacity_proxy(forecasts, window_days=60, out_col="cap_mw"):
    f = forecasts.copy()
    day = f["issued_at"].dt.floor("D")
    daily_peak = f.assign(_d=day).groupby("_d")["fc_mw"].max().sort_index()
    # trailing max over PRIOR days only (shift(1) drops the current day = leak-safe)
    cap = (daily_peak.rolling(f"{window_days}D", min_periods=1).max()
                     .shift(1).ffill().bfill())
    f[out_col] = day.map(cap)
    return f