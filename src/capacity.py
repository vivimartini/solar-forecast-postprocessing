# src/capacity.py
"""Fleet-size proxy: trailing peak of the forecast itself.

No installed-capacity data in this challenge, but the max forecast over the last few
weeks tracks it well enough (~41 -> ~56 GW over the record). Used to express residuals
as a fraction of the fleet so the target doesn't grow with the buildout.
"""
import pandas as pd


def add_capacity_proxy(forecasts, window_days=60, out_col="cap_mw"):
    f = forecasts.copy()
    day = f["issued_at"].dt.floor("D")
    daily_peak = f.assign(_d=day).groupby("_d")["fc_mw"].max().sort_index()
    # shift(1) so today's own peak doesn't feed into today's cap
    cap = (daily_peak.rolling(f"{window_days}D", min_periods=1).max()
                     .shift(1).ffill().bfill())
    f[out_col] = day.map(cap)
    return f