# src/lagged_ensemble.py
"""Lagged-ensemble dispersion.

Idea: every valid hour gets forecast several times by successive issuances, so the
earlier runs form a free poor-man's ensemble. If they disagree (high std), the weather
situation is unsettled and the current forecast is probably less trustworthy.
Only uses runs issued strictly before the current one, so safe at issue time.
"""
import pandas as pd


def add_dispersion(forecasts, k=4, out_col="disp_mw"):
    """out_col = std of the latest k forecasts issued BEFORE this one, same valid hour.
    The shift(1) is what keeps it leak-safe -- drop it and the current forecast
    leaks into its own spread."""
    f = forecasts.sort_values(["step", "issued_at"]).copy()
    f[out_col] = (
        f.groupby("step")["fc_mw"]
         .transform(lambda s: s.shift(1).rolling(k, min_periods=2).std())
    )
    return f