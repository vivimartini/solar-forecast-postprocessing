# src/lagged_ensemble.py
"""Signature feature: lagged-ensemble dispersion.

For a forecast issued at t0 for valid hour `step`, the lagged ensemble is the set of
EARLIER forecasts for the same `step`, issued strictly before t0. Their spread (std) is
a leak-safe proxy for forecast uncertainty: agreement -> settled weather, disagreement
-> volatile. Built in issuance order, using the latest k prior members.
"""
import pandas as pd


def add_dispersion(forecasts, k=4, out_col="disp_mw"):
    """Add `out_col` = std of the latest k PRIOR forecasts for the same valid hour.
    shift(1) excludes the current forecast, so only earlier members are used."""
    f = forecasts.sort_values(["step", "issued_at"]).copy()
    f[out_col] = (
        f.groupby("step")["fc_mw"]
         .transform(lambda s: s.shift(1).rolling(k, min_periods=2).std())
    )
    return f