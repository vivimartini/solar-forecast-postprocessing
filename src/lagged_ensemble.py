# src/lagged_ensemble.py
"""Signature feature: lagged-ensemble dispersion.

For a forecast issued at t0 for valid hour `step`, the lagged ensemble is the set of
EARLIER forecasts for the same `step`, issued strictly before t0. Their spread (std) is
a leak-safe proxy for forecast uncertainty: agreement -> settled weather, disagreement
-> volatile. Built in issuance order, using the latest k prior members.
"""
import pandas as pd

def add_forecast_shape(forecasts):
    """Leak-safe (same issued_at): the forecast for step-1h and step+1h from the SAME cycle,
    so the model can see the forecast's local slope/curvature (the ramp it often mistimes)."""
    f = forecasts.copy()
    base = f[["issued_at", "step", "fc_mw"]]
    prev = base.rename(columns={"fc_mw": "fc_prev1h"}).copy(); prev["step"] += pd.Timedelta(hours=1)
    nxt  = base.rename(columns={"fc_mw": "fc_next1h"}).copy(); nxt["step"]  -= pd.Timedelta(hours=1)
    return f.merge(prev, on=["issued_at", "step"], how="left").merge(nxt, on=["issued_at", "step"], how="left")


def add_dispersion(forecasts, k=4, out_col="disp_mw"):
    """Add `out_col` = std of the latest k PRIOR forecasts for the same valid hour.
    shift(1) excludes the current forecast, so only earlier members are used."""
    f = forecasts.sort_values(["step", "issued_at"]).copy()
    f[out_col] = (
        f.groupby("step")["fc_mw"]
         .transform(lambda s: s.shift(1).rolling(k, min_periods=2).std())
    )
    return f