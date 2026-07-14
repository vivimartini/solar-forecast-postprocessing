"""Std of prior issuances for the same valid hour — instability proxy."""
import pandas as pd


def add_dispersion(forecasts, k=4, out_col="disp_mw"):
    f = forecasts.sort_values(["step", "issued_at"]).copy()
    f[out_col] = (
        f.groupby("step")["fc_mw"]
         .transform(lambda s: s.shift(1).rolling(k, min_periods=2).std())  # shift(1) = no leak
    )
    return f
