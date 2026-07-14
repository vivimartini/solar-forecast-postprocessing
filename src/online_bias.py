# src/online_bias.py
"""Adaptive (online) bias correction. The forecast's bias drifts over the years and even
flips sign, so a static learned correction fails out-of-sample. A slowly-updating estimate
of the RECENT bias tracks the drift. Leak-safe: uses only realized past days (shift(1)).
"""
import pandas as pd


def add_online_bias(df, window_days=45, out_col="online_bias"):
    """df needs step, issued_at, actual_mw, fc_mw (daytime rows).
    out_col = recent mean(actual - forecast), as-of the issue day (past days only)."""
    resid = df["actual_mw"] - df["fc_mw"]
    daily = resid.groupby(df["step"].dt.floor("D")).mean().sort_index()
    b = daily.rolling(f"{window_days}D", min_periods=3).mean().shift(1).ffill()
    df = df.copy()
    df[out_col] = df["issued_at"].dt.floor("D").map(b).fillna(0.0).values
    return df