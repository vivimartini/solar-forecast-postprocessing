"""Rolling trailing bias, per hour-of-day. shift(1) on valid days so issue-day rows don't peek."""
import pandas as pd


def add_online_bias(df, window_days=60, per_hour=True, out_col="online_bias"):
    d = df.copy()
    d["_r"] = d["actual_mw"] - d["fc_mw"]
    d["_vday"] = d["step"].dt.floor("D")
    d["_iday"] = d["issued_at"].dt.floor("D")
    if per_hour:
        d["_hour"] = d["step"].dt.hour
        piv = d.pivot_table(index="_vday", columns="_hour", values="_r", aggfunc="mean").sort_index()
        rolled = piv.rolling(f"{window_days}D", min_periods=3).mean().shift(1).ffill()
        rs = rolled.stack().rename("_b").reset_index()
        rs.columns = ["_bday", "_hour", "_b"]
        merged = d.merge(rs, left_on=["_iday", "_hour"], right_on=["_bday", "_hour"], how="left")
        out = df.copy(); out[out_col] = merged["_b"].fillna(0.0).values
    else:
        daily = d.groupby("_vday")["_r"].mean().sort_index()
        b = daily.rolling(f"{window_days}D", min_periods=3).mean().shift(1).ffill()
        out = df.copy(); out[out_col] = d["_iday"].map(b).fillna(0.0).values
    return out
