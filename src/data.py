# src/data.py
"""Load, align and clean the forecast + actuals data.

Key decisions (see DECISIONS.md):
- Forecasts are in GW, actuals in MW -> multiply forecast by 1000 to compare.
- Operational lead = step - issued_at (when the forecast is actually usable),
  NOT step - init_time. Rows with lead <= 0 are dropped (valid time already past).
- Actuals are 15-min instantaneous snapshots; the forecast is "generation within
  the hour", so we average the four snapshots in each hour to line them up.
"""
import yaml
import numpy as np
import pandas as pd
import pvlib

from src.lagged_ensemble import add_dispersion
from src.capacity import add_capacity_proxy


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_raw(cfg):
    fc = pd.read_parquet(cfg["paths"]["forecasts"])
    ac = pd.read_parquet(cfg["paths"]["actuals"]).dropna(subset=["value"])
    return fc, ac


def hourly_actuals(actuals):
    """15-min instantaneous snapshots -> hourly mean (MW), matching the
    forecast's 'generation within the hour' definition."""
    return (actuals.set_index("time")["value"]
                   .resample("1h").mean()
                   .rename("actual_mw"))


def prepare_forecasts(forecasts):
    """Add reconciled units (GW->MW) and both lead definitions."""
    fc = forecasts.copy()
    fc["fc_mw"] = fc["value"] * 1000.0
    fc["op_lead_h"] = (fc["step"] - fc["issued_at"]).dt.total_seconds() / 3600
    fc["model_age_h"] = (fc["step"] - fc["init_time"]).dt.total_seconds() / 3600
    return fc


def add_solar_position(df, cfg):
    loc = pvlib.location.Location(cfg["location"]["lat"], cfg["location"]["lon"], tz=cfg["location"]["tz"])
    idx = pd.DatetimeIndex(df["step"])
    df = df.copy()
    df["solar_elevation"] = loc.get_solarposition(idx)["elevation"].values
    df["clearsky_ghi"] = loc.get_clearsky(idx)["ghi"].values     # NEW: the cloudless ceiling
    return df


def add_forecast_shape(forecasts, group_col="issued_at"):
    """Neighbouring-hour forecast values from the SAME run — leak-safe, since one issuance
    covers many valid hours and all of them are known at issue time.
    fc_prev1h / fc_next1h = this run's forecast for the valid hour one step before / after."""
    f = forecasts.sort_values([group_col, "step"]).copy()
    g = f.groupby(group_col)["fc_mw"]
    f["fc_prev1h"] = g.shift(1)
    f["fc_next1h"] = g.shift(-1)
    return f


def build_dataset(cfg, lead_band=None):
    """One clean row per forecast case, joined to its hourly actual."""
    fc, ac = load_raw(cfg)
    fc = prepare_forecasts(fc)
    fc = add_dispersion(fc, k=cfg["lagged_ensemble"]["k_default"])   
    fc = add_capacity_proxy(fc, window_days=cfg["capacity"]["window_days"])   # fleet proxy
    fc = add_forecast_shape(fc)   

    lo, hi = lead_band or cfg["lead_band_h"]
    fc = fc[(fc["op_lead_h"] > lo) & (fc["op_lead_h"] <= hi)]        # usable leads only

    ah = hourly_actuals(ac)
    df = (fc.merge(ah, left_on="step", right_index=True, how="inner")
            .dropna(subset=["actual_mw"]))

    df = add_solar_position(df, cfg)
    df["is_day"] = df["solar_elevation"] > cfg["daytime_elevation_deg"]
    df["residual_mw"] = df["actual_mw"] - df["fc_mw"]  
    df["residual_norm"] = df["residual_mw"] / df["cap_mw"]                    # error as fraction of fleet       
    return df.sort_values(["issued_at", "step"]).reset_index(drop=True)  #for reproducibility