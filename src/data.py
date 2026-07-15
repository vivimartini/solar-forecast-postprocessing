"""Load and align forecast + actuals. GW->MW, operational lead, hourly actuals."""
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
    # actuals are 15-min snapshots; forecast is hourly generation -> mean to align
    return (actuals.set_index("time")["value"]
                   .resample("1h").mean()
                   .rename("actual_mw"))


def prepare_forecasts(forecasts):
    fc = forecasts.copy()
    fc["fc_mw"] = fc["value"] * 1000.0
    fc["op_lead_h"] = (fc["step"] - fc["issued_at"]).dt.total_seconds() / 3600   # usable lead
    fc["model_age_h"] = (fc["step"] - fc["init_time"]).dt.total_seconds() / 3600  # model run age
    return fc


def add_solar_position(df, cfg):
    loc = pvlib.location.Location(cfg["location"]["lat"], cfg["location"]["lon"], tz=cfg["location"]["tz"])
    idx = pd.DatetimeIndex(df["step"])
    df = df.copy()
    df["solar_elevation"] = loc.get_solarposition(idx)["elevation"].values
    df["clearsky_ghi"] = loc.get_clearsky(idx)["ghi"].values
    return df


def add_forecast_shape(forecasts, group_col="issued_at"):
    f = forecasts.sort_values([group_col, "step"]).copy()
    g = f.groupby(group_col)["fc_mw"]
    f["fc_prev1h"] = g.shift(1)
    f["fc_next1h"] = g.shift(-1)
    return f

def add_next_revision(forecasts):
    """Label for metric (b) only — future info, not a feature."""
    f = forecasts.sort_values(["step", "issued_at"]).copy()
    f["next_revision"] = f.groupby("step")["fc_mw"].shift(-1) - f["fc_mw"]
    return f


def add_climatology_baseline(df, out_col="clim_mw", min_obs=7):
    """Month×hour mean of actuals at valid times strictly before issue day."""
    d = df.copy()
    d["_idx"] = np.arange(len(d))
    d["_m"] = d["step"].dt.month
    d["_h"] = d["step"].dt.hour
    d["_iday"] = d["issued_at"].dt.floor("D")
    d["_vday"] = d["step"].dt.floor("D")

    daily = (
        d.groupby(["_vday", "_m", "_h"], observed=True)["actual_mw"]
         .mean()
         .reset_index()
         .sort_values(["_m", "_h", "_vday"])
    )
    daily[out_col] = (
        daily.groupby(["_m", "_h"], observed=True)["actual_mw"]
             .transform(lambda s: s.expanding(min_periods=min_obs).mean())
    )

    lookup = daily.rename(columns={"_vday": "_ref_day"})
    merged = pd.merge_asof(
        d.sort_values("_iday"),
        lookup.sort_values("_ref_day"),
        left_on="_iday",
        right_on="_ref_day",
        by=["_m", "_h"],
        direction="backward",
        allow_exact_matches=False,
    )
    out = df.copy()
    out[out_col] = merged.sort_values("_idx")[out_col].values
    return out


def build_dataset(cfg, lead_band=None):
    fc, ac = load_raw(cfg)
    fc = prepare_forecasts(fc)
    fc = add_dispersion(fc, k=cfg["lagged_ensemble"]["k_default"])
    fc = add_capacity_proxy(fc, window_days=cfg["capacity"]["window_days"])
    fc = add_forecast_shape(fc)
    fc = add_next_revision(fc)

    lo, hi = lead_band or cfg["lead_band_h"]
    fc = fc[(fc["op_lead_h"] > lo) & (fc["op_lead_h"] <= hi)]  # drop lead<=0 (valid time passed)

    ah = hourly_actuals(ac)
    df = (fc.merge(ah, left_on="step", right_index=True, how="inner")
            .dropna(subset=["actual_mw"]))

    df = add_solar_position(df, cfg)
    df["is_day"] = df["solar_elevation"] > cfg["daytime_elevation_deg"]
    df["residual_mw"] = df["actual_mw"] - df["fc_mw"]
    df["residual_norm"] = df["residual_mw"] / df["cap_mw"]
    return df.sort_values(["issued_at", "step"]).reset_index(drop=True)
