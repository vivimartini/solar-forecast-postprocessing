# src/features.py
"""Issue-time-safe features. Every column is known at issuance:
the forecast itself, lead, calendar, and deterministic solar geometry.
Backward-looking features (lagged-ensemble spread) are added later, as an ablation.
"""
import numpy as np

BASE_FEATURES = [
    "fc_mw", "op_lead_h", "model_age_h",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "solar_elevation",
]

SHAPE_FEATURES = ["fc_prev1h", "fc_next1h", "fc_ramp", "fc_curv", "clearsky_ghi", "fc_over_cs"]
RICH_FEATURES = BASE_FEATURES + SHAPE_FEATURES


def add_shape_derived(df):
    df = df.copy()
    df["fc_ramp"] = df["fc_mw"] - df["fc_prev1h"]                       # local slope
    df["fc_curv"] = df["fc_next1h"] - 2 * df["fc_mw"] + df["fc_prev1h"] # curvature (ramp bending)
    df["fc_over_cs"] = df["fc_mw"] / (df["clearsky_ghi"] + 1.0)         # clear-sky-scaled proxy (cloudiness)
    return df


def add_calendar(df):
    """Cyclical encodings of hour-of-day and day-of-year."""
    df = df.copy()
    hour = df["step"].dt.hour + df["step"].dt.minute / 60.0
    doy = df["step"].dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def make_features(df, feature_cols=BASE_FEATURES):
    """Return (X, y): features known at issuance, and y = residual (actual - forecast).
    Shape-derived columns are built on demand when requested in feature_cols."""
    df = add_calendar(df)
    if any(c in feature_cols for c in ("fc_ramp", "fc_curv", "fc_over_cs")):
        df = add_shape_derived(df)
    return df[feature_cols], df["residual_mw"]