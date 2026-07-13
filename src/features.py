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
    """Return (X, y): features known at issuance, and y = residual (actual - forecast)."""
    df = add_calendar(df)
    return df[feature_cols], df["residual_mw"]