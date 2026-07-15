import pandas as pd

from src.data import add_climatology_baseline


def _synthetic():
    rows = []
    for i, d in enumerate(pd.date_range("2024-06-01", periods=6, freq="D")):
        rows.append({
            "step": d + pd.Timedelta(hours=12),
            "issued_at": d + pd.Timedelta(hours=6),
            "actual_mw": 1000 + 100 * i,
            "fc_mw": 1000.0,
        })
    return pd.DataFrame(rows)


def test_climatology_ignores_same_day_actual():
    df = _synthetic()
    out = add_climatology_baseline(df, min_obs=1)
    assert pd.isna(out.iloc[0].clim_mw)


def test_climatology_uses_only_past_valid_days():
    df = _synthetic()
    out = add_climatology_baseline(df, min_obs=1)
    row = out.iloc[3]
    assert abs(row.clim_mw - 1100.0) < 1.0


def test_changing_future_actuals_does_not_change_past_climatology():
    df = _synthetic()
    base = add_climatology_baseline(df, min_obs=1)
    df2 = df.copy()
    df2.loc[5, "actual_mw"] = 99999
    alt = add_climatology_baseline(df2, min_obs=1)
    assert base.iloc[2].clim_mw == alt.iloc[2].clim_mw
    assert base.iloc[3].clim_mw == alt.iloc[3].clim_mw
