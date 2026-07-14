import pandas as pd

from src.online_bias import add_online_bias


def _synthetic():
    rows = []
    for i, d in enumerate(pd.date_range("2024-06-01", periods=5, freq="D")):
        rows.append({
            "step": d + pd.Timedelta(hours=11),
            "issued_at": d + pd.Timedelta(hours=5),
            "actual_mw": 5000 + 100 * i,
            "fc_mw": 5000.0,
        })
    return pd.DataFrame(rows)


def test_bias_ignores_same_day_residual():
    df = _synthetic()
    out = add_online_bias(df, window_days=45, per_hour=True)
    row1 = out.iloc[1]
    assert abs(row1.online_bias) < 1.0


def test_bias_uses_only_past_days():
    df = _synthetic()
    out = add_online_bias(df, window_days=45, per_hour=True)
    row = out.iloc[4]
    assert abs(row.online_bias - 150.0) < 10.0


def test_changing_future_actuals_does_not_change_past_bias():
    df = _synthetic()
    base = add_online_bias(df, window_days=45, per_hour=True)
    df2 = df.copy()
    df2.loc[2, "actual_mw"] = 99999
    alt = add_online_bias(df2, window_days=45, per_hour=True)
    assert base.iloc[0].online_bias == alt.iloc[0].online_bias
    assert base.iloc[1].online_bias == alt.iloc[1].online_bias
