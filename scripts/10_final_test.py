# scripts/10_final_test.py
"""FINAL evaluation on the sealed test set (touched once).
Point = fc + per-hour online bias (the static GBDT scored -15% here, see DECISIONS.md).
Intervals = quantile GBDTs + online ACI with dispersion scaling.
Also dumps outputs/predictions.csv for make_figures.py.
Run: PYTHONPATH=. python scripts/10_final_test.py
"""
from pathlib import Path
import numpy as np, pandas as pd
from src.data import load_config, build_dataset
from src.features import make_features, RICH_FEATURES
from src.splits import train_test_seal
from src.models import train_quantile_model
from src.online_bias import add_online_bias
from src.metrics import rmse, skill_score

LO, HI, TARGET, GAMMA = 0.1, 0.9, 0.8, 0.05


def main():
    cfg = load_config(); day = build_dataset(cfg); day = day[day.is_day].reset_index(drop=True)
    ob = cfg["online_bias"]
    day = add_online_bias(day, window_days=ob["window_days"], per_hour=ob["per_hour"])  # adaptive point correction
    X, _ = make_features(day, feature_cols=RICH_FEATURES); yn = day["residual_norm"]
    med = day["disp_mw"].median()

    dev_mask, test_mask = train_test_seal(day, cfg["validation"]["sealed_test_frac"])
    dev = np.where(dev_mask)[0]
    ds = dev[np.argsort(day.loc[dev, "issued_at"].values)]
    cut = int(0.85*len(ds)); fit, es = ds[:cut], ds[cut:]
    test = np.where(test_mask)[0]
    test = test[np.argsort(day.loc[test, "issued_at"].values)]        # time order for ACI

    fc = lambda i: day.loc[i, "fc_mw"].values; cap = lambda i: day.loc[i, "cap_mw"].values
    actual = day.loc[test, "actual_mw"].values

    corrected = np.clip(fc(test) + day.loc[test, "online_bias"].values, 0, None)   # adaptive point model
    base_rmse, mod_rmse = rmse(actual, fc(test)), rmse(actual, corrected)

    def qp(m, i): return fc(i) + m.predict(X.iloc[i]) * cap(i)
    m_lo = train_quantile_model(X.iloc[fit], yn.iloc[fit], LO, X.iloc[es], yn.iloc[es])
    m_hi = train_quantile_model(X.iloc[fit], yn.iloc[fit], HI, X.iloc[es], yn.iloc[es])
    qlo, qhi = qp(m_lo, test), qp(m_hi, test)
    u = day.loc[test, "disp_mw"].fillna(med).clip(lower=med*0.2).values

    Qm = 0.0; inside = np.zeros(len(actual), int); w = np.zeros(len(actual))
    p10 = np.zeros(len(actual)); p90 = np.zeros(len(actual))
    for t in range(len(actual)):
        add = max(Qm, 0.0) * u[t]; lo_t, hi_t = max(qlo[t]-add, 0.0), qhi[t]+add
        p10[t], p90[t] = lo_t, hi_t
        inside[t] = int(lo_t <= actual[t] <= hi_t); w[t] = hi_t - lo_t
        Qm += GAMMA * ((1-inside[t]) - (1-TARGET))
    
    Path("outputs").mkdir(exist_ok=True)
    pd.DataFrame({
        "issued_at": day.loc[test, "issued_at"].values,
        "step":      day.loc[test, "step"].values,
        "fc_mw":     fc(test),
        "corrected": corrected,
        "p10": p10, "p90": p90,
        "actual_mw": actual,
        "disp":      day.loc[test, "disp_mw"].values,
    }).to_csv("outputs/predictions.csv", index=False)

    print("=== SEALED TEST (touched once) ===")
    print(f"rows {len(test)} | {day.loc[test,'issued_at'].min().date()} -> {day.loc[test,'issued_at'].max().date()}")
    print(f"POINT:     baseline RMSE {base_rmse:7.1f} -> model {mod_rmse:7.1f} | skill {skill_score(mod_rmse, base_rmse)*100:+.2f}%")
    print(f"INTERVALS: coverage {inside.mean()*100:5.1f}% | width {w.mean():7.1f}")
    t = day.loc[test].copy(); t["_in"] = inside; t["abserr"] = np.abs(actual - fc(test))
    t = t.dropna(subset=["disp_mw"]); t["band"] = pd.qcut(t["disp_mw"], 3, labels=["calm", "mid", "volatile"])
    print("  conditional coverage:", t.groupby("band", observed=True)["_in"].mean().round(3).to_dict())
    print("  metric(b) |error| by instability (MW):", t.groupby("band", observed=True)["abserr"].mean().round(0).to_dict())


if __name__ == "__main__":
    main()