from pathlib import Path
import numpy as np, pandas as pd
from src.data import load_config, build_dataset
from src.features import make_features, RICH_FEATURES
from src.splits import train_test_seal
from src.models import train_quantile_model, train_residual_model
from src.online_bias import add_online_bias
from src.metrics import rmse, skill_score

LO, HI, TARGET, GAMMA = 0.1, 0.9, 0.8, 0.05


def main():
    cfg = load_config(); day = build_dataset(cfg); day = day[day.is_day].reset_index(drop=True)
    ob = cfg["online_bias"]
    day = add_online_bias(day, window_days=ob["window_days"], per_hour=ob["per_hour"])
    day["residual_debias"] = day["residual_mw"] - day["online_bias"]
    X, _ = make_features(day, feature_cols=RICH_FEATURES); yn = day["residual_norm"]
    med = day["disp_mw"].median()

    dev_mask, test_mask = train_test_seal(day, cfg["validation"]["sealed_test_frac"])
    dev = np.where(dev_mask)[0]
    ds = dev[np.argsort(day.loc[dev, "issued_at"].values)]
    cut = int(0.85*len(ds)); fit, es = ds[:cut], ds[cut:]
    test = np.where(test_mask)[0]
    test = test[np.argsort(day.loc[test, "issued_at"].values)]

    fc = lambda i: day.loc[i, "fc_mw"].values; cap = lambda i: day.loc[i, "cap_mw"].values
    actual = day.loc[test, "actual_mw"].values

    corrected = np.clip(fc(test) + day.loc[test, "online_bias"].values, 0, None)
    base_rmse, mod_rmse = rmse(actual, fc(test)), rmse(actual, corrected)

    y_deb = day["residual_debias"]
    gbdt = train_residual_model(X.iloc[fit], y_deb.iloc[fit], X.iloc[es], y_deb.iloc[es])
    corrected2 = np.clip(fc(test) + day.loc[test, "online_bias"].values + gbdt.predict(X.iloc[test]), 0, None)
    mod2_rmse = rmse(actual, corrected2)

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
    pred = day.loc[test, ["issued_at", "step", "op_lead_h", "disp_mw"]].copy()
    pred["fc_mw"] = fc(test)
    pred["corrected"] = corrected
    pred["corrected_debias_gbdt"] = corrected2
    pred["p10"] = p10
    pred["p90"] = p90
    pred["actual_mw"] = actual
    pred.rename(columns={"disp_mw": "disp"}, inplace=True)
    pred.to_csv("outputs/predictions.csv", index=False)

    print("=== SEALED TEST (touched once) ===")
    print(f"rows {len(test)} | {day.loc[test,'issued_at'].min().date()} -> {day.loc[test,'issued_at'].max().date()}")
    print(f"POINT:     baseline RMSE {base_rmse:7.1f} -> model {mod_rmse:7.1f} | skill {skill_score(mod_rmse, base_rmse)*100:+.2f}%")
    print(f"POINT+GBDT: baseline RMSE {base_rmse:7.1f} -> model {mod2_rmse:7.1f} | skill {skill_score(mod2_rmse, base_rmse)*100:+.2f}%"
          f"  (online bias + GBDT on de-biased residual)")
    print(f"INTERVALS: coverage {inside.mean()*100:5.1f}% | width {w.mean():7.1f}")

    print("\nPOINT skill by lead band:")
    for lo, hi in [(0, 6), (6, 12), (12, 24), (24, 36)]:
        m = (day.loc[test, "op_lead_h"] > lo) & (day.loc[test, "op_lead_h"] <= hi)
        if not m.any():
            continue
        a, f, c = actual[m], fc(test)[m], corrected[m]
        print(f"  {lo:2d}-{hi:2d}h  n={int(m.sum()):5d}  skill {skill_score(rmse(a, c), rmse(a, f))*100:+6.2f}%")

    print("\nPOINT skill by month (bias drift visible here):")
    t = day.loc[test, ["step", "actual_mw", "fc_mw"]].copy()
    t["corrected"] = corrected
    t["ym"] = t.step.dt.to_period("M").astype(str)
    for ym, g in t.groupby("ym"):
        sk = skill_score(rmse(g.actual_mw, g.corrected), rmse(g.actual_mw, g.fc_mw)) * 100
        print(f"  {ym}: skill {sk:+6.2f}%  n={len(g):5d}")

    t = day.loc[test].copy(); t["_in"] = inside; t["abserr"] = np.abs(actual - fc(test))
    t = t.dropna(subset=["disp_mw"]); t["band"] = pd.qcut(t["disp_mw"], 3, labels=["calm", "mid", "volatile"])
    print("  conditional coverage:", t.groupby("band", observed=True)["_in"].mean().round(3).to_dict())
    print("  metric(b) |error| by instability (MW):", t.groupby("band", observed=True)["abserr"].mean().round(0).to_dict())


if __name__ == "__main__":
    main()
