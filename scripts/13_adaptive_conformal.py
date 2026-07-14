import numpy as np, pandas as pd
from src.data import load_config, build_dataset
from src.features import make_features, RICH_FEATURES
from src.splits import rolling_origin_splits
from src.models import train_quantile_model
from src.metrics import coverage, interval_width

LO, HI, ALPHA = 0.1, 0.9, 0.2


def run(day, folds, adaptive):
    X, _ = make_features(day, feature_cols=RICH_FEATURES)
    yn = day["residual_norm"]
    med = day["disp_mw"].median()
    rows, cov_agg, w_agg = [], [], []
    for tr, va in folds:
        s = tr[np.argsort(day.loc[tr, "issued_at"].values)]
        n = len(s); fit, es, cal = s[:int(.70*n)], s[int(.70*n):int(.85*n)], s[int(.85*n):]
        fc = lambda i: day.loc[i, "fc_mw"].values; cap = lambda i: day.loc[i, "cap_mw"].values
        def qp(m, i): return fc(i) + m.predict(X.iloc[i]) * cap(i)
        u = lambda i: (day.loc[i, "disp_mw"].fillna(med).clip(lower=med*0.2).values
                       if adaptive else np.ones(len(i)))

        m_lo = train_quantile_model(X.iloc[fit], yn.iloc[fit], LO, X.iloc[es], yn.iloc[es])
        m_hi = train_quantile_model(X.iloc[fit], yn.iloc[fit], HI, X.iloc[es], yn.iloc[es])

        y_cal = day.loc[cal, "actual_mw"].values
        E = np.maximum(qp(m_lo, cal) - y_cal, y_cal - qp(m_hi, cal)) / u(cal)
        level = min(1.0, np.ceil((len(E)+1)*(1-ALPHA))/len(E)); Q = np.quantile(E, level)

        y = day.loc[va, "actual_mw"].values; uv = u(va)
        p10 = np.clip(qp(m_lo, va) - Q*uv, 0, None)
        p90 = np.clip(qp(m_hi, va) + Q*uv, 0, None)
        cov_agg.append(coverage(y, p10, p90)); w_agg.append(interval_width(p10, p90))
        d = day.loc[va].copy(); d["_in"] = ((y >= p10) & (y <= p90)).astype(int)
        rows.append(d[["disp_mw", "_in"]])

    allrows = pd.concat(rows).dropna(subset=["disp_mw"])
    allrows["band"] = pd.qcut(allrows["disp_mw"], 3, labels=["calm", "mid", "volatile"])
    cond = allrows.groupby("band", observed=True)["_in"].mean().round(3).to_dict()
    return np.mean(cov_agg)*100, np.mean(w_agg), cond


def main():
    cfg = load_config(); day = build_dataset(cfg); day = day[day.is_day].reset_index(drop=True)
    v = cfg["validation"]; folds, _ = rolling_origin_splits(day, v["n_folds"], v["embargo_days"], v["sealed_test_frac"])
    for name, adap in [("global conformal", False), ("adaptive (dispersion-scaled)", True)]:
        cov, w, cond = run(day, folds, adap)
        print(f"{name:28s}: coverage {cov:5.1f}% | width {w:7.1f}")
        print(f"    conditional coverage by instability: {cond}")


if __name__ == "__main__":
    main()
