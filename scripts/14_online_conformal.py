# scripts/14_online_conformal.py
"""Online adaptive conformal (ACI; Gibbs & Candes 2021 / Suresh 2026).
Self-corrects coverage while moving through the DRIFTING validation block, so it works
where offline conformal fails. Optional dispersion-scaling = the full synthesis
(online adaptation fixes the drift; instability signal allocates width by regime).
Run: PYTHONPATH=. python scripts/14_online_conformal.py
"""
import numpy as np, pandas as pd
from src.data import load_config, build_dataset
from src.features import make_features, RICH_FEATURES
from src.splits import rolling_origin_splits
from src.models import train_quantile_model

LO, HI, TARGET, GAMMA = 0.1, 0.9, 0.8, 0.05


def run(day, folds, scaled):
    X, _ = make_features(day, feature_cols=RICH_FEATURES)
    yn = day["residual_norm"]; med = day["disp_mw"].median()
    covs, widths, rows = [], [], []
    for tr, va in folds:
        s = tr[np.argsort(day.loc[tr, "issued_at"].values)]
        cut = int(0.85 * len(s)); fit, es = s[:cut], s[cut:]
        fc = lambda i: day.loc[i, "fc_mw"].values; cap = lambda i: day.loc[i, "cap_mw"].values
        def qp(m, i): return fc(i) + m.predict(X.iloc[i]) * cap(i)
        m_lo = train_quantile_model(X.iloc[fit], yn.iloc[fit], LO, X.iloc[es], yn.iloc[es])
        m_hi = train_quantile_model(X.iloc[fit], yn.iloc[fit], HI, X.iloc[es], yn.iloc[es])

        va = va[np.argsort(day.loc[va, "issued_at"].values)]           # process in time order
        qlo, qhi, y = qp(m_lo, va), qp(m_hi, va), day.loc[va, "actual_mw"].values
        u = (day.loc[va, "disp_mw"].fillna(med).clip(lower=med*0.2).values
             if scaled else np.full(len(va), (qhi - qlo).mean() / 2))   # scale for the offset
        Qm, inside, w = 0.0, np.zeros(len(y), int), np.zeros(len(y))
        for t in range(len(y)):
            add = max(Qm, 0.0) * u[t]
            lo_t, hi_t = max(qlo[t] - add, 0.0), qhi[t] + add
            inside[t] = int(lo_t <= y[t] <= hi_t); w[t] = hi_t - lo_t
            Qm += GAMMA * ((1 - inside[t]) - (1 - TARGET))               # widen if missing >20%
        covs.append(inside.mean()); widths.append(w.mean())
        d = day.loc[va].copy(); d["_in"] = inside; rows.append(d[["disp_mw", "_in"]])
    allrows = pd.concat(rows).dropna(subset=["disp_mw"])
    allrows["band"] = pd.qcut(allrows["disp_mw"], 3, labels=["calm", "mid", "volatile"])
    return np.mean(covs) * 100, np.mean(widths), allrows.groupby("band", observed=True)["_in"].mean().round(3).to_dict()


def main():
    cfg = load_config(); day = build_dataset(cfg); day = day[day.is_day].reset_index(drop=True)
    v = cfg["validation"]; folds, _ = rolling_origin_splits(day, v["n_folds"], v["embargo_days"], v["sealed_test_frac"])
    for name, sc in [("online ACI (global)", False), ("online ACI (dispersion-scaled)", True)]:
        cov, w, cond = run(day, folds, sc)
        print(f"{name:32s}: coverage {cov:5.1f}% | width {w:7.1f}")
        print(f"    conditional coverage by instability: {cond}")


if __name__ == "__main__":
    main()