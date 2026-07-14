# scripts/13b_conformal_diag.py
"""Quick and dirty: why did 13 invert? Print Q's sign and the cal-slice coverage per fold.
Answer: base intervals OVER-cover the (older) cal slice on folds 1-2 -> Q < 0 -> conformal
narrows exactly where the drifted future needed widening. Not a bug, an exchangeability break.
Run: PYTHONPATH=. python scripts/13b_conformal_diag.py
"""
import numpy as np
from src.data import load_config, build_dataset
from src.features import make_features, RICH_FEATURES
from src.splits import rolling_origin_splits
from src.models import train_quantile_model
LO, HI, ALPHA = 0.1, 0.9, 0.2

cfg = load_config(); day = build_dataset(cfg); day = day[day.is_day].reset_index(drop=True)
X, _ = make_features(day, feature_cols=RICH_FEATURES); yn = day["residual_norm"]
med = day["disp_mw"].median()
v = cfg["validation"]; folds, _ = rolling_origin_splits(day, v["n_folds"], v["embargo_days"], v["sealed_test_frac"])

for fi, (tr, va) in enumerate(folds, 1):
    s = tr[np.argsort(day.loc[tr, "issued_at"].values)]
    n = len(s); fit, es, cal = s[:int(.70*n)], s[int(.70*n):int(.85*n)], s[int(.85*n):]
    fc = lambda i: day.loc[i, "fc_mw"].values; cap = lambda i: day.loc[i, "cap_mw"].values
    def qp(m, i): return fc(i) + m.predict(X.iloc[i]) * cap(i)
    u = lambda i: day.loc[i, "disp_mw"].fillna(med).clip(lower=med*0.2).values
    m_lo = train_quantile_model(X.iloc[fit], yn.iloc[fit], LO, X.iloc[es], yn.iloc[es])
    m_hi = train_quantile_model(X.iloc[fit], yn.iloc[fit], HI, X.iloc[es], yn.iloc[es])
    y_cal = day.loc[cal, "actual_mw"].values
    lo_c, hi_c = qp(m_lo, cal), qp(m_hi, cal)
    E = np.maximum(lo_c - y_cal, y_cal - hi_c) / u(cal)
    level = min(1.0, np.ceil((len(E)+1)*(1-ALPHA))/len(E)); Q = np.quantile(E, level)
    cal_cov = np.mean((y_cal >= lo_c) & (y_cal <= hi_c)) * 100
    print(f"fold {fi}: Q = {Q:+.3f}  ({'NARROWS' if Q < 0 else 'widens'}) | cal-slice base coverage {cal_cov:.1f}%")