import numpy as np
from src.data import load_config, build_dataset
from src.features import make_features, BASE_FEATURES
from src.splits import rolling_origin_splits
from src.models import train_quantile_model
from src.metrics import pinball_loss, coverage, interval_width

LO, MID, HI, ALPHA = 0.1, 0.5, 0.9, 0.2


def run(day, folds, conformal):
    X, _ = make_features(day, feature_cols=BASE_FEATURES)
    yn = day["residual_norm"]
    pbs, covs, widths = [], [], []
    for tr, va in folds:
        s = tr[np.argsort(day.loc[tr, "issued_at"].values)]
        n = len(s); fit, es, cal = s[:int(.70*n)], s[int(.70*n):int(.85*n)], s[int(.85*n):]
        fc  = lambda idx: day.loc[idx, "fc_mw"].values
        cap = lambda idx: day.loc[idx, "cap_mw"].values
        def qp(m, idx): return fc(idx) + m.predict(X.iloc[idx]) * cap(idx)

        m_lo = train_quantile_model(X.iloc[fit], yn.iloc[fit], LO, X.iloc[es], yn.iloc[es])
        m_hi = train_quantile_model(X.iloc[fit], yn.iloc[fit], HI, X.iloc[es], yn.iloc[es])
        m_md = train_quantile_model(X.iloc[fit], yn.iloc[fit], MID, X.iloc[es], yn.iloc[es])

        Q = 0.0
        if conformal:
            y_cal = day.loc[cal, "actual_mw"].values
            E = np.maximum(qp(m_lo, cal) - y_cal, y_cal - qp(m_hi, cal))
            level = min(1.0, np.ceil((len(E)+1)*(1-ALPHA))/len(E))
            Q = np.quantile(E, level)

        y_va = day.loc[va, "actual_mw"].values
        p10 = np.clip(qp(m_lo, va) - Q, 0, None)
        p90 = np.clip(qp(m_hi, va) + Q, 0, None)
        p50 = np.clip(qp(m_md, va), 0, None)
        Qs = np.sort(np.column_stack([p10, p50, p90]), axis=1)
        pbs.append(np.mean([pinball_loss(y_va, Qs[:, j], [LO, MID, HI][j]) for j in range(3)]))
        covs.append(coverage(y_va, Qs[:, 0], Qs[:, 2])); widths.append(interval_width(Qs[:, 0], Qs[:, 2]))
    return np.array(pbs), np.array(covs), np.array(widths)


def main():
    cfg = load_config()
    day = build_dataset(cfg); day = day[day["is_day"]].reset_index(drop=True)
    v = cfg["validation"]
    folds, _ = rolling_origin_splits(day, v["n_folds"], v["embargo_days"], v["sealed_test_frac"])
    for name, conf in [("normalized, no conformal", False), ("normalized + conformal", True)]:
        pb, cov, w = run(day, folds, conf)
        print(f"{name:26s}: pinball {pb.mean():7.1f} | coverage {cov.mean()*100:5.1f}% | width {w.mean():8.1f}")


if __name__ == "__main__":
    main()
