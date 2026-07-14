import numpy as np
from src.data import load_config, build_dataset
from src.features import make_features, BASE_FEATURES
from src.splits import rolling_origin_splits
from src.models import train_quantile_model
from src.metrics import pinball_loss, coverage, interval_width

QUANTILES = [0.1, 0.5, 0.9]


def run(day, cols, folds):
    X, y = make_features(day, feature_cols=cols)
    pbs, covs, widths = [], [], []
    for tr, va in folds:
        tr_sorted = tr[np.argsort(day.loc[tr, "issued_at"].values)]
        cut = int(len(tr_sorted) * 0.85)
        tr_in, val_in = tr_sorted[:cut], tr_sorted[cut:]
        fc_va = day.loc[va, "fc_mw"].values
        actual = day.loc[va, "actual_mw"].values

        cols_pred = []
        for tau in QUANTILES:
            m = train_quantile_model(X.iloc[tr_in], y.iloc[tr_in], tau, X.iloc[val_in], y.iloc[val_in])
            cols_pred.append(fc_va + m.predict(X.iloc[va]))
        Q = np.sort(np.clip(np.column_stack(cols_pred), 0, None), axis=1)
        p10, p50, p90 = Q[:, 0], Q[:, 1], Q[:, 2]

        pb = np.mean([pinball_loss(actual, Q[:, j], QUANTILES[j]) for j in range(3)])
        pbs.append(pb); covs.append(coverage(actual, p10, p90)); widths.append(interval_width(p10, p90))
    return np.array(pbs), np.array(covs), np.array(widths)


def main():
    cfg = load_config()
    day = build_dataset(cfg); day = day[day["is_day"]].reset_index(drop=True)
    v = cfg["validation"]
    folds, _ = rolling_origin_splits(day, v["n_folds"], v["embargo_days"], v["sealed_test_frac"])
    for name, cols in {"base": BASE_FEATURES, "base + dispersion": BASE_FEATURES + ["disp_mw"]}.items():
        pb, cov, w = run(day, cols, folds)
        print(f"{name:18s}: pinball {pb.mean():7.1f} | P10-90 coverage {cov.mean()*100:5.1f}% | width {w.mean():8.1f}")


if __name__ == "__main__":
    main()
