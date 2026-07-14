# scripts/metric_b.py
"""Metric (b): predict how the NEXT cycle will revise the forecast.
Direction turned out to be a coin flip (checked below), so the useful thing to predict
is the SIZE of the revision -- and this is where dispersion finally pays off.
Also fits P10/P50/P90 quantiles on next_revision for the probabilistic half of metric (b).
Run: PYTHONPATH=. python scripts/metric_b.py
"""
import numpy as np
from scipy.stats import spearmanr
from src.data import load_config, build_dataset
from src.features import make_features, BASE_FEATURES
from src.splits import rolling_origin_splits
from src.models import train_residual_model, train_quantile_model
from src.metrics import pinball_loss, coverage

QUANTILES = [0.1, 0.5, 0.9]


def main():
    cfg = load_config()
    day = build_dataset(cfg)
    day = day[day["is_day"]].dropna(subset=["next_revision", "disp_mw"]).reset_index(drop=True)
    day["abs_rev"] = day["next_revision"].abs()

    # (1) does disagreement between past runs predict how much the next run will move?
    rho = spearmanr(day["disp_mw"], day["abs_rev"]).correlation
    print(f"Spearman(dispersion, |next revision|) = {rho:.3f}")

    # (2) model |next revision| with and without dispersion, same harness as the point model
    v = cfg["validation"]
    folds, _ = rolling_origin_splits(day, v["n_folds"], v["embargo_days"], v["sealed_test_frac"])
    y = day["abs_rev"]
    for name, cols in [("base", BASE_FEATURES), ("base + dispersion", BASE_FEATURES + ["disp_mw"])]:
        X, _ = make_features(day, feature_cols=cols)
        maes = []
        for tr, va in folds:
            s = tr[np.argsort(day.loc[tr, "issued_at"].values)]
            cut = int(0.85 * len(s)); tr_in, val_in = s[:cut], s[cut:]
            m = train_residual_model(X.iloc[tr_in], y.iloc[tr_in], X.iloc[val_in], y.iloc[val_in])
            maes.append(np.mean(np.abs(m.predict(X.iloc[va]) - y.iloc[va].values)))
        print(f"{name:18s}: MAE predicting |next revision| = {np.mean(maes):.1f} MW")

    # (3) direction check -- if this is ~50% there's nothing to model there
    print(f"\ndirection base rate: {(day.next_revision > 0).mean()*100:.1f}% of revisions are UP "
          f"(near 50% => which-way is ~unpredictable; magnitude is the useful signal)")

    # (4) revision quantiles -- probabilistic framing for metric (b)
    print("\n=== revision quantiles (P10/P50/P90 of next_revision, MW) ===")
    y_rev = day["next_revision"]
    for name, cols in [("base", BASE_FEATURES), ("base + dispersion", BASE_FEATURES + ["disp_mw"])]:
        X, _ = make_features(day, feature_cols=cols)
        pbs, covs, widths = [], [], []
        for tr, va in folds:
            s = tr[np.argsort(day.loc[tr, "issued_at"].values)]
            cut = int(0.85 * len(s)); tr_in, val_in = s[:cut], s[cut:]
            preds = []
            for tau in QUANTILES:
                m = train_quantile_model(X.iloc[tr_in], y_rev.iloc[tr_in], tau,
                                         X.iloc[val_in], y_rev.iloc[val_in])
                preds.append(m.predict(X.iloc[va]))
            Q = np.sort(np.column_stack(preds), axis=1)
            y_va = y_rev.iloc[va].values
            pbs.append(np.mean([pinball_loss(y_va, Q[:, j], QUANTILES[j]) for j in range(3)]))
            covs.append(coverage(y_va, Q[:, 0], Q[:, 2]))
            widths.append(np.mean(Q[:, 2] - Q[:, 0]))
        print(f"{name:18s}: pinball {np.mean(pbs):6.1f} | P10-90 coverage {np.mean(covs)*100:5.1f}%"
              f" | width {np.mean(widths):6.1f} MW")


if __name__ == "__main__":
    main()