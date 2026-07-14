# scripts/metric_b.py
"""Metric (b): predict how the NEXT cycle will revise the forecast.
Direction turned out to be a coin flip (checked below), so the useful thing to predict
is the SIZE of the revision -- and this is where dispersion finally pays off after
being useless for the point model and the intervals.
Run: PYTHONPATH=. python scripts/metric_b.py
"""
import numpy as np
from scipy.stats import spearmanr
from src.data import load_config, build_dataset
from src.features import make_features, BASE_FEATURES
from src.splits import rolling_origin_splits
from src.models import train_residual_model


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


if __name__ == "__main__":
    main()