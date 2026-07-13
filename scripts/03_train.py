# scripts/03_train.py
"""Layer 1 — residual GBDT correction, evaluated on rolling-origin folds.
Run: PYTHONPATH=. python scripts/03_train.py
"""
import numpy as np
from src.data import load_config, build_dataset
from src.features import make_features
from src.splits import rolling_origin_splits
from src.models import train_residual_model
from src.metrics import rmse, skill_score


def main():
    cfg = load_config()
    day = build_dataset(cfg)
    day = day[day["is_day"]].reset_index(drop=True)
    X, y = make_features(day)                       # y = residual (actual - forecast)

    v = cfg["validation"]
    folds, _ = rolling_origin_splits(day, v["n_folds"], v["embargo_days"], v["sealed_test_frac"])

    skills = []
    print("fold | baseline RMSE | model RMSE |  skill")
    for i, (tr, va) in enumerate(folds, 1):
        model = train_residual_model(X.iloc[tr], y.iloc[tr])
        pred_resid = model.predict(X.iloc[va])
        corrected = np.clip(day.loc[va, "fc_mw"].values + pred_resid, 0, None)  # non-negativity

        actual = day.loc[va, "actual_mw"].values
        base = rmse(actual, day.loc[va, "fc_mw"].values)
        mod = rmse(actual, corrected)
        s = skill_score(mod, base)
        skills.append(s)
        print(f"  {i}  |   {base:8.1f}   |  {mod:8.1f}  | {s*100:+5.1f}%")

    skills = np.array(skills)
    print(f"\nmean skill: {skills.mean()*100:+.1f}%  (std {skills.std()*100:.1f}%, "
          f"positive in {(skills > 0).sum()}/{len(skills)} folds)")


if __name__ == "__main__":
    main()
