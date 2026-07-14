# scripts/11_features.py
"""Does richer feature engineering beat the base set? Fleet-normalized target.
Run: PYTHONPATH=. python scripts/11_features.py
"""
import numpy as np
from src.data import load_config, build_dataset
from src.features import make_features, BASE_FEATURES, RICH_FEATURES
from src.splits import rolling_origin_splits
from src.models import train_residual_model
from src.metrics import rmse, skill_score


def evaluate(day, folds, cols):
    X, _ = make_features(day, feature_cols=cols)
    yn = day["residual_norm"]
    skills = []
    for tr, va in folds:
        s = tr[np.argsort(day.loc[tr, "issued_at"].values)]
        cut = int(0.85 * len(s)); tr_in, val_in = s[:cut], s[cut:]
        m = train_residual_model(X.iloc[tr_in], yn.iloc[tr_in], X.iloc[val_in], yn.iloc[val_in])
        pred = m.predict(X.iloc[va]) * day.loc[va, "cap_mw"].values
        corrected = np.clip(day.loc[va, "fc_mw"].values + pred, 0, None)
        actual = day.loc[va, "actual_mw"].values
        skills.append(skill_score(rmse(actual, corrected), rmse(actual, day.loc[va, "fc_mw"].values)))
    return np.array(skills)


def main():
    cfg = load_config()
    day = build_dataset(cfg); day = day[day["is_day"]].reset_index(drop=True)
    v = cfg["validation"]
    folds, _ = rolling_origin_splits(day, v["n_folds"], v["embargo_days"], v["sealed_test_frac"])
    for name, cols in [("base", BASE_FEATURES), ("rich (shape+clearsky)", RICH_FEATURES)]:
        s = evaluate(day, folds, cols)
        print(f"{name:22s}: mean skill {s.mean()*100:+.2f}%  (folds {np.round(s*100,1)})")


if __name__ == "__main__":
    main()