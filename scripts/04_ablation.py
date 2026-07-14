import numpy as np
from src.data import load_config, build_dataset
from src.features import make_features, BASE_FEATURES
from src.splits import rolling_origin_splits
from src.models import train_residual_model
from src.metrics import rmse, skill_score


def run(day, X, y, folds):
    skills = []
    for tr, va in folds:
        tr_sorted = tr[np.argsort(day.loc[tr, "issued_at"].values)]
        cut = int(len(tr_sorted) * 0.85)
        tr_in, val_in = tr_sorted[:cut], tr_sorted[cut:]
        model = train_residual_model(X.iloc[tr_in], y.iloc[tr_in], X.iloc[val_in], y.iloc[val_in])
        pred = model.predict(X.iloc[va])
        corrected = np.clip(day.loc[va, "fc_mw"].values + pred, 0, None)
        actual = day.loc[va, "actual_mw"].values
        skills.append(skill_score(rmse(actual, corrected), rmse(actual, day.loc[va, "fc_mw"].values)))
    return np.array(skills)


def main():
    cfg = load_config()
    day = build_dataset(cfg)
    day = day[day["is_day"]].reset_index(drop=True)
    v = cfg["validation"]
    folds, _ = rolling_origin_splits(day, v["n_folds"], v["embargo_days"], v["sealed_test_frac"])

    for name, cols in {"base": BASE_FEATURES,
                       "base + dispersion": BASE_FEATURES + ["disp_mw"]}.items():
        X, y = make_features(day, feature_cols=cols)
        s = run(day, X, y, folds)
        print(f"{name:18s}: mean skill {s.mean()*100:+.2f}%  (folds {np.round(s*100,1)})")


if __name__ == "__main__":
    main()
