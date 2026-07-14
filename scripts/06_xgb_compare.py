import numpy as np
from src.data import load_config, build_dataset
from src.features import make_features
from src.splits import rolling_origin_splits
from src.models import train_residual_model, train_residual_model_xgb
from src.metrics import rmse, skill_score

TRAINERS = {"LightGBM": train_residual_model, "XGBoost": train_residual_model_xgb}


def run(trainer, day, X, y, folds):
    skills = []
    for tr, va in folds:
        tr_sorted = tr[np.argsort(day.loc[tr, "issued_at"].values)]
        cut = int(len(tr_sorted) * 0.85)
        tr_in, val_in = tr_sorted[:cut], tr_sorted[cut:]
        model = trainer(X.iloc[tr_in], y.iloc[tr_in], X.iloc[val_in], y.iloc[val_in])
        pred = model.predict(X.iloc[va])
        corrected = np.clip(day.loc[va, "fc_mw"].values + pred, 0, None)
        actual = day.loc[va, "actual_mw"].values
        skills.append(skill_score(rmse(actual, corrected), rmse(actual, day.loc[va, "fc_mw"].values)))
    return np.array(skills)


def main():
    cfg = load_config()
    day = build_dataset(cfg)
    day = day[day["is_day"]].reset_index(drop=True)
    X, y = make_features(day)

    v = cfg["validation"]
    folds, _ = rolling_origin_splits(day, v["n_folds"], v["embargo_days"], v["sealed_test_frac"])

    print("Layer-1 residual correction — point skill vs raw forecast (higher = better)\n")
    for name, trainer in TRAINERS.items():
        s = run(trainer, day, X, y, folds)
        print(f"{name:9s}: mean skill {s.mean()*100:+.2f}%  (std {s.std()*100:.1f}%, "
              f"positive {int((s > 0).sum())}/{len(s)}, folds {np.round(s*100, 1)})")


if __name__ == "__main__":
    main()
