# scripts/15_debias_gbdt.py
"""GBDT on the de-biased residual: does a static model help AFTER online bias?
Target = (actual - fc - online_bias). If bias drift was the main problem, what's left
might be stationary enough for a GBDT to pick up ramp/season structure.
Run: PYTHONPATH=. python scripts/15_debias_gbdt.py
"""
import numpy as np
from src.data import load_config, build_dataset
from src.features import make_features, RICH_FEATURES
from src.splits import rolling_origin_splits, train_test_seal
from src.models import train_residual_model
from src.online_bias import add_online_bias
from src.metrics import rmse, skill_score


def evaluate_fold(day, tr, va, X, y_debias, bias):
    s = tr[np.argsort(day.loc[tr, "issued_at"].values)]
    cut = int(0.85 * len(s))
    tr_in, val_in = s[:cut], s[cut:]
    m = train_residual_model(X.iloc[tr_in], y_debias.iloc[tr_in], X.iloc[val_in], y_debias.iloc[val_in])
    fc = day.loc[va, "fc_mw"].values
    actual = day.loc[va, "actual_mw"].values
    b = bias.iloc[va].values
    pred = m.predict(X.iloc[va])
    base_rmse = rmse(actual, fc)
    return {
        "bias_skill": skill_score(rmse(actual, np.clip(fc + b, 0, None)), base_rmse) * 100,
        "combined_skill": skill_score(rmse(actual, np.clip(fc + b + pred, 0, None)), base_rmse) * 100,
    }


def evaluate_sealed(day, X, y_debias, bias):
    v = load_config()["validation"]
    dev_mask, test_mask = train_test_seal(day, v["sealed_test_frac"])
    tr = np.where(dev_mask)[0]
    va = np.where(test_mask)[0]
    s = tr[np.argsort(day.loc[tr, "issued_at"].values)]
    cut = int(0.85 * len(s))
    tr_in, val_in = s[:cut], s[cut:]
    m = train_residual_model(X.iloc[tr_in], y_debias.iloc[tr_in], X.iloc[val_in], y_debias.iloc[val_in])
    fc = day.loc[va, "fc_mw"].values
    actual = day.loc[va, "actual_mw"].values
    b = bias.iloc[va].values
    pred = m.predict(X.iloc[va])
    base_rmse = rmse(actual, fc)
    return {
        "bias_skill": skill_score(rmse(actual, np.clip(fc + b, 0, None)), base_rmse) * 100,
        "combined_skill": skill_score(rmse(actual, np.clip(fc + b + pred, 0, None)), base_rmse) * 100,
    }


def main():
    cfg = load_config()
    day = build_dataset(cfg)
    day = day[day["is_day"]].reset_index(drop=True)
    ob = cfg["online_bias"]
    day = add_online_bias(day, window_days=ob["window_days"], per_hour=ob["per_hour"])
    day["residual_debias"] = day["residual_mw"] - day["online_bias"]

    X, _ = make_features(day, feature_cols=RICH_FEATURES)
    y = day["residual_debias"]
    bias = day["online_bias"]

    v = cfg["validation"]
    folds, _ = rolling_origin_splits(day, v["n_folds"], v["embargo_days"], v["sealed_test_frac"])

    print("point skill vs raw forecast (higher = better)\n")
    print(f"{'split':12s}  {'online bias':>12s}  {'bias + GBDT':>12s}")
    for i, (tr, va) in enumerate(folds):
        r = evaluate_fold(day, tr, va, X, y, bias)
        print(f"{'fold ' + str(i + 1):12s}  {r['bias_skill']:+11.2f}%  {r['combined_skill']:+11.2f}%")

    r = evaluate_sealed(day, X, y, bias)
    print(f"{'sealed test':12s}  {r['bias_skill']:+11.2f}%  {r['combined_skill']:+11.2f}%")


if __name__ == "__main__":
    main()
