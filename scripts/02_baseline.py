from src.data import load_config, build_dataset
from src.splits import rolling_origin_splits
from src.metrics import rmse


def main():
    cfg = load_config()
    day = build_dataset(cfg)
    day = day[day["is_day"]].reset_index(drop=True)

    v = cfg["validation"]
    folds, test_idx = rolling_origin_splits(day, v["n_folds"], v["embargo_days"], v["sealed_test_frac"])

    print("walk-forward eval rows:", len(test_idx),
          "| window:", day.loc[test_idx, "issued_at"].min().date(),
          "->", day.loc[test_idx, "issued_at"].max().date())
    print("\nrolling folds (train on past → validate on next block):")
    for i, (tr, va) in enumerate(folds, 1):
        base = rmse(day.loc[va, "actual_mw"], day.loc[va, "fc_mw"])
        print(f"  fold {i}: train={len(tr):6d}  val={len(va):5d}"
              f" | val {day.loc[va,'issued_at'].min().date()} → {day.loc[va,'issued_at'].max().date()}"
              f" | baseline RMSE={base:7.1f}")


if __name__ == "__main__":
    main()
