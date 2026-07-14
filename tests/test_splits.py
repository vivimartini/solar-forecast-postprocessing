"""Leak-safety checks on the time-split harness."""
import numpy as np
import pandas as pd

from src.splits import rolling_origin_splits, train_test_seal


def _tiny_day(n=100):
    t = pd.date_range("2023-01-01", periods=n, freq="6h")
    return pd.DataFrame({"issued_at": t, "step": t + pd.Timedelta(hours=1), "x": np.arange(n)})


def test_sealed_test_is_most_recent():
    df = _tiny_day(50)
    dev, test = train_test_seal(df, sealed_test_frac=0.2)
    assert df.loc[test, "issued_at"].min() >= df.loc[dev, "issued_at"].max()


def test_folds_do_not_overlap():
    df = _tiny_day(200)
    folds, test_idx = rolling_origin_splits(df, n_folds=3, embargo_days=1, sealed_test_frac=0.2)
    test_set = set(test_idx)
    val_sets = []
    for tr, va in folds:
        assert not (set(tr) & test_set)
        assert not (set(va) & test_set)
        val_sets.append(set(va))
    for i in range(len(val_sets)):
        for j in range(i + 1, len(val_sets)):
            assert not (val_sets[i] & val_sets[j])


def test_train_precedes_val_with_embargo():
    df = _tiny_day(300)
    folds, _ = rolling_origin_splits(df, n_folds=4, embargo_days=2, sealed_test_frac=0.2)
    for tr, va in folds:
        assert df.loc[tr, "issued_at"].max() < df.loc[va, "issued_at"].min()
