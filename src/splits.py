"""Split on issued_at. Sealed tail + rolling-origin CV with embargo."""
import numpy as np
import pandas as pd


def train_test_seal(df, sealed_test_frac, time_col="issued_at"):
    t = df[time_col]
    cutoff = t.min() + (t.max() - t.min()) * (1 - sealed_test_frac)
    return (t < cutoff).values, (t >= cutoff).values


def rolling_origin_splits(df, n_folds, embargo_days, sealed_test_frac, time_col="issued_at"):
    t = df[time_col]
    dev_mask, test_mask = train_test_seal(df, sealed_test_frac, time_col)
    lo, hi = t[dev_mask].min(), t[dev_mask].max()
    edges = [lo + (hi - lo) * k / (n_folds + 1) for k in range(n_folds + 2)]
    embargo = pd.Timedelta(days=embargo_days)

    folds = []
    for i in range(1, n_folds + 1):
        val_start, val_end = edges[i], edges[i + 1]
        val = (t >= val_start) & (t < val_end)
        train = t < (val_start - embargo)  # gap so backward-looking features don't peek
        folds.append((np.where(train.values)[0], np.where(val.values)[0]))
    return folds, np.where(test_mask)[0]
