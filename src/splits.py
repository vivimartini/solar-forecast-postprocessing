# src/splits.py
"""Leak-safe time splits.

- Split by issued_at (when the forecast became usable), never by valid time.
- Seal the most recent `sealed_test_frac` of time as an untouched final test set.
- Rolling-origin CV on the rest: expanding training window; each validation fold
  is the next contiguous time block.
- Embargo: drop training rows within `embargo_days` before a fold's validation
  start, so backward-looking features (lagged-ensemble spread) can't peek across.
"""
import numpy as np
import pandas as pd


def train_test_seal(df, sealed_test_frac, time_col="issued_at"):
    """(dev_mask, test_mask): the sealed final test is the most recent time slice."""
    t = df[time_col]
    cutoff = t.min() + (t.max() - t.min()) * (1 - sealed_test_frac)
    return (t < cutoff).values, (t >= cutoff).values


def rolling_origin_splits(df, n_folds, embargo_days, sealed_test_frac, time_col="issued_at"):
    """Return (list of (train_idx, val_idx)) for each rolling fold, plus sealed test_idx."""
    t = df[time_col]
    dev_mask, test_mask = train_test_seal(df, sealed_test_frac, time_col)
    lo, hi = t[dev_mask].min(), t[dev_mask].max()
    edges = [lo + (hi - lo) * k / (n_folds + 1) for k in range(n_folds + 2)]
    embargo = pd.Timedelta(days=embargo_days)

    folds = []
    for i in range(1, n_folds + 1):
        val_start, val_end = edges[i], edges[i + 1]
        val = (t >= val_start) & (t < val_end)
        train = t < (val_start - embargo)          # expanding window, minus the embargo gap
        folds.append((np.where(train.values)[0], np.where(val.values)[0]))
    return folds, np.where(test_mask)[0]