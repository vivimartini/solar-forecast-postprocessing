# src/metrics.py
"""Scoring functions. Kept boring on purpose; tests in tests/test_metrics.py."""
import numpy as np


def rmse(y_true, y_pred):
    e = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(e ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def skill_score(rmse_model, rmse_baseline):
    """Fraction of the baseline's error removed. >0 means better than baseline."""
    return 1.0 - rmse_model / rmse_baseline


def pinball_loss(y_true, q_pred, tau):
    """Pinball (quantile) loss at level tau. Lower is better."""
    diff = np.asarray(y_true) - np.asarray(q_pred)
    return float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))


def coverage(y_true, lower, upper):
    """Fraction of actuals inside [lower, upper]. For an 80% interval this should be ~0.8."""
    y = np.asarray(y_true)
    return float(np.mean((y >= np.asarray(lower)) & (y <= np.asarray(upper))))


def interval_width(lower, upper):
    return float(np.mean(np.asarray(upper) - np.asarray(lower)))