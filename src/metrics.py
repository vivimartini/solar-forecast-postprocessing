import numpy as np


def rmse(y_true, y_pred):
    e = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(e ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def skill_score(rmse_model, rmse_baseline):
    return 1.0 - rmse_model / rmse_baseline


def pinball_loss(y_true, q_pred, tau):
    diff = np.asarray(y_true) - np.asarray(q_pred)
    return float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))


def coverage(y_true, lower, upper):
    y = np.asarray(y_true)
    return float(np.mean((y >= np.asarray(lower)) & (y <= np.asarray(upper))))


def interval_width(lower, upper):
    return float(np.mean(np.asarray(upper) - np.asarray(lower)))
