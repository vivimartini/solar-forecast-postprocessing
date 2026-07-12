# tests/test_metrics.py
import numpy as np
from src.metrics import rmse, skill_score, pinball_loss, coverage


def test_rmse_zero_on_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0.0


def test_pinball_zero_on_perfect():
    y = np.array([1.0, 2.0, 3.0])
    for tau in (0.1, 0.5, 0.9):
        assert pinball_loss(y, y, tau) == 0.0


def test_pinball_asymmetry():
    # At tau=0.9, under-prediction should hurt 9x more than over-prediction.
    y = np.array([10.0])
    under = pinball_loss(y, np.array([8.0]), 0.9)    # guessed too low
    over = pinball_loss(y, np.array([12.0]), 0.9)    # guessed too high
    assert under > over
    assert np.isclose(under, 9 * over)


def test_skill_positive_when_better():
    assert np.isclose(skill_score(80.0, 100.0), 0.2)  # 20% less error


def test_coverage_counts_inside():
    y = np.array([1.0, 5.0, 9.0])
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([10.0, 4.0, 10.0])
    assert coverage(y, lo, hi) == 2 / 3              # the 5 is outside [0,4]