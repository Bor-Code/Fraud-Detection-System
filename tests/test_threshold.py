import numpy as np
from src.threshold import find_optimal_threshold

def test_find_optimal_threshold() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.6, 0.9])
    t = find_optimal_threshold(y_true, y_prob)
    assert 0.0 < t < 1.0
