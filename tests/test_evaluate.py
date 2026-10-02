import numpy as np
from src.evaluate import evaluate_predictions, get_confusion_matrix

def test_evaluate_predictions() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_prob = np.array([0.1, 0.6, 0.8, 0.9])
    res = evaluate_predictions(y_true, y_pred, y_prob)
    assert "precision" in res
    assert "recall" in res
    assert "pr_auc" in res

def test_get_confusion_matrix() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    cm = get_confusion_matrix(y_true, y_pred)
    assert cm["tn"] == 1
    assert cm["fp"] == 1
