import numpy as np
from typing import Tuple
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from src.config import REPORTS_DIR
import os

def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray, fp_cost: float = 1.0, fn_cost: float = 10.0) -> float:
    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold = 0.5
    min_cost = float("inf")
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        cost = (fp * fp_cost) + (fn * fn_cost)
        if cost < min_cost:
            min_cost = cost
            best_threshold = t
    return float(best_threshold)

def plot_precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray, filename: str = "pr_curve.png") -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision, marker=".")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    path = os.path.join(REPORTS_DIR, filename)
    plt.savefig(path)
    plt.close()
