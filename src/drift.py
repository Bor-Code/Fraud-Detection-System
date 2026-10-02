import pandas as pd
import numpy as np
from typing import Dict, Any
from scipy.stats import ks_2samp

def calculate_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    breakpoints = np.arange(0, buckets + 1) / buckets * 100
    expected_perc = np.percentile(expected, breakpoints)
    expected_counts = np.histogram(expected, expected_perc)[0]
    actual_counts = np.histogram(actual, expected_perc)[0]
    expected_counts = np.maximum(expected_counts, 1)
    actual_counts = np.maximum(actual_counts, 1)
    expected_frac = expected_counts / sum(expected_counts)
    actual_frac = actual_counts / sum(actual_counts)
    psi = np.sum((actual_frac - expected_frac) * np.log(actual_frac / expected_frac))
    return float(psi)

def detect_drift(df_train: pd.DataFrame, df_new: pd.DataFrame, features: list[str]) -> Dict[str, Any]:
    report = {}
    for f in features:
        ks_stat, p_value = ks_2samp(df_train[f], df_new[f])
        psi_val = calculate_psi(df_train[f].values, df_new[f].values)
        report[f] = {
            "ks_statistic": float(ks_stat),
            "p_value": float(p_value),
            "psi": float(psi_val),
            "drift_detected": bool(p_value < 0.05 or psi_val > 0.2)
        }
    return report
