import shap
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any
import os
from src.config import REPORTS_DIR

def plot_shap_summary(model: Any, X: pd.DataFrame, filename: str = "shap_summary.png") -> None:
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    path = os.path.join(REPORTS_DIR, filename)
    plt.savefig(path, bbox_inches="tight")
    plt.close()

def plot_shap_single(model: Any, X: pd.DataFrame, idx: int, filename: str = "shap_local.png") -> None:
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X.iloc[[idx]])
    plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    path = os.path.join(REPORTS_DIR, filename)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
