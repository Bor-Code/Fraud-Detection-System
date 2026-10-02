import pandas as pd
from typing import Dict, Any, List
from src.models import MODEL_REGISTRY
from src.evaluate import evaluate_predictions, get_confusion_matrix
from src.tracking import log_experiment
from src.imbalance import apply_smote, apply_undersampling
from src.report import save_comparison_report

def train_single_model(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, model_name: str, **kwargs: Any) -> Dict[str, Any]:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    model = MODEL_REGISTRY[model_name](**kwargs)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_val)[:, 1]
    else:
        y_prob = model.decision_function(X_val)
    metrics = evaluate_predictions(y_val, y_pred, y_prob)
    cm = get_confusion_matrix(y_val, y_pred)
    log_experiment(run_name=model_name, params=kwargs, metrics=metrics, model=model)
    return {"model": model, "metrics": metrics, "confusion_matrix": cm}

def run_all_experiments(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> List[Dict[str, Any]]:
    results = []
    samplers = {
        "none": lambda X, y: (X, y),
        "smote": apply_smote,
        "undersample": apply_undersampling
    }
    for model_name in MODEL_REGISTRY.keys():
        for sampler_name, sampler_func in samplers.items():
            X_res, y_res = sampler_func(X_train, y_train)
            res = train_single_model(X_res, y_res, X_val, y_val, model_name)
            metrics = res["metrics"].copy()
            metrics["model"] = model_name
            metrics["sampler"] = sampler_name
            results.append(metrics)
    save_comparison_report(results)
    return results
