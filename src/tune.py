import optuna
import pandas as pd
from typing import Dict, Any
import mlflow
from src.models import MODEL_REGISTRY
from src.evaluate import evaluate_predictions
from src.config import SEED, MLFLOW_TRACKING_URI

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("fraud_detection")
def objective_xgb(trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3)
    }
    model = MODEL_REGISTRY["xgb"](**params)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)
    metrics = evaluate_predictions(y_val, y_pred, y_prob)
    return metrics["pr_auc"]

def tune_xgboost(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, n_trials: int = 20) -> Dict[str, Any]:
    study = optuna.create_study(direction="maximize")
    func = lambda trial: objective_xgb(trial, X_train, y_train, X_val, y_val)
    study.optimize(func, n_trials=n_trials)
    with mlflow.start_run(run_name="xgb_tuned"):
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_pr_auc", study.best_value)
    return study.best_params
