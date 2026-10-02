import mlflow
from typing import Dict, Any

def log_experiment(run_name: str, params: Dict[str, Any], metrics: Dict[str, float], model: Any) -> None:
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
