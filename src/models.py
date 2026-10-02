from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from typing import Any
from src.config import SEED

def get_logistic_regression(**kwargs: Any) -> LogisticRegression:
    params = {"random_state": SEED, "max_iter": 1000}
    params.update(kwargs)
    return LogisticRegression(**params)

def get_random_forest(**kwargs: Any) -> RandomForestClassifier:
    params = {"random_state": SEED, "n_estimators": 100}
    params.update(kwargs)
    return RandomForestClassifier(**params)
