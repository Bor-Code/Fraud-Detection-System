from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
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

def get_xgboost(**kwargs: Any) -> XGBClassifier:
    params = {"random_state": SEED, "n_estimators": 100, "eval_metric": "logloss"}
    params.update(kwargs)
    return XGBClassifier(**params)

def get_lightgbm(**kwargs: Any) -> LGBMClassifier:
    params = {"random_state": SEED, "n_estimators": 100}
    params.update(kwargs)
    return LGBMClassifier(**params)
