import joblib
from typing import Any, Tuple
from src.config import MODEL_PATH, SCALER_PATH, THRESHOLD_PATH

def save_artifacts(model: Any, scaler: Any, threshold: float) -> None:
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(threshold, THRESHOLD_PATH)

def load_artifacts() -> Tuple[Any, Any, float]:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    threshold = joblib.load(THRESHOLD_PATH)
    return model, scaler, threshold
