import pandas as pd
from src.models import MODEL_REGISTRY

def test_model_registry() -> None:
    assert "lr" in MODEL_REGISTRY
    assert "rf" in MODEL_REGISTRY
    assert "xgb" in MODEL_REGISTRY
    assert "lgbm" in MODEL_REGISTRY
    assert "mlp" in MODEL_REGISTRY

def test_model_instantiation() -> None:
    model = MODEL_REGISTRY["lr"]()
    assert model is not None
