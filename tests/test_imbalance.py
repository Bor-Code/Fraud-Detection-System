import pandas as pd
from src.imbalance import apply_smote

def test_apply_smote() -> None:
    X = pd.DataFrame({"F1": range(20), "F2": range(20)})
    y = pd.Series([0]*14 + [1]*6)
    X_res, y_res = apply_smote(X, y)
    assert y_res.value_counts()[0] == y_res.value_counts()[1]
