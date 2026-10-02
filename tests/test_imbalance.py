import pandas as pd
from src.imbalance import apply_smote

def test_apply_smote() -> None:
    X = pd.DataFrame({"F1": [1, 2, 3, 4], "F2": [1, 2, 3, 4]})
    y = pd.Series([0, 0, 0, 1])
    X_res, y_res = apply_smote(X, y)
    assert y_res.value_counts()[0] == y_res.value_counts()[1]
