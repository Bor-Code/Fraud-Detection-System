import pandas as pd
from src.features import build_features

def test_build_features() -> None:
    df = pd.DataFrame({"Time": [3600, 7200], "Amount": [10.0, 100.0]})
    res = build_features(df)
    assert "Hour" in res.columns
    assert "Is_Night" in res.columns
    assert "Log_Amount" in res.columns
