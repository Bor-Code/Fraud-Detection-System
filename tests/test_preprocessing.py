import pandas as pd
from src.preprocessing import Preprocessor

def test_preprocessor() -> None:
    df = pd.DataFrame({"Time": [0.0, 10.0], "Amount": [100.0, 200.0], "Class": [0, 1]})
    p = Preprocessor()
    scaled = p.fit_transform(df)
    assert scaled["Time"].mean() < 1e-10
    assert scaled["Amount"].mean() < 1e-10
    test_scaled = p.transform(df)
    assert test_scaled["Time"].mean() < 1e-10
