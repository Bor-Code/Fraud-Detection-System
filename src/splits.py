import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from src.config import SEED

def split_data(
    df: pd.DataFrame, 
    target_col: str = "Class", 
    test_size: float = 0.2, 
    val_size: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=SEED
    )
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=SEED
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
