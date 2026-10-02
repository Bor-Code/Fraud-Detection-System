import pandas as pd
from typing import Tuple
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    def __init__(self) -> None:
        self.scaler = StandardScaler()

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        X_scaled = X_train.copy()
        X_scaled[["Time", "Amount"]] = self.scaler.fit_transform(X_train[["Time", "Amount"]])
        return X_scaled

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_scaled = X.copy()
        X_scaled[["Time", "Amount"]] = self.scaler.transform(X[["Time", "Amount"]])
        return X_scaled
