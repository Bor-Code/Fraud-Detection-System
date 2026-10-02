import pandas as pd
import numpy as np

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    df_out["Hour"] = (df_out["Time"] // 3600) % 24
    df_out["Is_Night"] = df_out["Hour"].apply(lambda x: 1 if x < 6 else 0)
    return df_out

def add_log_amount(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    df_out["Log_Amount"] = np.log1p(df_out["Amount"])
    return df_out

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df_out = add_time_features(df)
    df_out = add_log_amount(df_out)
    return df_out
