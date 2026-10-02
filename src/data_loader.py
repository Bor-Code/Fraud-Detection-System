import pandas as pd
import logging
from typing import Tuple
from src.config import DATA_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def load_data(file_path: str = DATA_PATH) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise

def validate_schema(df: pd.DataFrame) -> bool:
    required_cols = {"Time", "Amount", "Class"}
    v_cols = {f"V{i}" for i in range(1, 29)}
    expected_cols = required_cols.union(v_cols)
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    return True

def validate_missing_values(df: pd.DataFrame) -> bool:
    missing = df.isnull().sum().sum()
    if missing > 0:
        raise ValueError(f"Found {missing} missing values")
    return True

def log_class_distribution(df: pd.DataFrame) -> None:
    counts = df["Class"].value_counts()
    logger.info(f"Class distribution: {counts.to_dict()}")

def load_and_validate(file_path: str = DATA_PATH) -> pd.DataFrame:
    df = load_data(file_path)
    validate_schema(df)
    validate_missing_values(df)
    log_class_distribution(df)
    return df
