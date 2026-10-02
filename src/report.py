import pandas as pd
from typing import List, Dict, Any
from src.config import REPORTS_DIR
import os

def save_comparison_report(results: List[Dict[str, Any]], filename: str = "model_comparison.csv") -> None:
    df = pd.DataFrame(results)
    path = os.path.join(REPORTS_DIR, filename)
    df.to_csv(path, index=False)
