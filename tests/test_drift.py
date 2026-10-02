import pandas as pd
import numpy as np
from src.drift import calculate_psi

def test_calculate_psi() -> None:
    expected = np.random.normal(0, 1, 1000)
    actual = np.random.normal(0, 1, 1000)
    psi = calculate_psi(expected, actual)
    assert psi < 0.2
