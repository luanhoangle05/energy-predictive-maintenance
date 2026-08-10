"""Build model-ready features from validated sensor data."""

import pandas as pd

from src.data.load_data import RUL_COLUMN

NON_FEATURE_COLUMNS = [
    "unit_number",
    RUL_COLUMN
]

def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Return model inputs without engine identifiers or RUL targets."""
    columns_to_remove = []

    for column in NON_FEATURE_COLUMNS:
        if column in data.columns:
            columns_to_remove.append(column)

    return data.drop(columns=columns_to_remove).copy()
