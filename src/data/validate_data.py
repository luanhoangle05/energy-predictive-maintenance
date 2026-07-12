"""Validation helpers for equipment sensor data."""

import pandas as pd


def validate_not_empty(data: pd.DataFrame) -> None:
    """Raise a clear error when a loaded dataset contains no rows."""
    if data.empty:
        raise ValueError("The dataset is empty.")
