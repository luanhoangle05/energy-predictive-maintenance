"""Validation helpers for equipment sensor data."""

import numpy as np
import pandas as pd

from src.data.load_data import CMAPSS_COLUMNS


def validate_not_empty(data: pd.DataFrame) -> None:
    """Raise a clear error when a loaded dataset contains no rows."""
    if data.empty:
        raise ValueError("The dataset is empty.")


def validate_required_columns(data: pd.DataFrame) -> None:
    """Raise a clear error when required C-MAPSS columns are missing."""
    missing_columns = sorted(set(CMAPSS_COLUMNS) - set(data.columns))

    if missing_columns:
        raise ValueError(f"Required columns are missing: {missing_columns}")


def validate_no_missing_values(data: pd.DataFrame) -> None:
    """Raise an error when the dataset contains missing values."""
    missing_value_count = data.isna().sum().sum()

    if missing_value_count > 0:
        raise ValueError(
            f"The dataset contains {missing_value_count} missing value(s)."
        )


def validate_no_infinite_values(data: pd.DataFrame) -> None:
    """Raise an error when numeric columns contain infinite values."""
    numeric_data = data.select_dtypes(include="number")
    infinite_value_count = np.isinf(numeric_data.to_numpy()).sum()

    if infinite_value_count > 0:
        raise ValueError(
            f"The dataset contains {infinite_value_count} "
            "infinite numeric value(s)."
        )


def validate_unique_engine_cycles(data: pd.DataFrame) -> None:
    """Raise an error when an engine-cycle combination is duplicated."""
    duplicate_count = data.duplicated(
        subset=["unit_number", "time_in_cycles"]
    ).sum()

    if duplicate_count > 0:
        raise ValueError(
            f"The dataset contains {duplicate_count} "
            "duplicate engine-cycle record(s)."
        )


def validate_cycle_sequence(data: pd.DataFrame) -> None:
    """Raise an error when engine cycle sequences are invalid."""
    starting_cycles = data.groupby("unit_number")["time_in_cycles"].min()

    cycle_differences = (
        data.groupby("unit_number")["time_in_cycles"]
        .diff()
        .dropna()
    )

    invalid_start_count = (starting_cycles != 1).sum()
    invalid_step_count = (cycle_differences != 1).sum()

    if invalid_start_count > 0 or invalid_step_count > 0:
        raise ValueError(
            f"Invalid cycle sequences: {invalid_start_count} engine(s) "
            f"do not start at cycle 1; {invalid_step_count} "
            "invalid cycle step(s)."
        )


def validate_cmapss_data(data: pd.DataFrame) -> None:
    """Run all validation checks on a C-MAPSS dataset."""
    validate_not_empty(data)
    validate_required_columns(data)
    validate_no_missing_values(data)
    validate_no_infinite_values(data)
    validate_unique_engine_cycles(data)
    validate_cycle_sequence(data)
