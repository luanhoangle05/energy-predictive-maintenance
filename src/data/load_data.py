"""Load C-MAPSS source data from disk."""

from pathlib import Path

import pandas as pd


CMAPSS_COLUMNS = [
    "unit_number",
    "time_in_cycles",
    "operational_setting_1",
    "operational_setting_2",
    "operational_setting_3",
] + [f"sensor_{number}" for number in range(1, 22)]

RUL_COLUMN = "remaining_useful_life"


def load_cmapss_file(file_path: Path) -> pd.DataFrame:
    """Load one whitespace-delimited C-MAPSS file with standard column names."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"C-MAPSS file not found: {path}")

    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=CMAPSS_COLUMNS,
    )


def load_cmapss_rul_file(file_path: Path) -> pd.DataFrame:
    """Load endpoint RUL values and assign one-based engine IDs."""
    # Endpoint RUL values: each C-MAPSS test trajectory stops
    # before failure. The separate RUL file records how many
    # cycles remain after each engine's final observed test cycle.
    #
    # Example:
    # Test engine 1 ends at observed cycle 31.
    # Its endpoint RUL value is 112.
    # Therefore, it has 112 cycles remaining after cycle 31.

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"C-MAPSS RUL file not found: {path}")

    rul_data = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,  # The first row is data, not column names
        names=[RUL_COLUMN],  # Assign a column name manually
    )

    # Explanation:
    # General syntax: dataframe.insert(position, column_name, values)
    #
    # Before:
    # row    remaining_useful_life
    # 0      112
    # 1       98
    # 2       69
    #
    # After:
    # unit_number    remaining_useful_life
    # 1              112
    # 2               98
    # 3               69
    #
    # The first RUL value belongs to test engine 1, and so on.
    # We can later join these values to test data using unit_number.

    rul_data.insert(
        0,
        "unit_number",
        range(1, len(rul_data) + 1),
    )

    return rul_data