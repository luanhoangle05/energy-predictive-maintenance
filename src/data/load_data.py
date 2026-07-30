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
