"""Load C-MAPSS source data from disk."""

from pathlib import Path

import pandas as pd


def load_cmapss_file(file_path: Path) -> pd.DataFrame:
    """Load one whitespace-delimited C-MAPSS file.

    Column names and dataset-specific validation will be added when the dataset
    is introduced in a later milestone.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"C-MAPSS file not found: {path}")

    return pd.read_csv(path, sep=r"\s+", header=None)
