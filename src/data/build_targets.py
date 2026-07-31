"""Build Remaining Useful Life targets for C-MAPSS data."""

import pandas as pd


RUL_COLUMN = "remaining_useful_life"


def add_training_rul(data: pd.DataFrame) -> pd.DataFrame:
    """Add an uncapped RUL target to run-to-failure training data."""
    result = data.copy()

    final_cycles = result.groupby("unit_number")[
        "time_in_cycles"
    ].transform("max")

    result[RUL_COLUMN] = final_cycles - result["time_in_cycles"]

    return result