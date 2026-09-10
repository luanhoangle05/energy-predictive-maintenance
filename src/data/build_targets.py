"""Build Remaining Useful Life targets for C-MAPSS data."""

import pandas as pd

from src.data.load_data import RUL_COLUMN


def add_training_rul(data: pd.DataFrame) -> pd.DataFrame:
    """Add an uncapped RUL target to run-to-failure training data."""
    result = data.copy()

    final_cycles = result.groupby("unit_number")[
        "time_in_cycles"
    ].transform("max")

    result[RUL_COLUMN] = final_cycles - result["time_in_cycles"]

    return result


def add_test_rul(
    data: pd.DataFrame,
    endpoint_rul: pd.DataFrame,
) -> pd.DataFrame:
    """Add RUL targets to test trajectories using endpoint RUL values."""

    # unit number for the engine
    trajectory_units = set(data["unit_number"].unique())

    # unit number for the engine but in RUL file
    endpoint_units = set(endpoint_rul["unit_number"].unique())

    if trajectory_units != endpoint_units:
        raise ValueError(
            "Test trajectories and endpoint RUL engine IDs do not match."
        )

    result = data.copy()

    last_observed_cycle = result.groupby("unit_number")[
        "time_in_cycles"
    ].transform("max")

    # In load_data/load_cmapss_rul_file
    # We already insert a new column called unit_number
    # So, .set_index(unit_number) will set index for unit_number but be the same
    # Before: unit_number        remaining_useful_life
    #           1                         4
    #           2                         7
    #
    # After: unit_number index    remaining_useful_life
    #           1                         4
    #           2                         7
    #
    # Now, engine IDs act like lookup keys
    endpoint_lookup = endpoint_rul.set_index("unit_number")[RUL_COLUMN]
    #
    #
    # Example:
    # Engine 1:
    # last_observed_cycle = 31, endpoint_RUL=112
    # => Estimate failure cycle = 112 + 31 = 143
    # Then
    #   Current cycle         Calculated RUL
    #      1                    143 - 1 = 142
    #      2                    143 - 2 = 141
    #      .....                     .......
    #      31                   143 - 31 = 112
    #
    result[RUL_COLUMN] = (
        result["unit_number"].map(endpoint_lookup) # return the value of the matching engine
        + last_observed_cycle
        - result["time_in_cycles"]
    )

    return result