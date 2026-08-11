"""Tests for the complete leakage-safe data preparation workflow."""

import pandas as pd

from src.data.load_data import CMAPSS_COLUMNS, RUL_COLUMN
from src.data.prepare_data import prepare_training_validation_data


def test_prepare_training_validation_data_end_to_end() -> None:
    rows = []

# creating 5 engines, each has 3 cycles
    for unit_number in range(1, 6):
        for cycle in range(1, 4):

            # create a dictionary containing 26 required C-MAPSS columns
            # initially filled with 0.0
            # to pass the require-column validation
            row = {
                column: 0.0
                for column in CMAPSS_COLUMNS
            }

            # given important values
            # sensor 2 is given changing values so dataset has at leats 1 nonconstant sensor
            row["unit_number"] = unit_number
            row["time_in_cycles"] = cycle
            row["sensor_2"] = unit_number * 10 + cycle

            rows.append(row)

    data = pd.DataFrame(
        rows,
        columns=CMAPSS_COLUMNS,
    )

    (
        training_features,
        validation_features,
        training_targets,
        validation_targets,
        preprocessor,
    ) = prepare_training_validation_data(
        data,
        validation_size=0.4,
        random_state=42,
        rolling_window=2,
    )

    assert len(training_features) == 8
    assert len(validation_features) == 6

    assert training_features.index.equals(
        training_targets.index
    )
    assert validation_features.index.equals(
        validation_targets.index
    )

    assert "unit_number" not in training_features.columns
    assert RUL_COLUMN not in training_features.columns

    assert not training_features.isna().any().any()
    assert not validation_features.isna().any().any()

    training_units = set(
        data.loc[training_features.index, "unit_number"]
    )
    validation_units = set(
        data.loc[validation_features.index, "unit_number"]
    )

    assert training_units.isdisjoint(validation_units)
    assert preprocessor is not None