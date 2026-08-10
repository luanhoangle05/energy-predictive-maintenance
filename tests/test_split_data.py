"""Tests for leakage-safe engine-level dataset splitting."""

import pandas as pd

from src.data.split_data import split_by_engine

def test_split_by_engine_trajectories_separate() -> None:
    data = pd.DataFrame(
        {
            "unit_number": [1,1,2,2,3,3,4,4],
            "time_in_cycles": [1,2,1,2,1,2,1,2],
        }
    )

    training_data, validation_data = split_by_engine(
        data,
        validation_size=0.5,
        random_state=42,
    )

    training_units = set(training_data["unit_number"].unique())
    validation_units = set(validation_data["unit_number"].unique())

    # first assertion: no engine leakage
    # proves that engine are not allowed to appear in both sets
    assert training_units.isdisjoint(validation_units)

    # union
    assert training_units | validation_units == {1,2,3,4}

    # make sure every row was included
    assert len(training_data) + len(validation_data) == len(data)


def test_split_by_engine_is_reproducible() -> None:
    data = pd.DataFrame(
        {
            "unit_number": [1,1,2,2,3,3,4,4],
            "time_in_cycles": [1,2,1,2,1,2,1,2]
        }
    )


    first_training, first_validation = split_by_engine(
        data,
        validation_size=0.5,
        random_state=42
    )

    second_training, second_validation= split_by_engine(
        data,
        validation_size=0.5,
        random_state= 42
    )

    pd.testing.assert_frame_equal(
        first_training, second_training
    )

    pd.testing.assert_frame_equal(
        first_validation, second_validation
    )
