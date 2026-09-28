"""Tests for leakage-safe engine-level dataset splitting."""

import pandas as pd

from src.data.split_data import split_by_engine

from src.data.split_data import (
    split_by_engine,
    split_training_calibration_evaluation,
)

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

def test_three_way_split_keeps_engines_separate() -> None:
    data = pd.DataFrame({
        "unit_number": [
            engine
            for engine in range(1, 101)
            for _ in range(3)
        ],
        "time_in_cycles": [1, 2, 3] * 100,
    })

    training, calibration, evaluation = (
        split_training_calibration_evaluation(data)
    )

    training_ids = set(training["unit_number"])
    calibration_ids = set(calibration["unit_number"])
    evaluation_ids = set(evaluation["unit_number"])

    assert len(training_ids) == 64
    assert len(calibration_ids) == 16
    assert len(evaluation_ids) == 20

    assert training_ids.isdisjoint(calibration_ids)
    assert training_ids.isdisjoint(evaluation_ids)
    assert calibration_ids.isdisjoint(evaluation_ids)

    combined = pd.concat(
        [training, calibration, evaluation]
    ).sort_index()

    pd.testing.assert_frame_equal(combined, data)
