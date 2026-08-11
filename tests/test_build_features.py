"""Tests for model feature construction."""

import pandas as pd

from src.data.load_data import RUL_COLUMN
from src.features.build_features import (
build_features,
add_rolling_sensor_means,
add_rolling_sensor_standard_deviations,
add_sensor_difference
)


def test_build_features_excludes_identifier_and_target() -> None:
    data = pd.DataFrame(
        {
            "unit_number": [1, 1],
            "time_in_cycles": [1, 2],
            "sensor_1": [10.0, 11.0],
            RUL_COLUMN: [1, 0],
        }
    )

    result = build_features(data)

    assert result.columns.tolist() == [
        "time_in_cycles",
        "sensor_1",
        "sensor_1_rolling_mean_5",
        "sensor_1_rolling_std_5",
        "sensor_1_difference"
    ]

    assert result["sensor_1_difference"].tolist() == [
        0.0,
        1.0,
    ]

    assert result["sensor_1_rolling_std_5"].tolist() == [
        0.0,
        0.5,
    ]

    assert result["sensor_1_rolling_mean_5"].tolist() == [
        10.0,
        10.5
    ]

    assert "unit_number" not in result.columns
    assert RUL_COLUMN not in result.columns

    # The original data must remain unchanged.
    assert "unit_number" in data.columns
    assert RUL_COLUMN in data.columns

def test_build_features_accepts_unlabeled_prediction_data() -> None:
    data = pd.DataFrame(
        {
            "unit_number": [5, 5],
            "time_in_cycles": [10, 11],
            "sensor_1": [20.0, 21.0],
        }
    )

    result = build_features(data)

    expected = pd.DataFrame(
        {
            "time_in_cycles": [10, 11],
            "sensor_1": [20.0, 21.0],
            "sensor_1_rolling_mean_5": [20.0, 20.5],
            "sensor_1_rolling_std_5": [0.0, 0.5],
            "sensor_1_difference": [0.0, 1.0],
        }
    )

    pd.testing.assert_frame_equal(result, expected)


def test_rolling_sensor_means_do_not_cross_engine_boundaries() -> None:
    data = pd.DataFrame(
        {
            "unit_number": [1, 1, 1, 2, 2],
            "time_in_cycles": [1, 2, 3, 1, 2],
            "sensor_1": [10.0, 20.0, 30.0, 100.0, 200.0],
        }
    )

# window_size = 2
# current cycle and 1 previous cycle
# cycle 1 uses [10] => mean = 10
# cycle 2 uses [10.20] => mean =15
# cycle 3 uses [20,30] => mean = 25


    result = add_rolling_sensor_means(
        data,
        window_size=2,
    )

    assert result["sensor_1_rolling_mean_2"].tolist() == [
        10.0,
        15.0,
        25.0,
        100.0,
        150.0,
    ]

    assert "sensor_1_rolling_mean_2" not in data.columns


def test_rolling_sensor_std_does_not_cross_engine_boundaries() -> None:
    data = pd.DataFrame(
        {
            "unit_number": [1, 1, 1, 2, 2],
            "time_in_cycles": [1, 2, 3, 1, 2],
            "sensor_1": [10.0, 20.0, 30.0, 100.0, 200.0],
        }
    )

    result = add_rolling_sensor_standard_deviations(
        data,
        window_size=2,
    )

    assert result["sensor_1_rolling_std_2"].tolist() == [
        0.0,
        5.0,
        5.0,
        0.0,
        50.0,
    ]

    assert "sensor_1_rolling_std_2" not in data.columns

def test_sensor_differences_do_not_cross_engine_boundaries() -> None:
    data = pd.DataFrame(
        {
            "unit_number": [1, 1, 1, 2, 2],
            "time_in_cycles": [1, 2, 3, 1, 2],
            "sensor_1": [10.0, 20.0, 30.0, 100.0, 200.0],
        }
    )

    result = add_sensor_difference(data)

    assert result["sensor_1_difference"].tolist() == [
        0.0,
        10.0,
        10.0,
        0.0,
        100.0,
    ]

    assert "sensor_1_difference" not in data.columns