"""Tests for model feature construction."""

import pandas as pd

from src.data.load_data import RUL_COLUMN
from src.features.build_features import build_features


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
        }
    )

    pd.testing.assert_frame_equal(result, expected)