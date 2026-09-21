"""Tests for the complete leakage-safe data preparation workflow."""

import pandas as pd

import pytest

import pytest

from src.data.prepare_data import (
    prepare_training_validation_data,
    select_engine_endpoints,
)

from src.data.load_data import CMAPSS_COLUMNS, RUL_COLUMN
from src.data.prepare_data import prepare_training_validation_data

from src.features.build_features import build_features
from src.evaluation.evaluate_model import align_endpoint_predictions


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

    assert len(training_features) == 9
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


def test_select_engine_endpoints_selects_latest_cycle() -> None:
    observed = pd.DataFrame(
        {
            "unit_number": [2, 1, 2, 1],
            "time_in_cycles": [75, 10, 20, 40],
        },
        index=[101, 205, 309, 412],
    )

    features = pd.DataFrame(
        {
            "sensor_2_rolling_mean_5": [7.5, 1.0, 2.0, 4.0],
        },
        index=observed.index,
    )

    result = select_engine_endpoints(observed, features)

    expected = pd.DataFrame(
        {
            "sensor_2_rolling_mean_5": [4.0, 7.5],
        },
        index=pd.Index([1, 2], name="unit_number"),
    )

    pd.testing.assert_frame_equal(result, expected)


def test_select_engine_endpoints_rejects_misaligned_indexes() -> None:
    observed = pd.DataFrame(
        {
            "unit_number": [1, 1],
            "time_in_cycles": [10, 20],
        },
        index=[100, 200],
    )

    features = pd.DataFrame(
        {"sensor_2": [1.0, 2.0]},
        index=[200, 100],
    )

    with pytest.raises(ValueError, match="matching indexes"):
        select_engine_endpoints(observed, features)


def test_endpoint_workflow_preserves_history_and_target_alignment() -> None:
    observed = pd.DataFrame({
        "unit_number": [1, 1, 1, 2, 2, 2],
        "time_in_cycles": [1, 2, 3, 1, 2, 3],
        "sensor_2": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
    })

    features = build_features(
        observed,
        rolling_window=2,
    )

    endpoints = select_engine_endpoints(
        observed,
        features,
    )

    assert endpoints.index.tolist() == [1, 2]
    assert endpoints["sensor_2_rolling_mean_2"].tolist() == [
        25.0,
        250.0,
    ]
    assert endpoints["sensor_2_difference"].tolist() == [
        10.0,
        100.0,
    ]
    assert "unit_number" not in endpoints.columns

    # Stand-in predictions in the endpoint feature-row order.
    predictions = pd.Series(
        [12.0, 22.0],
        index=endpoints.index,
    )

    # Targets deliberately arrive in the opposite engine order.
    targets = pd.Series(
        [20.0, 10.0],
        index=[2, 1],
    )

    aligned = align_endpoint_predictions(predictions, targets)

    assert aligned.index.tolist() == [1, 2]
    assert aligned["actual"].tolist() == [10.0, 20.0]
    assert aligned["predicted"].tolist() == [12.0, 22.0]