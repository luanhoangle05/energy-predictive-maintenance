import pandas as pd
import pytest

from src.models.train_model import (
    train_dummy_regressor,
    train_linear_regression
)

def test_dummy_regressor_baseline_predicts_mean_target() -> None:

    # For Dummy Regressor: the sensor features do not matter at all
    # It only looks at target and calculate the average

    # A real model may actually learn the pattern
    # sensor_1 = 10 => maybe high RUL
    # sensor_1 = 20 => maybe medium RUL
    # sensor_ 1 = 30 => maybe low RUL

    features = pd.DataFrame(
        {
            "sensor_1": [10.0, 20.0, 30.0]
        }
    )

    target = pd.Series([9.0, 6.0, 3.0])

    model = train_dummy_regressor(features,target)
    predictions = model.predict(features)

    assert predictions.tolist() == pytest.approx(
        [6.0, 6.0, 6.0]
    )

def test_linear_regression_learns_linear_relationship() -> None:

    # This creates a test for Linear Regression
    # The training set has:
    # x=0 => y=5
    # x=1 => y=7
    # x=2 => y=9
    # So, we have a function: y=2x+5
    # So the model learn this pattern
    # Do the testing on prediction_feature which has x=3
    # => y=2*3+5 = 11
    training_features = pd.DataFrame(
        {
            "sensor_1": [0.0, 1.0, 2.0],
        }
    )
    training_target = pd.Series([5.0, 7.0, 9.0])

    prediction_features = pd.DataFrame(
        {
            "sensor_1": [3.0],
        }
    )

    model = train_linear_regression(
        training_features,
        training_target,
    )
    predictions = model.predict(prediction_features)

    assert predictions.tolist() == pytest.approx([11.0])