import pandas as pd
import pytest

from src.models.train_model import (
train_decision_tree_regressor,
train_dummy_regressor,
train_linear_regression,
train_random_forest_regressor,
train_gradient_boosting_regressor
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

def test_decision_tree_regressor_learns_threshold_relationship() -> None:

    # The relationship has a clear threshold
    # sensor_1 below 2 -> target 10
    # sensor_1 at least 2 -> target 2
    training_features = pd.DataFrame(
        {
            "sensor_1": [0.0, 1.0, 2.0, 3.0],
        }
    )
    training_target = pd.Series(
        [10.0, 10.0, 2.0, 2.0]
    )

    model = train_decision_tree_regressor(
        training_features,
        training_target,
    )
    predictions = model.predict(training_features)

    assert predictions.tolist() == pytest.approx(
        [10.0, 10.0, 2.0, 2.0]
    )

def test_decision_tree_regressor_applies_complexity_controls() -> None:
    training_features = pd.DataFrame(
        {
            "sensor_1": [0.0, 1.0, 2.0, 3.0],
        }
    )
    training_target = pd.Series(
        [10.0, 10.0, 2.0, 2.0]
    )

    model = train_decision_tree_regressor(
        training_features,
        training_target,
        max_depth=1,
        min_samples_leaf=2,
    )

    assert model.max_depth == 1
    assert model.min_samples_leaf == 2

def test_random_forest_regressor_trains_ensemble() -> None:
    training_features = pd.DataFrame(
        {
            "sensor_1": [
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ],
        }
    )
    training_target = pd.Series(
        [10.0, 10.0, 8.0, 5.0, 2.0, 2.0]
    )

    model = train_random_forest_regressor(
        training_features,
        training_target,
        n_estimators=10,
    )

    predictions = model.predict(training_features)

    assert model.n_estimators == 10
    assert model.random_state == 42
    assert predictions.shape == (6,)

def test_gradient_boosting_regressor_trains_ensemble() -> None:
    training_features = pd.DataFrame(
        {
            "sensor_1": [
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ],
        }
    )
    training_target = pd.Series(
        [10.0, 10.0, 8.0, 5.0, 2.0, 2.0]
    )

    model = train_gradient_boosting_regressor(
        training_features,
        training_target,
        n_estimators=10,
    )

    predictions = model.predict(training_features)

    assert model.n_estimators == 10
    assert model.learning_rate == 0.1
    assert model.random_state == 42
    assert predictions.shape == (6,)