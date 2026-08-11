"""Tests for leakage-safe feature preprocessing."""

import pandas as pd
import pytest

from src.features.preprocess_features import (
    create_feature_preprocessor,
)

# fit_transform(): learn mean, std, and variance
# fit_transform(): rules from training data, then transform training data

# .transform(): reused learned training rules without learning anything new

# create the training vs validation feature

# training feature: let preprocessor learn that:
# sensor_variable = keep
# sensor_train_constant = remove because variance = 0

# validation feature: then apply the same rule
# sensor_variable = keep
# sensor_train_constant = remove although variance > 0

# => Avoid preprocessor learn from the testing set

def test_variance_filter_is_fitted_using_training_data_only() -> None:
    training_features = pd.DataFrame(
        {
            "sensor_variable": [1.0, 2.0, 3.0],
            "sensor_train_constant": [10.0, 10.0, 10.0],
        }
    )

    validation_features = pd.DataFrame(
        {
            "sensor_variable": [4.0, 5.0],
            "sensor_train_constant": [10.0, 20.0],
        }
    )

    preprocessor = create_feature_preprocessor()

    transformed_training = preprocessor.fit_transform(
        training_features
    )
    transformed_validation = preprocessor.transform(
        validation_features
    )

    assert transformed_training.columns.tolist() == [
        "sensor_variable"
    ]

    assert transformed_validation.columns.tolist() == [
        "sensor_variable"
    ]

    # in transform_training: preprocessor calculates
    # mean= (1+2+3)/3 =2
    # std approx = 0.8165
    # then StandardScaler transforms the values:
    #
    # 1 → (1 - 2) / 0.8165 ≈ -1.2247
    # 2 → (2 - 2) / 0.8165 =  0
    # 3 → (3 - 2) / 0.8165 ≈  1.2247
    # # The mean of these three scaled values is approximately 0
    assert transformed_training["sensor_variable"].mean() == pytest.approx(
        0.0
    )

    assert transformed_training["sensor_variable"].std(
        ddof=0
    ) == pytest.approx(1.0)

# In this step: StandardScaler remembers what it learned from training
# training mean =2
# training std = 0.8165
# then validation 4 becomes: (4-2)/0.8165 = 2.44949
# then validation 5 becomes : (5-2)/0.8165 = 3.67423
# validate the assertion
    assert transformed_validation["sensor_variable"].tolist() == pytest.approx(
        [
            2.449489743,
            3.674234614,
        ]
    )

# StandardScaler () transform each feature using:
# z= (x-mean)/std


