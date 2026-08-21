import pandas as pd
import pytest

from src.models.train_model import train_dummy_regressor

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