import pandas as pd
import pytest

from src.models.train_model import train_model

def test_dummy_regressor_baseline_predicts_mean_target() -> None:
    features = pd.DataFrame(
        {
            "sensor_1": [10.0, 20.0, 30.0]
        }
    )

    target = pd.Series([9.0, 6.0, 3.0])

    model = train_model(features,target)
    predictions = model.predict(features)

    assert predictions.tolist() == pytest.approx(
        [6.0, 6.0, 6.0]
    )