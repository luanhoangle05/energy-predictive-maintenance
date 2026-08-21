"""Train Remaining Useful Life regression models."""

from typing import Any

import pandas as pd

from sklearn.dummy import DummyRegressor

def train_dummy_regressor(
        features: pd.DataFrame,
        target: pd.Series
) -> Any:
    """Train a Dummy Regressor model in a future milestone."""

    # Dummy Regressor acts as a baseline model
    # Why? The baseline asks:
    # Can my actual ML model do better than just predicting the average RUL for everything

    # Example:
    # Mean baseline RMSE = 35 cycles
    # Random Forest RMSE = 23 cycles
    # => Random Forest provides meaningful predictive value

    model = DummyRegressor(strategy="mean")
    model.fit(features,target)

    return model

