"""Train Remaining Useful Life regression models."""

from typing import Any

import pandas as pd

from sklearn.dummy import DummyRegressor

def train_model(
        features: pd.DataFrame,
        target: pd.Series
) -> Any
    """Train a regression model in a future milestone."""

    model = DummyRegressor(strategy="mean")
    model.fit(features,target)

    return model

