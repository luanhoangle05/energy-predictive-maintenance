"""Train Remaining Useful Life regression models."""

from typing import Any

import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

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

def train_linear_regression(
        features: pd.DataFrame,
        target: pd.Series
) -> Any:
    """Train a Linear Regression RUL model."""
    model = LinearRegression()
    model.fit(features,target)
    return model

def train_decision_tree_regressor(
    features: pd.DataFrame,
    target: pd.Series,
    random_state: int = 42,
) -> Any:
    """Train an unrestricted Decision Tree baseline."""
    model = DecisionTreeRegressor(
        random_state=random_state,
    )
    model.fit(features, target)

    return model
