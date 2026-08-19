"""Tests for RUL regression evaluation metrics."""

import numpy as np
import pytest

from src.evaluation.evaluate_model import regression_metrics

def test_regression_metrics_calculates_mean_absolute_error() -> None:
    actual = np.array([10.0, 20.0, 30.0])
    predicted = np.array([12.0, 18.0, 32.0])

    metrics = regression_metrics(actual, predicted)

    assert metrics["mae"] == pytest.approx(2.0)

    # We assert MAE is 2.0 because
    # |10-12| = 2
    # |20-18| = 2
    # |30-32| = 2

    # MAE = (2+2+2) / 3 = 2

def test_regression_metrics_calculates_root_mean_squared_error() -> None:
    actual = np.array([10.0, 20.0, 30.0])
    predicted = np.array([10.0, 20.0, 33.0])

    metrics = regression_metrics(actual, predicted)

    assert metrics["rmse"] == pytest.approx(
        np.sqrt(3.0)
    )

    # The calculation is
    # Errors: 0,0,3
    # Squared errors: 0,0,9
    # Mean squared: (0+0+9)/3= 3
    # RMSE: square root of 3 = 1.732

def test_regression_metrics_calculates_r_squared() -> None:
    actual = np.array([10.0, 20.0, 30.0])
    predicted = np.array([12.0, 18.0, 32.0])

    metrics = regression_metrics(actual, predicted)

    assert metrics["r2"] == pytest.approx(0.94)

    # Why R squared = 0.94
    # Actual mean = 20
    # Squared prediction errors: (10 - 12)² + (20 - 18)² + (30 - 32)² = 12
    # Total variation from the mean: (10 - 20)² + (20 - 20)² + (30 - 20)² = 200
    # R squared = 1 - (12 / 200) = 0.94