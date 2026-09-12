"""Tests for RUL regression evaluation metrics."""

import numpy as np
import pytest

from src.evaluation.evaluate_model import (
regression_metrics,
nasa_asymmetric_score,
near_failure_metrics,
engine_level_metrics
)

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

def test_nasa_score_is_zero_for_exact_prediction() -> None:
    actual = np.array([10.0, 20.0, 30.0])
    predicted = np.array([10.0, 20.0, 30.0])

    score = nasa_asymmetric_score(actual, predicted)

    assert score == pytest.approx(0.0)

def test_nasa_score_penalizes_overestimation_more() -> None:
    actual = np.array([100.0])

    under_score = nasa_asymmetric_score(
        actual,
        np.array([90.0])
    )

    over_score = nasa_asymmetric_score(
        actual,
        np.array([110.0])
    )

    assert under_score == pytest.approx(np.exp(10/13) - 1)
    assert over_score == pytest.approx(np.exp(10/10) - 1)
    assert over_score > under_score

def test_nasa_score_sums_individual_penalties() -> None:
    actual = np.array([100.0, 100.0, 100.0])
    predicted = np.array([100.0, 90.0, 110.0])

    score = nasa_asymmetric_score(actual, predicted)

    expected = (
        0.0 +
        (np.exp(10/13) - 1) +
        (np.exp(10/10) - 1)

    )

    assert score == pytest.approx(expected)
    
def test_regression_metrics_includes_nasa_score() -> None:
    actual = np.array([100.0, 80.0])
    predicted = np.array([110.0, 80.0])
    
    metrics = regression_metrics(actual, predicted)
    
    assert metrics["nasa_score"] == pytest.approx(
        # the first 10 comes from : 110- 100 = 10
        # because the predicted > actual = 10
        # so we use the formula overscore
        np.exp(10.0/10.0) - 1
    )

def test_near_failure_metrics_uses_only_rows_at_or_below_threshold() -> None:
    """Calculate metrics for observations close to failure <=30"""

    actual = np.array([10.0, 30.0, 31.0, 60.0])
    predicted = np.array([12.0, 27.0, 1000.0, 0.0])

    metrics = near_failure_metrics(
        actual,
        predicted,
        maximum_rul=30.0,
    )

    assert metrics["sample_count"] == 2
    assert metrics["mae"] == pytest.approx(2.5)
    assert metrics["rmse"] == pytest.approx(np.sqrt(6.5))
    assert metrics["nasa_score"] == pytest.approx(
        (np.exp(2.0 / 10.0) - 1.0)
        + (np.exp(3.0 / 13.0) - 1.0)
    )

def test_near_failure_metrics_rejects_empty_range() -> None:
    """Test if there are no observations with true RUL <=30"""

    actual = np.array([31.0, 50.0])
    predicted = np.array([30.0, 45.0])

    with pytest.raises(
        ValueError,
        match="No observations found at or below the maximum RUL",
    ):
        near_failure_metrics(
            actual,
            predicted,
            maximum_rul=30.0,
        )

def test_engine_level_metrics_evaluates_engines_separately() -> None:
    """Test how well the model perform on each individual engine"""
    # To check if a model can have a good overall MAE but performing badly on specific engines

    actual = np.array([10.0, 20.0, 10.0, 20.0])
    predicted = np.array([12.0, 18.0, 20.0, 30.0])
    engine_ids = np.array([1, 1, 2, 2])

    results= engine_level_metrics(
        actual,
        predicted,
        engine_ids
    )

    assert results["unit_number"].tolist() == [1, 2]

    engine_1 = results.iloc[0]

    assert engine_1["sample_count"] == 2
    assert engine_1["mae"] == pytest.approx(2.0)
    assert engine_1["rmse"] == pytest.approx(2.0)
    assert engine_1["mean_error"] == pytest.approx(0.0)
    assert engine_1["overestimation_rate"] == pytest.approx(0.5)

    engine_2 = results.iloc[1]

    assert engine_2["sample_count"] == 2
    assert engine_2["mae"] == pytest.approx(10.0)
    assert engine_2["rmse"] == pytest.approx(10.0)
    assert engine_2["mean_error"] == pytest.approx(10.0)
    assert engine_2["overestimation_rate"] == pytest.approx(1.0)

def test_engine_level_metrics_rejects_different_lengths() -> None:
    actual = np.array([10.0, 20.0])
    predicted = np.array([12.0])
    engine_ids = np.array([1, 1])

    with pytest.raises(
        ValueError,
        match="actual, predicted, and engine_ids must have equal lengths",
    ):
        engine_level_metrics(
            actual,
            predicted,
            engine_ids,
        )

     
