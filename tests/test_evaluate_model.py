"""Tests for RUL regression evaluation metrics."""

import numpy as np
import pytest

from src.evaluation.evaluate_model import (
regression_metrics,
nasa_asymmetric_score,
near_failure_metrics,
engine_level_metrics,
rul_band_metrics,
feature_importance_table,
summarize_models_by_engine
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

def test_rul_band_metrics_assigns_boundaries_correctly() -> None:
    actual = np.array([
        0.0, 30.0,
        31.0, 60.0,
        61.0, 120.0,
        121.0, 150.0,
    ])
    predicted = np.array([
        1.0, 28.0,
        35.0, 55.0,
        71.0, 110.0,
        111.0, 140.0,
    ])

    results = rul_band_metrics(
        actual,
        predicted,
    )

    assert results["rul_band"].tolist() == [
        "0-30",
        "31-60",
        "61-120",
        ">120",
    ]
    assert results["sample_count"].tolist() == [
        2,
        2,
        2,
        2,
    ]

    near_failure = results.iloc[0]
    assert near_failure["mae"] == pytest.approx(1.5)
    assert near_failure["mean_error"] == pytest.approx(-0.5)
    assert near_failure["overestimation_rate"] == pytest.approx(0.5)

    high_rul = results.iloc[3]
    assert high_rul["mae"] == pytest.approx(10.0)
    assert high_rul["mean_error"] == pytest.approx(-10.0)
    assert high_rul["overestimation_rate"] == pytest.approx(0.0)

def test_rul_band_metrics_rejects_different_lengths() -> None:
    with pytest.raises(
        ValueError,
        match="actual and predicted must have equal lengths",
    ):
        rul_band_metrics(
            np.array([10.0, 20.0]),
            np.array([12.0]),
        )


def test_feature_importance_table_sorts_and_accumulates() -> None:
    feature_names = [
        "feature_a",
        "feature_b",
        "feature_c",
    ]
    importances = np.array([
        0.2,
        0.5,
        0.3,
    ])

    results = feature_importance_table(
        feature_names,
        importances,
    )

    assert results["rank"].tolist() == [1, 2, 3]
    assert results["feature"].tolist() == [
        "feature_b",
        "feature_c",
        "feature_a",
    ]
    assert results["importance"].tolist() == pytest.approx([
        0.5,
        0.3,
        0.2,
    ])
    assert results["cumulative_importance"].tolist() == pytest.approx([
        0.5,
        0.8,
        1.0,
    ])


def test_feature_importance_table_rejects_different_lengths() -> None:
    with pytest.raises(
        ValueError,
        match="feature_names and importances must have equal lengths",
    ):
        feature_importance_table(
            ["feature_a", "feature_b"],
            np.array([0.5]),
        )

def test_model_summary_weights_engines_equally() -> None:
    actual = np.array([10.0, 20.0, 10.0, 20.0, 30.0, 40.0])
    engine_ids = np.array([1, 1, 2, 2, 2, 2])

    predictions = {
        "Uneven errors": actual + np.array([
            10.0, 10.0,
            0.0, 0.0, 0.0, 0.0,
        ]),
        "Uniform errors": actual + 4.0,
    }

    results = summarize_models_by_engine(
        actual,
        predictions,
        engine_ids,
    )

    assert results["model"].tolist() == [
        "Uniform errors",
        "Uneven errors",
    ]

    indexed = results.set_index("model")
    uneven = indexed.loc["Uneven errors"]

    assert uneven["engine_count"] == 2
    assert uneven["mean_engine_mae"] == pytest.approx(5.0)
    assert uneven["mean_engine_rmse"] == pytest.approx(5.0)
    assert uneven["worst_engine_mae"] == pytest.approx(10.0)
    assert uneven["mean_overestimation_rate"] == pytest.approx(0.5)
    assert uneven["mean_engine_nasa_penalty"] == pytest.approx(
        (np.exp(1.0) - 1.0) / 2.0
    )

     
