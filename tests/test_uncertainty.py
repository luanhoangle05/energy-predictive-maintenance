"""Tests for uncertainty calibration."""

import numpy as np
import pytest

from src.evaluation.uncertainty import (
    calibrate_interval_radius,
    build_prediction_intervals,
    classify_threshold_status
)


def test_radius_uses_corrected_rank_and_absolute_errors() -> None:
    actual = np.full(5, 20.0)
    predicted = np.array([28.0, 19.0, 24.0, 18.0, 23.0])

    # Sorted absolute errors: [1, 2, 3, 4, 8]
    # Corrected rank: ceil((5 + 1) * 0.60) = 4
    radius = calibrate_interval_radius(
        actual,
        predicted,
        confidence=0.60,
    )

    assert radius == pytest.approx(4.0)


def test_radius_uses_largest_error_for_16_engines_at_90_percent() -> None:
    actual = np.zeros(16)
    predicted = np.arange(1.0, 17.0)

    radius = calibrate_interval_radius(
        actual,
        predicted,
        confidence=0.90,
    )

    assert radius == pytest.approx(16.0)


def test_radius_rejects_insufficient_calibration_data() -> None:
    with pytest.raises(ValueError, match="Too few calibration engines"):
        calibrate_interval_radius(
            np.zeros(5),
            np.ones(5),
            confidence=0.90,
        )


@pytest.mark.parametrize(
    "actual, predicted, message",
    [
        ([], [], "equal nonzero lengths"),
        ([1.0, 2.0], [1.0], "equal nonzero lengths"),
        ([[1.0]], [1.0], "one-dimensional"),
        ([1.0], [[1.0]], "one-dimensional"),
        ([np.nan], [1.0], "finite values"),
        ([1.0], [np.inf], "finite values"),
    ],
)
def test_radius_rejects_invalid_arrays(
        actual, predicted, message,
) -> None:
    with pytest.raises(ValueError, match=message):
        calibrate_interval_radius(actual, predicted)


@pytest.mark.parametrize(
    "confidence",
    [0.0, 1.0, -0.1, 1.1, np.nan, np.inf],
)
def test_radius_rejects_invalid_confidence(confidence) -> None:
    with pytest.raises(
        ValueError,
        match="confidence must be between 0 and 1",
    ):
        calibrate_interval_radius(
            np.zeros(16),
            np.ones(16),
            confidence=confidence,
        )

def test_intervals_apply_zero_floor_and_preserve_upper_bounds() -> None:
    lower, upper = build_prediction_intervals(
        np.array([10.0, 100.0]),
        radius=20.0,
    )

    np.testing.assert_allclose(lower, [0.0, 80.0])
    np.testing.assert_allclose(upper, [30.0, 120.0])
    np.testing.assert_allclose(upper - lower, [30.0, 40.0])


def test_zero_radius_returns_point_predictions() -> None:
    predicted = np.array([0.0, 25.0])

    lower, upper = build_prediction_intervals(
        predicted,
        radius=0.0,
    )

    np.testing.assert_array_equal(lower, predicted)
    np.testing.assert_array_equal(upper, predicted)


@pytest.mark.parametrize("radius", [-1.0, np.nan, np.inf])
def test_intervals_reject_invalid_radius(radius) -> None:
    with pytest.raises(
        ValueError,
        match="radius must be finite and non-negative",
    ):
        build_prediction_intervals(np.array([10.0]), radius)


@pytest.mark.parametrize(
    "predicted, message",
    [
        ([], "non-empty one-dimensional"),
        ([[10.0]], "non-empty one-dimensional"),
        ([np.nan], "finite values"),
        ([np.inf], "finite values"),
        ([-1.0], "non-negative"),
    ],
)
def test_intervals_reject_invalid_predictions(predicted, message) -> None:
    with pytest.raises(ValueError, match=message):
        build_prediction_intervals(predicted, radius=20.0)

def test_threshold_status_handles_boundaries() -> None:
    lower = np.array([10.0, 30.0, 31.0, 30.0, 0.0])
    upper = np.array([30.0, 50.0, 50.0, 30.0, 20.0])

    result = classify_threshold_status(
        lower,
        upper,
        threshold=30.0,
    )

    assert result.tolist() == [
        "entirely_at_or_below",  # [10, 30]
        "crosses_threshold",    # [30, 50]
        "entirely_above",       # [31, 50]
        "entirely_at_or_below",  # [30, 30]
        "entirely_at_or_below",  # [0, 20]
    ]

@pytest.mark.parametrize(
    "lower, upper, message",
    [
        ([], [], "equal nonzero lengths"),
        ([0.0, 10.0], [20.0], "equal nonzero lengths"),
        ([[0.0]], [20.0], "one-dimensional"),
        ([0.0], [[20.0]], "one-dimensional"),
        ([np.nan], [20.0], "finite values"),
        ([0.0], [np.inf], "finite values"),
        ([-1.0], [20.0], "bounds must satisfy"),
        ([30.0], [20.0], "bounds must satisfy"),
    ],
)
def test_threshold_status_rejects_invalid_bounds(
        lower, upper, message,
) -> None:
    with pytest.raises(ValueError, match=message):
        classify_threshold_status(lower, upper)


@pytest.mark.parametrize("threshold", [-1.0, np.nan, np.inf])
def test_threshold_status_rejects_invalid_threshold(threshold) -> None:
    with pytest.raises(
        ValueError,
        match="threshold must be finite and non-negative",
    ):
        classify_threshold_status(
            np.array([0.0]),
            np.array([20.0]),
            threshold=threshold,
        )