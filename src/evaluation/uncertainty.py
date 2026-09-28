"""Helpers for RUL uncertainty calibration."""

import numpy as np


def calibrate_interval_radius(
        actual: np.ndarray,
        predicted: np.ndarray,
        confidence: float = 0.90,
) -> float:
    """Calculate an interval radius from aligned calibration pairs."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError(
            "actual and predicted must be one-dimensional"
        )

    if actual.size == 0 or actual.size != predicted.size:
        raise ValueError(
            "actual and predicted must have equal nonzero lengths"
        )

    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError(
            "actual and predicted must contain finite values"
        )

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    sorted_errors = np.sort(np.abs(actual - predicted))
    calibration_count = len(sorted_errors)

    rank = int(np.ceil(
        (calibration_count + 1) * confidence
    ))

    if rank > calibration_count:
        raise ValueError(
            "Too few calibration engines for a finite interval "
            "at the requested confidence."
        )

    return float(sorted_errors[rank - 1])

def build_prediction_intervals(
        predicted: np.ndarray,
        radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build RUL intervals around non-negative point predictions."""
    predicted = np.asarray(predicted, dtype=float)

    if predicted.ndim != 1 or predicted.size == 0:
        raise ValueError(
            "predicted must be a non-empty one-dimensional array"
        )

    if not np.isfinite(predicted).all():
        raise ValueError("predicted must contain finite values")

    if np.any(predicted < 0.0):
        raise ValueError("predicted must be non-negative")

    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("radius must be finite and non-negative")

    lower = np.maximum(predicted - radius, 0.0)
    upper = predicted + radius

    return lower, upper

def classify_threshold_status(
        lower: np.ndarray,
        upper: np.ndarray,
        threshold: float = 30.0,
) -> np.ndarray:
    """Describe each RUL interval's position relative to a threshold."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    if lower.ndim != 1 or upper.ndim != 1:
        raise ValueError("bounds must be one-dimensional")

    if lower.size == 0 or lower.size != upper.size:
        raise ValueError("bounds must have equal nonzero lengths")

    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("bounds must contain finite values")

    if np.any(lower < 0.0) or np.any(lower > upper):
        raise ValueError("bounds must satisfy 0 <= lower <= upper")

    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be finite and non-negative")

    return np.select(
        [
            upper <= threshold,
            lower <= threshold,
        ],
        [
            "entirely_at_or_below",
            "crosses_threshold",
        ],
        default="entirely_above",
    )