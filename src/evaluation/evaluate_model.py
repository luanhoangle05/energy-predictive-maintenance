"""Evaluate Remaining Useful Life regression predictions."""

from typing import Dict

import pandas as pd

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(
        actual: np.ndarray,
        predicted: np.ndarray
) -> Dict[str, float]:
    """Calculate evaluation metrics in a future milestone."""
    mae = mean_absolute_error(actual, predicted)

    mse = mean_squared_error(actual,predicted)
    rmse = np.sqrt(mse)

    r2=r2_score(actual,predicted)

    nasa_score = nasa_asymmetric_score(actual, predicted)

    return {
        "mae":float(mae),
        "rmse":float(rmse),
        "r2":float(r2),
        "nasa_score": float(nasa_score)
    }

def nasa_asymmetric_score(
        actual: np.ndarray,
        predicted: np.ndarray
) -> float:
    """Calculate the NASA asymmetric RUL prediction score."""
    errors = predicted - actual

    penalties = np.where(
        errors < 0,
        np.exp(-errors / 13.0) - 1.0,
        np.exp(errors / 10.0) - 1.0 # e^x
    )

    return float(np.sum(penalties))

def near_failure_metrics(
        actual: np.ndarray,
        predicted: np.ndarray,
        maximum_rul: float = 30.0,
) -> Dict[str, float | int]:
    """Calculate metrics for observations close to failure <=30"""

    # checks every actual values in the array actual to see if they are <= 30
    # Then the test: actual = np.array([10.0, 30.0, 31.0, 60.0])
    # Becomes: [True, True, False, False]
    near_failure_mask = actual <= maximum_rul

    # This one counts how many True values exist
    # Because True = 1 , sp they are considered non zero
    sample_count = int(np.count_nonzero(near_failure_mask))

    if sample_count == 0:
        raise ValueError(
            "No observations found at or below the maximum RUL"
        )


    # actual[near_failure_mask] becomes [10.0, 30.0]
    # predicted = np.array([12.0, 27.0, 1000.0, 0.0]) becomes:[12.0, 27.0]
    # Then it just evaluate the metrics including: MAE, RMSE, Nasa Score
    metrics = regression_metrics(
        actual[near_failure_mask],
        predicted[near_failure_mask],
    )

    return {
        "sample_count": sample_count,
        **metrics,
    }

def engine_level_metrics(
        actual: np.ndarray,
        predicted: np.ndarray,
        engine_ids: np.ndarray,
) -> pd.DataFrame:
    """Calculate prediction diagnostics separately for each engine."""
    actual_array = np.asarray(actual)
    predicted_array = np.asarray(predicted)
    engine_id_array = np.asarray(engine_ids)

    if not (
        len(actual_array)
        == len(predicted_array)
        == len(engine_id_array)
    ):
        raise ValueError(
            "actual, predicted, and engine_ids must have equal lengths"
        )

    engine_results = []

    for engine_id in np.unique(engine_id_array):
        engine_mask = engine_id_array == engine_id

        engine_actual = actual_array[engine_mask]
        engine_predicted = predicted_array[engine_mask]

        metrics = regression_metrics(
            engine_actual,
            engine_predicted,
        )

        errors = engine_predicted - engine_actual
        sample_count = len(engine_actual)

        engine_results.append({
            "unit_number": int(engine_id),
            "sample_count": int(sample_count),
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "nasa_score": metrics["nasa_score"],
            "mean_nasa_penalty": (
                metrics["nasa_score"] / sample_count
            ),
            "mean_error": float(np.mean(errors)),
            "overestimation_rate": float(np.mean(errors > 0)),
        })

    return pd.DataFrame(engine_results)



