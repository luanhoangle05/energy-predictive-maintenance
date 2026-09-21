"""Evaluate Remaining Useful Life regression predictions."""

from typing import Dict, Sequence

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

def rul_band_metrics(
        actual: np.ndarray,
        predicted: np.ndarray,
) -> pd.DataFrame:
    """Calculate prediction diagnostics across RUL lifecycle bands."""
    actual_array = np.asarray(actual)
    predicted_array = np.asarray(predicted)

    if len(actual_array) != len(predicted_array):
        raise ValueError(
            "actual and predicted must have equal lengths"
        )

    band_masks = [
        ("0-30", actual_array <= 30),
        (
            "31-60",
            (actual_array > 30)
            & (actual_array <= 60),
        ),
        (
            "61-120",
            (actual_array > 60)
            & (actual_array <= 120),
        ),
        (">120", actual_array > 120),
    ]

    band_results = []

    for band_name, band_mask in band_masks:
        sample_count = int(np.count_nonzero(band_mask))

        if sample_count == 0:
            continue

        band_actual = actual_array[band_mask]
        band_predicted = predicted_array[band_mask]

        metrics = regression_metrics(
            band_actual,
            band_predicted,
        )

        errors = band_predicted - band_actual

        band_results.append({
            "rul_band": band_name,
            "sample_count": sample_count,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "nasa_score": metrics["nasa_score"],
            "mean_nasa_penalty": (
                metrics["nasa_score"] / sample_count
            ),
            "mean_error": float(np.mean(errors)),
            "overestimation_rate": float(
                np.mean(errors > 0)
            ),
        })

    return pd.DataFrame(band_results)

def feature_importance_table(
        feature_names: Sequence[str],
        importances: np.ndarray,
) -> pd.DataFrame:
    """Create a ranked feature-importance table."""
    feature_name_list = list(feature_names)
    importance_array = np.asarray(
        importances,
        dtype=float,
    )

    if len(feature_name_list) != len(importance_array):
        raise ValueError(
            "feature_names and importances must have equal lengths"
        )

    results = pd.DataFrame({
        "feature": feature_name_list,
        "importance": importance_array,
    })

    results = results.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

    results.insert(
        0,
        "rank",
        np.arange(1, len(results) + 1),
    )

    results["cumulative_importance"] = (
        results["importance"].cumsum()
    )

    return results

def summarize_models_by_engine(
        actual: np.ndarray,
        predictions_by_model: Dict[str, np.ndarray],
        engine_ids: np.ndarray,
) -> pd.DataFrame:
    """Compare models with equal weight given to each engine."""
    if not predictions_by_model:
        raise ValueError(
            "predictions_by_model must contain at least one model"
        )

    if len(actual) == 0:
        raise ValueError("actual must contain at least one observation")

    summary_results = []

    for model_name, predicted in predictions_by_model.items():
        per_engine = engine_level_metrics(
            actual,
            predicted,
            engine_ids,
        )

        summary_results.append({
            "model": model_name,
            "engine_count": int(len(per_engine)),
            "mean_engine_mae": float(per_engine["mae"].mean()),
            "median_engine_mae": float(per_engine["mae"].median()),
            "worst_engine_mae": float(per_engine["mae"].max()),
            "mean_engine_rmse": float(per_engine["rmse"].mean()),
            "mean_engine_nasa_penalty": float(
                per_engine["mean_nasa_penalty"].mean()
            ),
            "worst_engine_nasa_penalty": float(
                per_engine["mean_nasa_penalty"].max()
            ),
            "mean_overestimation_rate": float(
                per_engine["overestimation_rate"].mean()
            ),
        })

    return (
        pd.DataFrame(summary_results)
        .sort_values("mean_engine_mae")
        .reset_index(drop=True)
    )

def align_endpoint_predictions(
        predictions: pd.Series,
        targets: pd.Series,
) -> pd.DataFrame:
    """Match endpoint predictions and targets by engine ID."""
    if predictions.empty or targets.empty:
        raise ValueError("Predictions and targets must not be empty")

    if not predictions.index.is_unique or not targets.index.is_unique:
        raise ValueError("Engine IDs must be unique")

    if predictions.index.hasnans or targets.index.hasnans:
        raise ValueError("Engine IDs must not be missing")

    missing_predictions = targets.index.difference(predictions.index)
    missing_targets = predictions.index.difference(targets.index)

    if len(missing_predictions) or len(missing_targets):
        raise ValueError(
            "Predictions and targets must contain the same engine IDs"
        )

    aligned = pd.DataFrame({
        "actual": targets,
        "predicted": predictions.reindex(targets.index),
    }).sort_index()

    if not np.isfinite(aligned.to_numpy(dtype=float)).all():
        raise ValueError("Predictions and targets must be finite")

    aligned.index.name = "unit_number"

    return aligned



