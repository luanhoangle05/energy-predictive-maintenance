"""Create visual diagnostics for RUL model predictions."""
# Plot actual versus predicted RUL for engines 45 and 84

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd


def plot_engine_prediction_trajectory(
        cycles: np.ndarray,
        actual: np.ndarray,
        predicted: np.ndarray,
        engine_id: int,
        model_name: str,
        output_path: Path,
) -> None:
    """Plot actual and predicted RUL across one engine trajectory."""
    cycle_array = np.asarray(cycles)
    actual_array = np.asarray(actual)
    predicted_array = np.asarray(predicted)

    if not (
        len(cycle_array)
        == len(actual_array)
        == len(predicted_array)
    ):
        raise ValueError(
            "cycles, actual, and predicted must have equal lengths"
        )

    cycle_order = np.argsort(cycle_array)

    sorted_cycles = cycle_array[cycle_order]
    sorted_actual = actual_array[cycle_order]
    sorted_predicted = predicted_array[cycle_order]

    figure, axis = plt.subplots(figsize=(10, 6))

    axis.plot(
        sorted_cycles,
        sorted_actual,
        label="Actual RUL",
        color="black",
        linewidth=2,
    )
    axis.plot(
        sorted_cycles,
        sorted_predicted,
        label=f"{model_name} prediction",
        color="tab:blue",
        linewidth=2,
    )

    axis.axhspan(
        0,
        30,
        color="tab:red",
        alpha=0.1,
        label="Near failure: RUL <= 30",
    )

    axis.set_title(
        f"{model_name} RUL trajectory — Engine {engine_id}"
    )
    axis.set_xlabel("Time in cycles")
    axis.set_ylabel("Remaining Useful Life")
    axis.grid(alpha=0.3)
    axis.legend()

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)

def plot_residuals_by_actual_rul(
        actual: np.ndarray,
        predicted: np.ndarray,
        model_name: str,
        output_path: Path,
) -> None:
    """Plot prediction errors against actual RUL."""
    actual_array = np.asarray(actual)
    predicted_array = np.asarray(predicted)

    if len(actual_array) != len(predicted_array):
        raise ValueError(
            "actual and predicted must have equal lengths"
        )

    errors = predicted_array - actual_array
    overestimation_mask = errors > 0

    figure, axis = plt.subplots(figsize=(10, 6))

    # The first scatter is gonna be NOT overestimation
    # The sign ~ means NOT
    axis.scatter(
        actual_array[~overestimation_mask], # x-axis: actual RUL
        errors[~overestimation_mask], # y-axis: prediction errors
        color="tab:blue",
        alpha=0.35,
        s=18,
        label="Underestimate or exact",
    )

    # This scatter is for overestimation

    axis.scatter(
        actual_array[overestimation_mask], # x-axis: actual RUL, how much life actually remained
        errors[overestimation_mask], # y-axis: prediction errors, how wrong the prediction was
        color="tab:red",
        alpha=0.35,
        s=18,
        label="Overestimate",
    )

    # create a horizontal zero-error line, means perfect and have no errors
    axis.axhline(
        0,
        color="black",
        linewidth=1.5,
    )
    axis.axvspan(
        0,
        30,
        color="tab:red",
        alpha=0.08,
        label="Near failure: RUL <= 30",
    )

    axis.set_title(
        f"{model_name} residuals by actual RUL"
    )
    axis.set_xlabel("Actual Remaining Useful Life")
    axis.set_ylabel("Prediction error: predicted - actual")
    axis.grid(alpha=0.3)
    axis.legend()

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)

def plot_feature_importances(
        feature_importance: pd.DataFrame,
        model_name: str,
        output_path: Path,
        top_n: int = 15,
) -> None:
    """Plot the most important model features."""
    required_columns = {
        "feature",
        "importance",
    }

    if not required_columns.issubset(feature_importance.columns):
        raise ValueError(
            "feature_importance must contain feature and importance columns"
        )

    if top_n <= 0:
        raise ValueError("top_n must be greater than zero")

    top_features = (
        feature_importance
        .nlargest(top_n, "importance")
        .sort_values("importance")
    )

    figure, axis = plt.subplots(figsize=(10, 7))

    bars = axis.barh(
        top_features["feature"],
        top_features["importance"],
        color="tab:blue",
    )

    axis.bar_label(
        bars,
        fmt="%.3f",
        padding=3,
    )

    axis.set_title(
        f"{model_name} top {len(top_features)} feature importances"
    )
    axis.set_xlabel("Split-based feature importance")
    axis.set_ylabel("Feature")
    axis.grid(
        axis="x",
        alpha=0.3,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)