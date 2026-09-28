"""Run the development uncertainty-calibration workflow."""

from src.data.load_data import load_cmapss_file, RUL_COLUMN
from src.data.prepare_data import (
    prepare_uncertainty_splits,
    select_cutoff_indices,
)
from src.features.build_features import build_features
from src.features.preprocess_features import create_feature_preprocessor

import numpy as np

from src.models.train_model import train_gradient_boosting_regressor
from src.models.predict import predict_rul

import pandas as pd

from src.evaluation.uncertainty import (
    calibrate_interval_radius,
    build_prediction_intervals,
    classify_threshold_status
)


def main() -> None:
    raw_data = load_cmapss_file(
        "data/raw/cmapss/train_FD001.txt"
    )

    training_data, calibration_data, evaluation_data = (
        prepare_uncertainty_splits(raw_data)
    )

    training_targets = training_data[RUL_COLUMN].copy()
    calibration_targets = calibration_data[RUL_COLUMN].copy()
    evaluation_targets = evaluation_data[RUL_COLUMN].copy()

    training_features = build_features(training_data)
    calibration_features = build_features(calibration_data)
    evaluation_features = build_features(evaluation_data)

    preprocessor = create_feature_preprocessor()

    training_features = preprocessor.fit_transform(
        training_features
    )
    calibration_features = preprocessor.transform(
        calibration_features
    )
    evaluation_features = preprocessor.transform(
        evaluation_features
    )

    calibration_indices = select_cutoff_indices(
        calibration_data,
        random_state=42,
    )
    evaluation_indices = select_cutoff_indices(
        evaluation_data,
        random_state=43,
    )

    calibration_features = calibration_features.loc[
        calibration_indices
    ].copy()
    calibration_targets = calibration_targets.loc[
        calibration_indices
    ].copy()

    evaluation_features = evaluation_features.loc[
        evaluation_indices
    ].copy()
    evaluation_targets = evaluation_targets.loc[
        evaluation_indices
    ].copy()

    for name, features, targets in (
        ("Training", training_features, training_targets),
        ("Calibration", calibration_features, calibration_targets),
        ("Evaluation", evaluation_features, evaluation_targets),
    ):
        print(f"{name} feature shape: {features.shape}")
        print(
            f"{name} indexes aligned: "
            f"{features.index.equals(targets.index)}"
        )

    for name, data, indices in (
        ("Calibration", calibration_data, calibration_indices),
        ("Evaluation", evaluation_data, evaluation_indices),
    ):
        snapshots = data.loc[
            indices,
            ["unit_number", "time_in_cycles", RUL_COLUMN],
        ]

        print(f"\n{name} snapshots:")
        print(snapshots.head().to_string(index=False))
        print(
            "One snapshot per engine:",
            snapshots["unit_number"].is_unique,
        )
        print(
            "All cutoffs at cycle 30 or later:",
            (snapshots["time_in_cycles"] >= 30).all(),
        )
        print(
            "All snapshots before failure:",
            (snapshots[RUL_COLUMN] > 0).all(),
        )

    model = train_gradient_boosting_regressor(
        training_features,
        training_targets,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=30,
        random_state=42,
    )

    calibration_predictions = np.clip(
        predict_rul(model, calibration_features),
        a_min=0.0,
        a_max=None,
    )
    evaluation_predictions = np.clip(
        predict_rul(model, evaluation_features),
        a_min=0.0,
        a_max=None,
    )

    print("\nCalibration predictions:", calibration_predictions.shape)
    print("Evaluation predictions:", evaluation_predictions.shape)
    print(
        "First five calibration predictions:",
        np.round(calibration_predictions[:5], 2),
    )
    print(
        "First five calibration targets:",
        calibration_targets.iloc[:5].to_numpy(),
    )

    calibration_errors = np.abs(
        calibration_targets.to_numpy()
        - calibration_predictions
    )

    sorted_errors = np.sort(calibration_errors)

    print(
        "\nSorted calibration absolute errors:",
        np.round(sorted_errors, 2),
    )
    print("Calibration error count:", len(sorted_errors))

    confidence = 0.90

    interval_radius = calibrate_interval_radius(
        calibration_targets.to_numpy(),
        calibration_predictions,
        confidence=confidence,
    )

    evaluation_lower, evaluation_upper = build_prediction_intervals(
        evaluation_predictions,
        interval_radius,
    )

    print(f"Interval radius: {interval_radius:.2f} cycles")
    print("First five lower bounds:", np.round(evaluation_lower[:5], 2))
    print("First five upper bounds:", np.round(evaluation_upper[:5], 2))

    evaluation_actual = evaluation_targets.to_numpy()

    covered = (
        (evaluation_actual >= evaluation_lower)
        & (evaluation_actual <= evaluation_upper)
    )
    interval_widths = evaluation_upper - evaluation_lower

    print(
        "\nCovered evaluation engines:",
        f"{covered.sum()}/{len(covered)}",
    )
    print(f"Empirical coverage: {covered.mean():.1%}")
    print(
        f"Mean interval width: "
        f"{interval_widths.mean():.2f} cycles"
    )
    print(
        "Actual RUL below lower bound:",
        int((evaluation_actual < evaluation_lower).sum()),
    )
    print(
        "Actual RUL above upper bound:",
        int((evaluation_actual > evaluation_upper).sum()),
    )

    interval_results = pd.DataFrame({
        "unit_number": evaluation_data.loc[
            evaluation_indices, "unit_number"
        ].to_numpy(),
        "actual_rul": evaluation_actual,
        "predicted_rul": evaluation_predictions,
        "lower_bound": evaluation_lower,
        "upper_bound": evaluation_upper,
        "interval_width": interval_widths,
        "covered": covered,
    })

    print("\nEvaluation intervals, ordered by actual RUL:")
    print(
        interval_results
        .sort_values("actual_rul")
        .round(2)
        .to_string(index=False)
    )

    risk_threshold = 30.0

    interval_results["threshold_status"] = classify_threshold_status(
        interval_results["lower_bound"].to_numpy(),
        interval_results["upper_bound"].to_numpy(),
        threshold=risk_threshold,
    )

    print(f"\nInterval status at {risk_threshold:.0f} cycles:")
    print(
        interval_results["threshold_status"]
        .value_counts()
        .to_string()
    )

    flagged = interval_results["lower_bound"] <= risk_threshold
    actual_near_failure = (
        interval_results["actual_rul"] <= risk_threshold
    )

    true_positives = int((flagged & actual_near_failure).sum())
    false_positives = int((flagged & ~actual_near_failure).sum())
    false_negatives = int((~flagged & actual_near_failure).sum())
    true_negatives = int((~flagged & ~actual_near_failure).sum())

    print("\nLow-RUL flag evaluation:")
    print("Flagged and actually near failure:", true_positives)
    print("Flagged but above 30 cycles:", false_positives)
    print("Not flagged but actually near failure:", false_negatives)
    print("Not flagged and above 30 cycles:", true_negatives)


if __name__ == "__main__":
    main()