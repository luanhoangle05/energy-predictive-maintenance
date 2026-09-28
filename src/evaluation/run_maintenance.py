"""Run the exploratory development maintenance simulation."""

from src.data.load_data import load_cmapss_file, RUL_COLUMN
from src.data.prepare_data import (
    prepare_uncertainty_splits,
    select_cutoff_indices,
)

from src.features.build_features import build_features
from src.features.preprocess_features import create_feature_preprocessor

from src.models.train_model import train_gradient_boosting_regressor

import numpy as np

from src.models.predict import predict_rul
from src.evaluation.uncertainty import (
    calibrate_interval_radius,
    build_prediction_intervals,
)

import pandas as pd

from src.evaluation.maintenance import (
    simulate_run_to_failure,
    simulate_fixed_age,
    simulate_rul_threshold_policy,
    simulate_lower_bound_policy,
)

from pathlib import Path


def main() -> None:
    raw_data = load_cmapss_file(
        "data/raw/cmapss/train_FD001.txt"
    )

    training_data, calibration_data, evaluation_data = (
        prepare_uncertainty_splits(
            raw_data,
            evaluation_size=0.2,
            calibration_size=0.2,
            random_state=42,
        )
    )

    for name, data in (
        ("Training", training_data),
        ("Calibration", calibration_data),
        ("Evaluation", evaluation_data),
    ):
        engine_count = data["unit_number"].nunique()
        print(f"{name}: {engine_count} engines, {len(data)} rows")

    training_features = build_features(training_data)
    calibration_features = build_features(calibration_data)
    evaluation_features = build_features(evaluation_data)

    preprocessor = create_feature_preprocessor()

    training_features = preprocessor.fit_transform(training_features)
    calibration_features = preprocessor.transform(calibration_features)
    evaluation_features = preprocessor.transform(evaluation_features)

    for name, features, data in (
        ("Training", training_features, training_data),
        ("Calibration", calibration_features, calibration_data),
        ("Evaluation", evaluation_features, evaluation_data),
    ):
        print(f"{name} feature shape: {features.shape}")
        print(f"{name} indexes aligned: {features.index.equals(data.index)}")

    training_targets = training_data[RUL_COLUMN].copy()

    print("\nFitting the frozen Gradient Boosting configuration...")

    model = train_gradient_boosting_regressor(
        training_features,
        training_targets,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=30,
        random_state=42,
    )

    print(f"Model fitted: {type(model).__name__}")

    calibration_indices = select_cutoff_indices(
        calibration_data,
        minimum_cycle=30,
        random_state=42,
    )

    calibration_snapshot_features = calibration_features.loc[
        calibration_indices
    ]
    calibration_targets = calibration_data.loc[
        calibration_indices, RUL_COLUMN
    ]

    calibration_predictions = np.maximum(
        predict_rul(model, calibration_snapshot_features),
        0.0,
    )

    interval_radius = calibrate_interval_radius(
        calibration_targets.to_numpy(),
        calibration_predictions,
        confidence=0.90,
    )

    print(f"Calibration snapshots: {len(calibration_indices)}")
    print(f"Interval radius: {interval_radius:.2f} cycles")

    evaluation_predictions = np.maximum(
        predict_rul(model, evaluation_features),
        0.0,
    )

    evaluation_lower, _ = build_prediction_intervals(
        evaluation_predictions,
        interval_radius,
    )

    evaluation_history = evaluation_data[
        ["unit_number", "time_in_cycles"]
    ].copy()

    evaluation_history["predicted_rul"] = evaluation_predictions
    evaluation_history["lower_bound"] = evaluation_lower

    print("\nEvaluation history shape:", evaluation_history.shape)
    print(
        "Evaluation engines:",
        evaluation_history["unit_number"].nunique(),
    )
    print(evaluation_history.head().round(2).to_string(index=False))

    all_outcomes = []

    for engine_id, engine_history in evaluation_history.groupby(
        "unit_number", sort=True
    ):
        engine_history = engine_history.sort_values("time_in_cycles")
        cycles = engine_history["time_in_cycles"].to_numpy()

        # Actual lifetime is used for retrospective scoring.
        failure_cycle = int(cycles[-1])

        outcomes = [
            simulate_run_to_failure(failure_cycle=failure_cycle),
            simulate_fixed_age(failure_cycle=failure_cycle),
            simulate_rul_threshold_policy(
                cycles=cycles,
                predicted_rul=engine_history["predicted_rul"].to_numpy(),
                failure_cycle=failure_cycle,
            ),
            simulate_lower_bound_policy(
                cycles=cycles,
                lower_bound=engine_history["lower_bound"].to_numpy(),
                failure_cycle=failure_cycle,
            ),
        ]

        for outcome in outcomes:
            all_outcomes.append({
                "unit_number": int(engine_id),
                **outcome,
            })

    maintenance_results = pd.DataFrame(all_outcomes)

    print("\nMaintenance outcome rows:", len(maintenance_results))
    print(
        "Duplicate engine-policy pairs:",
        maintenance_results.duplicated(
            ["unit_number", "policy"]
        ).sum(),
    )
    print("\nRows per policy:")
    print(maintenance_results.groupby("policy").size().to_string())

    print("\nFirst eight outcomes:")
    print(maintenance_results.head(8).to_string(index=False))

    timing_summary = maintenance_results.groupby("policy").agg(
        engines=("unit_number", "nunique"),
        preventive_successes=("preventive_maintenance_success", "sum"),
        failures=("ran_to_failure", "sum"),
        total_discarded_cycles=("discarded_useful_life", "sum"),
        mean_discarded_cycles_all_engines=("discarded_useful_life", "mean"),
    )

    successful_results = maintenance_results.loc[
        maintenance_results["preventive_maintenance_success"]
    ]

    timing_summary["mean_discarded_cycles_successes"] = (
        successful_results.groupby("policy")["discarded_useful_life"].mean()
    )

    print("\nExploratory timing summary:")
    print(timing_summary.round(2).to_string())

    output_directory = Path("reports/maintenance")
    output_directory.mkdir(parents=True, exist_ok=True)

    maintenance_results.to_csv(
        output_directory / "engine_timing_outcomes.csv",
        index=False,
    )

    timing_summary.to_csv(
        output_directory / "timing_summary.csv",
        index=True,
        index_label="policy",
    )

    print(f"\nTiming results saved to: {output_directory.resolve()}")


if __name__ == "__main__":
    main()