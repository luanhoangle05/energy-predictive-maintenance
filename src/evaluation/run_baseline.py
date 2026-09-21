"""Run FD001 baseline model comparison and grouped tuning."""

from pathlib import Path

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.data.load_data import load_cmapss_file
from src.data.prepare_data import prepare_training_validation_data
from src.evaluation.evaluate_model import (regression_metrics,
    near_failure_metrics,
    engine_level_metrics,
    rul_band_metrics,
    feature_importance_table,
    summarize_models_by_engine)
from src.models.predict import predict_rul
from src.models.train_model import (
    train_decision_tree_regressor,
    train_dummy_regressor,
    train_linear_regression,
    train_random_forest_regressor,
train_gradient_boosting_regressor
)

from src.features.build_features import build_features

from src.features.preprocess_features import (
create_feature_preprocessor
)

from src.evaluation.plot_diagnostics import (
plot_engine_prediction_trajectory,
plot_residuals_by_actual_rul,
plot_feature_importances

)


def cross_validate_decision_tree(
    features: pd.DataFrame,
    targets: pd.Series,
    engine_groups: pd.Series,
    max_depth: int,
    min_samples_leaf: int,
) -> dict[str, float]:
    """Evaluate one Decision Tree configuration using grouped folds."""
    group_kfold = GroupKFold(n_splits=5)
    fold_mae_scores = []
    fold_rmse_scores = []

    for (
        fold_training_indices,
        fold_validation_indices,
    ) in group_kfold.split(
        features,
        targets,
        groups=engine_groups,
    ):
        fold_training_features_before_preprocessing = features.iloc[
            fold_training_indices
        ]
        fold_validation_features_before_preprocessing = features.iloc[
            fold_validation_indices
        ]

        fold_preprocessor = create_feature_preprocessor()

        fold_training_features = fold_preprocessor.fit_transform(
            fold_training_features_before_preprocessing
        )

        fold_validation_features = fold_preprocessor.transform(
            fold_validation_features_before_preprocessing
        )
        fold_training_targets = targets.iloc[
            fold_training_indices
        ]
        fold_validation_targets = targets.iloc[
            fold_validation_indices
        ]

        fold_model = train_decision_tree_regressor(
            fold_training_features,
            fold_training_targets,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        )
        fold_predictions = predict_rul(
            fold_model,
            fold_validation_features,
        )
        fold_metrics = regression_metrics(
            fold_validation_targets.to_numpy(),
            fold_predictions,
        )

        fold_mae_scores.append(fold_metrics["mae"])
        fold_rmse_scores.append(fold_metrics["rmse"])

    return {
        "mean_mae": float(np.mean(fold_mae_scores)),
        "mean_rmse": float(np.mean(fold_rmse_scores)),
    }


def cross_validate_random_forest(
    features: pd.DataFrame,
    targets: pd.Series,
    engine_groups: pd.Series,
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    max_features: float,
) -> dict[str, float]:
    """Evaluate one Random Forest configuration using grouped folds."""
    group_kfold = GroupKFold(n_splits=5)
    fold_mae_scores = []
    fold_rmse_scores = []

    for (
        fold_training_indices,
        fold_validation_indices,
    ) in group_kfold.split(
        features,
        targets,
        groups=engine_groups,
    ):
        fold_training_features_before_preprocessing = features.iloc[
            fold_training_indices
        ]
        fold_validation_features_before_preprocessing = features.iloc[
            fold_validation_indices
        ]

        fold_preprocessor = create_feature_preprocessor()

        fold_training_features = fold_preprocessor.fit_transform(
            fold_training_features_before_preprocessing
        )

        fold_validation_features = fold_preprocessor.transform(
            fold_validation_features_before_preprocessing
        )
        fold_training_targets = targets.iloc[
            fold_training_indices
        ]
        fold_validation_targets = targets.iloc[
            fold_validation_indices
        ]

        fold_model = train_random_forest_regressor(
            fold_training_features,
            fold_training_targets,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
        )
        fold_predictions = predict_rul(
            fold_model,
            fold_validation_features,
        )
        fold_metrics = regression_metrics(
            fold_validation_targets.to_numpy(),
            fold_predictions,
        )

        fold_mae_scores.append(fold_metrics["mae"])
        fold_rmse_scores.append(fold_metrics["rmse"])

    return {
        "mean_mae": float(np.mean(fold_mae_scores)),
        "mean_rmse": float(np.mean(fold_rmse_scores)),
    }

def cross_validate_gradient_boosting(
    features: pd.DataFrame,
    targets: pd.Series,
    engine_groups: pd.Series,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    min_samples_leaf: int,
) -> dict[str, float]:
    """Evaluate one Gradient Boosting configuration using grouped folds."""
    group_kfold = GroupKFold(n_splits=5)
    fold_mae_scores = []
    fold_rmse_scores = []

    for (
        fold_training_indices,
        fold_validation_indices,
    ) in group_kfold.split(
        features,
        targets,
        groups=engine_groups,
    ):
        fold_training_features_before_preprocessing = features.iloc[
            fold_training_indices
        ]
        fold_validation_features_before_preprocessing = features.iloc[
            fold_validation_indices
        ]

        # Fit preprocessing only on this fold's training engines.
        fold_preprocessor = create_feature_preprocessor()

        fold_training_features = fold_preprocessor.fit_transform(
            fold_training_features_before_preprocessing
        )
        fold_validation_features = fold_preprocessor.transform(
            fold_validation_features_before_preprocessing
        )

        fold_training_targets = targets.iloc[
            fold_training_indices
        ]
        fold_validation_targets = targets.iloc[
            fold_validation_indices
        ]

        fold_model = train_gradient_boosting_regressor(
            fold_training_features,
            fold_training_targets,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        )

        fold_predictions = predict_rul(
            fold_model,
            fold_validation_features,
        )
        fold_metrics = regression_metrics(
            fold_validation_targets.to_numpy(),
            fold_predictions,
        )

        fold_mae_scores.append(fold_metrics["mae"])
        fold_rmse_scores.append(fold_metrics["rmse"])

    return {
        "mean_mae": float(np.mean(fold_mae_scores)),
        "mean_rmse": float(np.mean(fold_rmse_scores)),
    }

def main(tune: bool = False) -> None:
    """Train and evaluate models, optionally running grouped tuning."""
    print(
        "Mode: full grouped tuning"
        if tune
        else "Mode: selected settings only (skipping cross-validation)"
    )

    data_path = Path("data/raw/cmapss/train_FD001.txt")
    raw_data = load_cmapss_file(data_path)

    (
        training_features,  # X_train
        validation_features,  # X_test
        training_targets,  # Y_train
        validation_targets,  # Y_test
        preprocessor,
    ) = prepare_training_validation_data(raw_data)

    training_source_data = raw_data.loc[
        training_features.index
    ].copy()

    training_features_before_preprocessing = build_features(
        training_source_data,
        rolling_window=5,
    )


    training_engine_groups = raw_data.loc[
        training_features.index,
        "unit_number",
    ]
    validation_engine_groups = raw_data.loc[
        validation_features.index,
        "unit_number",
    ]
    training_engine_ids = set(training_engine_groups.unique())
    validation_engine_ids = set(validation_engine_groups.unique())
    overlapping_engine_ids = (
        training_engine_ids & validation_engine_ids
    )

    print(
        "CV features before preprocessing: "
        f"{training_features_before_preprocessing.shape[1]}"
    )
    print(
        "CV feature indexes aligned: "
        f"{training_features_before_preprocessing.index.equals(training_targets.index)}"
    )
    print(
        "CV group indexes aligned: "
        f"{training_features_before_preprocessing.index.equals(training_engine_groups.index)}"
    )

    print(f"Training engines: {len(training_engine_ids)}")
    print(f"Validation engines: {len(validation_engine_ids)}")
    print(f"Overlapping engines: {len(overlapping_engine_ids)}")
    print(f"Training rows: {len(training_features)}")
    print(f"Validation rows: {len(validation_features)}")
    print(f"Retained features: {training_features.shape[1]}")

    # Mean DummyRegressor baseline
    dummy_model = train_dummy_regressor(
        training_features,
        training_targets,
    )
    dummy_predictions = predict_rul(
        dummy_model,
        validation_features,
    )
    dummy_metrics = regression_metrics(
        validation_targets.to_numpy(),
        dummy_predictions,
    )

    print(f"Prediction rows: {len(dummy_predictions)}")
    print(f"First five predictions: {dummy_predictions[:5]}")
    print("\nDummy mean baseline")
    print(f"MAE: {dummy_metrics['mae']:.2f} cycles")
    print(f"RMSE: {dummy_metrics['rmse']:.2f} cycles")
    print(f"R²: {dummy_metrics['r2']:.4f}")
    print(f"NASA score: {dummy_metrics['nasa_score']:.2f}")

    # Linear Regression baseline
    linear_model = train_linear_regression(
        training_features,
        training_targets,
    )
    linear_predictions = predict_rul(
        linear_model,
        validation_features,
    )
    linear_metrics = regression_metrics(
        validation_targets.to_numpy(),
        linear_predictions,
    )

    # Clipping at zero is based on the target’s domain—
    # RUL cannot be below zero—not on a threshold tuned to improve
    # validation performance.
    linear_clipped_predictions = np.clip(
        linear_predictions,
        a_min=0.0,
        a_max=None,
    )
    linear_clipped_metrics = regression_metrics(
        validation_targets.to_numpy(),
        linear_clipped_predictions,
    )

    print("\nRaw Linear Regression")
    print(f"MAE: {linear_metrics['mae']:.2f} cycles")
    print(f"RMSE: {linear_metrics['rmse']:.2f} cycles")
    print(f"R²: {linear_metrics['r2']:.4f}")
    print("\nLinear Regression clipped at zero")
    print(f"MAE: {linear_clipped_metrics['mae']:.2f} cycles")
    print(f"RMSE: {linear_clipped_metrics['rmse']:.2f} cycles")
    print(f"R²: {linear_clipped_metrics['r2']:.4f}")
    print(f"NASA score: {linear_clipped_metrics['nasa_score']:.2f}")

    # Unrestricted Decision Tree baseline
    decision_tree_model = train_decision_tree_regressor(
        training_features,
        training_targets,
    )
    tree_training_predictions = predict_rul(
        decision_tree_model,
        training_features,
    )
    tree_validation_predictions = predict_rul(
        decision_tree_model,
        validation_features,
    )
    tree_training_metrics = regression_metrics(
        training_targets.to_numpy(),
        tree_training_predictions,
    )
    tree_validation_metrics = regression_metrics(
        validation_targets.to_numpy(),
        tree_validation_predictions,
    )

    print("\nUnrestricted Decision Tree — training")
    print(f"MAE: {tree_training_metrics['mae']:.2f} cycles")
    print(f"RMSE: {tree_training_metrics['rmse']:.2f} cycles")
    print(f"R²: {tree_training_metrics['r2']:.4f}")
    print("\nUnrestricted Decision Tree — validation")
    print(f"MAE: {tree_validation_metrics['mae']:.2f} cycles")
    print(f"RMSE: {tree_validation_metrics['rmse']:.2f} cycles")
    print(f"R²: {tree_validation_metrics['r2']:.4f}")
    print(f"NASA score: {tree_validation_metrics['nasa_score']:.2f}")
    print(f"Decision Tree depth: {decision_tree_model.get_depth()}")
    print(f"Decision Tree leaves: {decision_tree_model.get_n_leaves()}")

    # Controlled Decision Tree tuning

    tree_params = {
        "max_depth": 4,
        "min_samples_leaf": 30,
    }

    if tune:
        max_depth_values = [4, 8, 12]
        min_samples_leaf_values = [1, 10, 30]
        tree_tuning_results = []

        print("\nDecision Tree grouped cross-validation")

        for max_depth in max_depth_values:
            for min_samples_leaf in min_samples_leaf_values:
                cv_metrics = cross_validate_decision_tree(
                    training_features_before_preprocessing,
                    training_targets,
                    training_engine_groups,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                )

                result = {
                    "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf,
                    "mean_mae": cv_metrics["mean_mae"],
                    "mean_rmse": cv_metrics["mean_rmse"],
                }
                tree_tuning_results.append(result)

                print(
                    f"max_depth={max_depth}, "
                    f"min_samples_leaf={min_samples_leaf}: "
                    f"MAE={result['mean_mae']:.2f}, "
                    f"RMSE={result['mean_rmse']:.2f}"
                )

        best_tree_result = min(
            tree_tuning_results,
            key=lambda result: (
                result["mean_mae"],
                result["mean_rmse"],
            ),
        )

        tree_params = {
            "max_depth": best_tree_result["max_depth"],
            "min_samples_leaf": best_tree_result["min_samples_leaf"],
        }

        print(f"Best Decision Tree CV result: {best_tree_result}")

    selected_tree_model = train_decision_tree_regressor(
        training_features,
        training_targets,
        **tree_params,
    )

    selected_tree_training_predictions = predict_rul(
        selected_tree_model,
        training_features,
    )
    selected_tree_validation_predictions = predict_rul(
        selected_tree_model,
        validation_features,
    )
    selected_tree_training_metrics = regression_metrics(
        training_targets.to_numpy(),
        selected_tree_training_predictions,
    )
    selected_tree_validation_metrics = regression_metrics(
        validation_targets.to_numpy(),
        selected_tree_validation_predictions,
    )

    print("\nSelected Decision Tree — training")
    print(f"MAE: {selected_tree_training_metrics['mae']:.2f} cycles")
    print(f"RMSE: {selected_tree_training_metrics['rmse']:.2f} cycles")
    print(f"R²: {selected_tree_training_metrics['r2']:.4f}")
    print("\nSelected Decision Tree — validation")
    print(f"MAE: {selected_tree_validation_metrics['mae']:.2f} cycles")
    print(f"RMSE: {selected_tree_validation_metrics['rmse']:.2f} cycles")
    print(f"R²: {selected_tree_validation_metrics['r2']:.4f}")
    print(f"NASA score: {selected_tree_validation_metrics['nasa_score']:.2f}")
    print(f"Depth: {selected_tree_model.get_depth()}")
    print(f"Leaves: {selected_tree_model.get_n_leaves()}")

    # Controlled Random Forest tuning

    forest_params = {
        "n_estimators": 100,
        "max_depth": 4,
        "min_samples_leaf": 30,
        "max_features": 1.0,
    }

    if tune:
        forest_max_depth_values = [4, 8, None]
        forest_min_samples_leaf_values = [1, 10, 30]
        forest_tuning_results = []

        print("\nRandom Forest grouped cross-validation")
        for max_depth in forest_max_depth_values:
            for min_samples_leaf in forest_min_samples_leaf_values:
                cv_metrics = cross_validate_random_forest(
                    training_features_before_preprocessing,
                    training_targets,
                    training_engine_groups,
                    n_estimators=100,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_features=1.0,
                )
                result = {
                    "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf,
                    "mean_mae": cv_metrics["mean_mae"],
                    "mean_rmse": cv_metrics["mean_rmse"],
                }
                forest_tuning_results.append(result)

                print(
                    f"max_depth={max_depth}, "
                    f"min_samples_leaf={min_samples_leaf}: "
                    f"MAE={result['mean_mae']:.2f}, "
                    f"RMSE={result['mean_rmse']:.2f}"
                )

        best_forest_result = min(
            forest_tuning_results,
            key=lambda result: (
                result["mean_mae"],
                result["mean_rmse"],
            ),
        )

        forest_params.update(
            {
                "max_depth": best_forest_result["max_depth"],
                "min_samples_leaf": best_forest_result["min_samples_leaf"],
            }
        )

        print(f"Best Random Forest CV result: {best_forest_result}")

    selected_forest_model = train_random_forest_regressor(
        training_features,
        training_targets,
        **forest_params,
    )

    selected_forest_training_predictions = predict_rul(
        selected_forest_model,
        training_features,
    )
    selected_forest_validation_predictions = predict_rul(
        selected_forest_model,
        validation_features,
    )
    selected_forest_training_metrics = regression_metrics(
        training_targets.to_numpy(),
        selected_forest_training_predictions,
    )
    selected_forest_validation_metrics = regression_metrics(
        validation_targets.to_numpy(),
        selected_forest_validation_predictions,
    )

    print("\nSelected Random Forest — training")
    print(f"MAE: {selected_forest_training_metrics['mae']:.2f} cycles")
    print(f"RMSE: {selected_forest_training_metrics['rmse']:.2f} cycles")
    print(f"R²: {selected_forest_training_metrics['r2']:.4f}")
    print("\nSelected Random Forest — validation")
    print(f"MAE: {selected_forest_validation_metrics['mae']:.2f} cycles")
    print(f"RMSE: {selected_forest_validation_metrics['rmse']:.2f} cycles")
    print(f"R²: {selected_forest_validation_metrics['r2']:.4f}")
    print(f"NASA score: {selected_forest_validation_metrics['nasa_score']:.2f}")

    # Initial Gradient Boosting grouped cross-validation

    boosting_params = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 2,
        "min_samples_leaf": 30,
    }

    if tune:
        initial_gradient_boosting_cv_metrics = (
            cross_validate_gradient_boosting(
                training_features_before_preprocessing,
                training_targets,
                training_engine_groups,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                min_samples_leaf=1,
            )
        )

        print("\nInitial Gradient Boosting grouped cross-validation")
        print(
            "n_estimators=100, learning_rate=0.1, "
            "max_depth=3, min_samples_leaf=1"
        )
        print(
            "Mean CV MAE: "
            f"{initial_gradient_boosting_cv_metrics['mean_mae']:.2f} cycles"
        )
        print(
            "Mean CV RMSE: "
            f"{initial_gradient_boosting_cv_metrics['mean_rmse']:.2f} cycles"
        )

        # Controlled Gradient Boosting tuning
        boosting_schedules = [
            (100, 0.1),
            (200, 0.05),
        ]
        boosting_max_depth_values = [2, 3]
        boosting_min_samples_leaf_values = [1, 30]
        boosting_tuning_results = []

        print("\nGradient Boosting grouped cross-validation")

        for n_estimators, learning_rate in boosting_schedules:
            for max_depth in boosting_max_depth_values:
                for min_samples_leaf in boosting_min_samples_leaf_values:

                    is_initial_configuration = (
                        n_estimators == 100
                        and learning_rate == 0.1
                        and max_depth == 3
                        and min_samples_leaf == 1
                    )

                    if is_initial_configuration:
                        cv_metrics = initial_gradient_boosting_cv_metrics
                    else:
                        cv_metrics = cross_validate_gradient_boosting(
                            training_features_before_preprocessing,
                            training_targets,
                            training_engine_groups,
                            n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            max_depth=max_depth,
                            min_samples_leaf=min_samples_leaf,
                        )

                    result = {
                        "n_estimators": n_estimators,
                        "learning_rate": learning_rate,
                        "max_depth": max_depth,
                        "min_samples_leaf": min_samples_leaf,
                        "mean_mae": cv_metrics["mean_mae"],
                        "mean_rmse": cv_metrics["mean_rmse"],
                    }
                    boosting_tuning_results.append(result)

                    print(
                        f"n_estimators={n_estimators}, "
                        f"learning_rate={learning_rate}, "
                        f"max_depth={max_depth}, "
                        f"min_samples_leaf={min_samples_leaf}: "
                        f"MAE={result['mean_mae']:.2f}, "
                        f"RMSE={result['mean_rmse']:.2f}"
                    )

        best_boosting_result = min(
            boosting_tuning_results,
            key=lambda result: (
                result["mean_mae"],
                result["mean_rmse"],
            ),
        )

        boosting_params = {
            "n_estimators": best_boosting_result["n_estimators"],
            "learning_rate": best_boosting_result["learning_rate"],
            "max_depth": best_boosting_result["max_depth"],
            "min_samples_leaf": best_boosting_result["min_samples_leaf"],
        }

        print(f"Best Gradient Boosting CV result: {best_boosting_result}")

        # Fit the selected Gradient Boosting configuration

    selected_boosting_model = train_gradient_boosting_regressor(
            training_features,
            training_targets,
            **boosting_params,
        )

    raw_boosting_training_predictions = predict_rul(
        selected_boosting_model,
        training_features,
    )
    raw_boosting_validation_predictions = predict_rul(
        selected_boosting_model,
        validation_features,
    )

    # RUL cannot be negative, so enforce the physical lower bound.
    selected_boosting_training_predictions = np.clip(
        raw_boosting_training_predictions,
        a_min=0.0,
        a_max=None,
    )
    selected_boosting_validation_predictions = np.clip(
        raw_boosting_validation_predictions,
        a_min=0.0,
        a_max=None,
    )

    selected_boosting_training_metrics = regression_metrics(
        training_targets.to_numpy(),
        selected_boosting_training_predictions,
    )
    selected_boosting_validation_metrics = regression_metrics(
        validation_targets.to_numpy(),
        selected_boosting_validation_predictions,
    )

    print("\nSelected Gradient Boosting — prediction diagnostics")
    print(
        "Minimum raw validation prediction: "
        f"{raw_boosting_validation_predictions.min():.2f} cycles"
    )
    print(
        "Negative raw validation predictions: "
        f"{np.sum(raw_boosting_validation_predictions < 0.0)}"
    )

    print("\nSelected Gradient Boosting — training")
    print(
        f"MAE: {selected_boosting_training_metrics['mae']:.2f} cycles"
    )
    print(
        f"RMSE: {selected_boosting_training_metrics['rmse']:.2f} cycles"
    )
    print(f"R²: {selected_boosting_training_metrics['r2']:.4f}")

    print("\nSelected Gradient Boosting — validation")
    print(
        f"MAE: {selected_boosting_validation_metrics['mae']:.2f} cycles"
    )
    print(
        f"RMSE: {selected_boosting_validation_metrics['rmse']:.2f} cycles"
    )
    print(f"R²: {selected_boosting_validation_metrics['r2']:.4f}")
    print(f"NASA score: {selected_boosting_validation_metrics['nasa_score']:.2f}")

    near_failure_predictions = {
        "Mean baseline": dummy_predictions,
        "Clipped Linear Regression": linear_clipped_predictions,
        "Selected Decision Tree": selected_tree_validation_predictions,
        "Selected Random Forest": selected_forest_validation_predictions,
        "Selected Gradient Boosting": selected_boosting_validation_predictions,
    }

    print("\nNear-failure comparison (true RUL <= 30 cycles)")

    for model_name, predictions in near_failure_predictions.items():
        metrics = near_failure_metrics(
            validation_targets.to_numpy(),
            predictions,
            maximum_rul=30.0,
        )

        print(
            f"{model_name}: "
            f"rows={metrics['sample_count']}, "
            f"MAE={metrics['mae']:.2f}, "
            f"RMSE={metrics['rmse']:.2f}, "
            f"R²={metrics['r2']:.4f}, "
            f"NASA score={metrics['nasa_score']:.2f}"
        )

    boosting_engine_results = engine_level_metrics(
        validation_targets.to_numpy(),
        selected_boosting_validation_predictions,
        validation_engine_groups.to_numpy(),
    )

    worst_boosting_engines = (
        boosting_engine_results
        .sort_values(
            "mean_nasa_penalty",
            ascending=False,
        )
        .head(5)
        .loc[
            :,
            [
                "unit_number",
                "sample_count",
                "mae",
                "rmse",
                "mean_error",
                "overestimation_rate",
                "mean_nasa_penalty",
            ],
        ]
        .round(2)
    )

    print(
        "\nWorst Gradient Boosting validation engines "
        "by mean NASA penalty"
    )
    print(worst_boosting_engines.to_string(index=False))


    validation_cycles = raw_data.loc[
        validation_features.index,
        "time_in_cycles",
    ].to_numpy()

    validation_actual = validation_targets.to_numpy()
    validation_engine_array = validation_engine_groups.to_numpy()

    for engine_id in (45, 84):
        engine_mask = validation_engine_array == engine_id

        figure_path = (
            Path("reports/figures")
            / f"gradient_boosting_engine_{engine_id}_trajectory.png"
        )

        plot_engine_prediction_trajectory(
            cycles=validation_cycles[engine_mask],
            actual=validation_actual[engine_mask],
            predicted=selected_boosting_validation_predictions[engine_mask],
            engine_id=engine_id,
            model_name="Selected Gradient Boosting",
            output_path=figure_path,
        )

        print(f"Saved trajectory plot: {figure_path}")

    residual_figure_path = (
        Path("reports/figures")
        / "gradient_boosting_residuals_by_actual_rul.png"
    )

    plot_residuals_by_actual_rul(
        actual=validation_actual,
        predicted=selected_boosting_validation_predictions,
        model_name="Selected Gradient Boosting",
        output_path=residual_figure_path,
    )

    print(f"Saved residual plot: {residual_figure_path}")


    # Divided RUL into 4 bands:
    # 0-30:nearly failure
    # 30-120: approaching failure
    # >120: still too high RUL
    boosting_rul_band_results = rul_band_metrics(
        validation_actual,
        selected_boosting_validation_predictions,
    )

    boosting_rul_band_table = (
        boosting_rul_band_results.loc[
            :,
            [
                "rul_band",
                "sample_count",
                "mae",
                "rmse",
                "r2",
                "mean_error",
                "overestimation_rate",
                "mean_nasa_penalty",
            ],
        ]
        .round(2)
    )

    print("\nGradient Boosting performance by RUL band")
    print(boosting_rul_band_table.to_string(index=False))


    # Feature importance table
    # To check which features Gradient Boosting rely mostly on
    boosting_feature_importance = feature_importance_table(
        feature_names=training_features.columns,
        importances=selected_boosting_model.feature_importances_,
    )

    top_boosting_features = (
        boosting_feature_importance
        .head(15)
        .round({
            "importance": 4,
            "cumulative_importance": 4,
        })
    )

    print("\nTop 15 Gradient Boosting features")
    print(top_boosting_features.to_string(index=False))

    print(
        "\nTotal feature importance: "
        f"{boosting_feature_importance['importance'].sum():.4f}"
    )

    feature_importance_figure_path = (
        Path("reports/figures")
        / "gradient_boosting_feature_importance.png"
    )

    plot_feature_importances(
        feature_importance=boosting_feature_importance,
        model_name="Selected Gradient Boosting",
        output_path=feature_importance_figure_path,
        top_n=15,
    )

    print(
        "Saved feature-importance plot: "
        f"{feature_importance_figure_path}"
    )


    # Ablation test that drops engine time_in_cycles to control the experiment
    ablated_training_features = training_features.drop(
        columns=["time_in_cycles"],
    )
    ablated_validation_features = validation_features.drop(
        columns=["time_in_cycles"],
    )

    ablated_boosting_model = train_gradient_boosting_regressor(
        ablated_training_features,
        training_targets,
        **boosting_params,
    )

    raw_ablated_predictions = predict_rul(
        ablated_boosting_model,
        ablated_validation_features,
    )

    ablated_predictions = np.clip(
        raw_ablated_predictions,
        a_min=0.0,
        a_max=None,
    )

    ablated_metrics = regression_metrics(
        validation_actual,
        ablated_predictions,
    )

    full_near_failure_metrics = near_failure_metrics(
        validation_actual,
        selected_boosting_validation_predictions,
        maximum_rul=30.0,
    )
    ablated_near_failure_metrics = near_failure_metrics(
        validation_actual,
        ablated_predictions,
        maximum_rul=30.0,
    )

    print("\nGradient Boosting time-in-cycles ablation")
    print(
        "Full model: "
        f"MAE={selected_boosting_validation_metrics['mae']:.2f}, "
        f"RMSE={selected_boosting_validation_metrics['rmse']:.2f}, "
        f"R²={selected_boosting_validation_metrics['r2']:.4f}, "
        f"NASA score="
        f"{selected_boosting_validation_metrics['nasa_score']:.2f}"
    )
    print(
        "Without time_in_cycles: "
        f"MAE={ablated_metrics['mae']:.2f}, "
        f"RMSE={ablated_metrics['rmse']:.2f}, "
        f"R²={ablated_metrics['r2']:.4f}, "
        f"NASA score={ablated_metrics['nasa_score']:.2f}"
    )
    print(
        "Full model near failure: "
        f"MAE={full_near_failure_metrics['mae']:.2f}, "
        f"RMSE={full_near_failure_metrics['rmse']:.2f}, "
        f"NASA score={full_near_failure_metrics['nasa_score']:.2f}"
    )
    print(
        "Without time_in_cycles near failure: "
        f"MAE={ablated_near_failure_metrics['mae']:.2f}, "
        f"RMSE={ablated_near_failure_metrics['rmse']:.2f}, "
        f"NASA score={ablated_near_failure_metrics['nasa_score']:.2f}"
    )

    engine_summary = summarize_models_by_engine(
        actual=validation_actual,
        predictions_by_model=near_failure_predictions,
        engine_ids=validation_engine_array,
    )

    print("\nModel comparison with equal weight per engine")
    print(engine_summary.round(2).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate FD001 regression models."
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run engine-grouped cross-validation parameter searches.",
    )
    args = parser.parse_args()

    main(tune=args.tune)

