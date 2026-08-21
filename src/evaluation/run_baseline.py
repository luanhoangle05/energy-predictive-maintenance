"""Run FD001 baseline model comparison and grouped tuning."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.data.load_data import load_cmapss_file
from src.data.prepare_data import prepare_training_validation_data
from src.evaluation.evaluate_model import regression_metrics
from src.models.predict import predict_rul
from src.models.train_model import (
    train_decision_tree_regressor,
    train_dummy_regressor,
    train_linear_regression,
    train_random_forest_regressor,
)

from src.features.build_features import build_features

from src.features.preprocess_features import (
create_feature_preprocessor
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


def main() -> None:
    """Run all current FD001 development-baseline experiments."""
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
    print(f"Decision Tree depth: {decision_tree_model.get_depth()}")
    print(f"Decision Tree leaves: {decision_tree_model.get_n_leaves()}")

    # Controlled Decision Tree tuning
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

    selected_tree_model = train_decision_tree_regressor(
        training_features,
        training_targets,
        max_depth=4,
        min_samples_leaf=30,
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
    print(f"Depth: {selected_tree_model.get_depth()}")
    print(f"Leaves: {selected_tree_model.get_n_leaves()}")

    # Controlled Random Forest tuning
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

    selected_forest_model = train_random_forest_regressor(
        training_features,
        training_targets,
        n_estimators=100,
        max_depth=4,
        min_samples_leaf=30,
        max_features=1.0,
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


if __name__ == "__main__":
    main()
