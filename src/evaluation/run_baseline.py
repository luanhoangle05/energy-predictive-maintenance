"""Run the initial mean-baseline evaluation workflow."""

from pathlib import Path

import numpy as np

from src.data.load_data import load_cmapss_file
from src.data.prepare_data import prepare_training_validation_data

from src.models.predict import predict_rul
from src.models.train_model import train_dummy_regressor,train_linear_regression
from src.evaluation.evaluate_model import regression_metrics

def main() -> None:
    data_path = Path("data/raw/cmapss/train_FD001.txt")

    raw_data = load_cmapss_file(data_path)

    (
        training_features, # X_train
        validation_features, # X_test
        training_targets, # Y_train
        validation_targets, # Y_test
        preprocessor
    ) = prepare_training_validation_data(raw_data)

    dummy_model = train_dummy_regressor(training_features, training_targets)

    dummy_predictions = predict_rul(
        dummy_model,
        validation_features
    )

    dummy_metrics = regression_metrics(
        validation_targets.to_numpy(),
        dummy_predictions
    )

    linear_model = train_linear_regression(training_features,training_targets)

    linear_predictions = predict_rul(
        linear_model,
        validation_features
    )

    # Clipping at zero is based on the target’s domain—
    # RUL cannot be below zero—not on a threshold tuned to improve validation performance.
    linear_clipped_predictions = np.clip(
        linear_predictions,
        a_min=0.0,
        a_max=None,
    )

    linear_clipped_metrics = regression_metrics(
        validation_targets.to_numpy(),
        linear_clipped_predictions,
    )

    linear_metrics = regression_metrics(
        validation_targets.to_numpy(),
        linear_predictions
    )



    print(f"Training rows: {len(training_features)}")
    print(f"Validation rows: {len(validation_features)}")
    print(f"Retained features: {training_features.shape[1]}")
    print("Prediction rows: ",len(dummy_predictions))
    print("First five predictions: ",dummy_predictions[:5])

    print("\nDummy mean baseline")
    print(f"MAE: {dummy_metrics['mae']:.2f} cycles")
    print(f"RMSE: {dummy_metrics['rmse']:.2f} cycles")
    print(f"R²: {dummy_metrics['r2']:.4f}")

    print("\nRaw Linear Regression")
    print(f"MAE: {linear_metrics['mae']:.2f} cycles")
    print(f"RMSE: {linear_metrics['rmse']:.2f} cycles")
    print(f"R²: {linear_metrics['r2']:.4f}")

    print("\nLinear Regression clipped at zero")
    print(f"MAE: {linear_clipped_metrics['mae']:.2f} cycles")
    print(f"RMSE: {linear_clipped_metrics['rmse']:.2f} cycles")
    print(f"R²: {linear_clipped_metrics['r2']:.4f}")

if __name__ == "__main__":
    main()