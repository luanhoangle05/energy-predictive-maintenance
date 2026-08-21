"""Run the initial mean-baseline evaluation workflow."""

from pathlib import Path

from src.data.load_data import load_cmapss_file
from src.data.prepare_data import prepare_training_validation_data

from src.models.predict import predict_rul
from src.models.train_model import train_dummy_regressor
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

    model = train_dummy_regressor(training_features, training_targets)
    validation_predictions = predict_rul(
        model,
        validation_features
    )

    metrics = regression_metrics(
        validation_targets.to_numpy(),
        validation_predictions
    )



    print(f"Training rows: {len(training_features)}")
    print(f"Validation rows: {len(validation_features)}")
    print(f"Retained features: {training_features.shape[1]}")
    print("Prediction rows: ",len(validation_predictions))
    print("First five predictions: ",validation_predictions[:5])

    print(f"MAE: {metrics['mae']:.2f} cycles")
    print(f"RMSE: {metrics['rmse']:.2f} cycles")
    print(f"R²: {metrics['r2']:.4f}")

if __name__ == "__main__":
    main()