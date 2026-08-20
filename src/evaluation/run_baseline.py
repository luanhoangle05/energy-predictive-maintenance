"""Run the initial mean-baseline evaluation workflow."""

from pathlib import Path

from src.data.load_data import load_cmapss_file
from src.data.prepare_data import prepare_training_validation_data


def main() -> None:
    data_path = Path("data/raw/cmapss/train_FD001.txt")

    raw_data = load_cmapss_file(data_path)

    (
        training_features,
        validation_features,
        training_targets,
        validation_targets,
        preprocessor
    ) = prepare_training_validation_data(raw_data)

    print(f"Training rows: {len(training_features)}")
    print(f"Validation rows: {len(validation_features)}")
    print(f"Retained features: {training_features.shape[1]}")

if __name__ == "__main__":
    main()