"""Prepare leakage-safe training and validation datasets."""

import  pandas as pd
from sklearn.pipeline import Pipeline

from src.data.build_targets import add_training_rul
from src.data.load_data import RUL_COLUMN
from src.data.split_data import split_by_engine
from src.data.validate_data import validate_cmapss_data
from src.features.build_features import build_features
from src.features.preprocess_features import create_feature_preprocessor

def prepare_training_validation_data(
        data: pd.DataFrame,
        validation_size: float= 0.2,
        random_state: int = 42,
        rolling_window: int = 5,
        variance_threshold: float= 0.0
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    Pipeline
]:
    """Prepare model-ready training and validation data without leakage."""
    validate_cmapss_data(data)

    labeled_data = add_training_rul(data)

    training_data, validation_data = split_by_engine(
        labeled_data,
        validation_size= validation_size,
        random_state=random_state
    )

    # creating y_train + y_test
    training_targets = training_data[RUL_COLUMN].copy()
    validation_targets = validation_data[RUL_COLUMN].copy()

    # x_train
    training_features = build_features(
        training_data,
        rolling_window=rolling_window
    )

    # x_test
    validation_features = build_features(
        validation_data,
        rolling_window=rolling_window
    )

    preprocessor = create_feature_preprocessor(
        variance_threshold=variance_threshold
    )

    transformed_training = preprocessor.fit_transform(
        training_features
    )

    transformed_validation = preprocessor.transform(
        validation_features
    )

    return (
        transformed_training,
        transformed_validation,
        training_targets,
        validation_targets,
        preprocessor
    )


def select_engine_endpoints(
        observed_data: pd.DataFrame,
        features: pd.DataFrame,
) -> pd.DataFrame:
    """Select each engine's last observed, already-built feature row."""
    required_columns = {"unit_number", "time_in_cycles"}

    if not required_columns.issubset(observed_data.columns):
        raise ValueError(
            "observed_data must contain unit_number and time_in_cycles"
        )

    if observed_data.empty:
        raise ValueError("observed_data must not be empty")

    if not observed_data.index.is_unique:
        raise ValueError("observed_data must have a unique index")

    if not observed_data.index.equals(features.index):
        raise ValueError(
            "observed_data and features must have matching indexes"
        )

    if observed_data[
        ["unit_number", "time_in_cycles"]
    ].isna().any().any():
        raise ValueError("Engine IDs and cycles must not be missing")

    if observed_data.duplicated(
        subset=["unit_number", "time_in_cycles"],
    ).any():
        raise ValueError("Duplicate engine-cycle observations found")

    endpoint_rows = (
        observed_data
        .sort_values(["unit_number", "time_in_cycles"])
        .groupby("unit_number", sort=False)
        .tail(1)
    )

    endpoint_features = features.loc[endpoint_rows.index].copy()

    endpoint_features.index = pd.Index(
        endpoint_rows["unit_number"].to_numpy(),
        name="unit_number",
    )

    return endpoint_features