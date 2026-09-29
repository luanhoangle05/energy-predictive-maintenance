"""Versioned development model artifacts and shared serving transformations."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.data.load_data import CMAPSS_COLUMNS
from src.data.validate_data import validate_cmapss_data
from src.features.build_features import build_features
from src.evaluation.uncertainty import (
    build_prediction_intervals, classify_threshold_status,
)

DISCLAIMER = (
    "Exploratory FD001 simulation only. Intervals are not operational guarantees "
    "and threshold flags are not maintenance recommendations."
)


def validate_history(data: pd.DataFrame) -> pd.DataFrame:
    """Require complete, ordered histories from cycle 1, with raw columns only."""
    if not data.columns.is_unique or set(data.columns) != set(CMAPSS_COLUMNS):
        raise ValueError("Provide exactly the 26 raw C-MAPSS columns; omit targets.")
    data = data.loc[:, CMAPSS_COLUMNS].copy().reset_index(drop=True)
    if any(not pd.api.types.is_numeric_dtype(t) or pd.api.types.is_bool_dtype(t)
           for t in data.dtypes):
        raise ValueError("All observations must be numeric, not strings or booleans.")
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("All observations must be finite.")
    for column in ("unit_number", "time_in_cycles"):
        values = data[column]
        if ((values < 1) | (values > 2**31 - 1) | (values % 1 != 0)).any():
            raise ValueError(f"{column} must contain positive 32-bit integers.")
        data[column] = values.astype(int)
    validate_cmapss_data(data)
    return data


@dataclass
class ModelBundle:
    preprocessor: object
    model: object
    radius: float
    metadata: dict
    rolling_window: int = 5
    schema_version: int = 1

    def predict(self, history: pd.DataFrame) -> dict:
        history = validate_history(history)
        features = build_features(history, rolling_window=self.rolling_window)
        endpoints = history.groupby("unit_number", sort=True).tail(1).sort_values("unit_number")
        transformed = self.preprocessor.transform(features.loc[endpoints.index])
        predicted = np.maximum(self.model.predict(transformed), 0.0)
        lower, upper = build_prediction_intervals(predicted, self.radius)
        status = classify_threshold_status(lower, upper, threshold=30.0)
        results = pd.DataFrame({
            "unit_number": endpoints.unit_number.to_numpy(),
            "time_in_cycles": endpoints.time_in_cycles.to_numpy(),
            "predicted_rul": predicted, "lower_bound": lower, "upper_bound": upper,
            "threshold_status": status, "low_rul_flag": lower <= 30.0,
        })
        return {"predictions": results.to_dict(orient="records"),
                "model_id": self.metadata["model_id"], "disclaimer": DISCLAIMER}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        joblib.dump(self, temporary)
        temporary.replace(path)


def load_bundle(path: str | Path) -> ModelBundle:
    """Load a trusted local artifact only: joblib files can execute code."""
    bundle = joblib.load(path)
    if not isinstance(bundle, ModelBundle) or bundle.schema_version != 1:
        raise ValueError("Unsupported model artifact schema.")
    if bundle.metadata.get("sklearn_version") != sklearn.__version__:
        raise ValueError("Artifact scikit-learn version differs from serving environment.")
    if not np.isfinite(bundle.radius) or bundle.radius < 0:
        raise ValueError("Invalid calibration radius in artifact.")
    return bundle
