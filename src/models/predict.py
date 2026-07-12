"""Generate Remaining Useful Life predictions."""

from typing import Any

import numpy as np
import pandas as pd


def predict_rul(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Use a fitted model to predict RUL for each feature row."""
    return np.asarray(model.predict(features))
