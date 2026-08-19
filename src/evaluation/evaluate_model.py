"""Evaluate Remaining Useful Life regression predictions."""

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(
        actual: np.ndarray,
        predicted: np.ndarray
) -> Dict[str, float]:
    """Calculate evaluation metrics in a future milestone."""
    mae = mean_absolute_error(actual, predicted)

    mse = mean_squared_error(actual,predicted)
    rmse = np.sqrt(mse)

    r2=r2_score(actual,predicted)

    return {
        "mae":float(mae),
        "rmse":float(rmse),
        "r2":float(r2)
    }


