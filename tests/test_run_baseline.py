"""Tests for the baseline model-comparison workflow."""

import pandas as pd
import pytest

from src.evaluation.run_baseline import (
cross_validate_decision_tree,
cross_validate_random_forest
)


def test_cross_validate_decision_tree_returns_mean_metrics() -> None:
    features = pd.DataFrame(
        {
            "sensor_1": [
                0.0, 1.0,
                0.0, 1.0,
                0.0, 1.0,
                0.0, 1.0,
                0.0, 1.0,
            ],
        }
    )

    targets = pd.Series(
        [
            10.0, 2.0,
            10.0, 2.0,
            10.0, 2.0,
            10.0, 2.0,
            10.0, 2.0,
        ]
    )

    engine_groups = pd.Series(
        [
            1, 1,
            2, 2,
            3, 3,
            4, 4,
            5, 5,
        ]
    )

    metrics = cross_validate_decision_tree(
        features,
        targets,
        engine_groups,
        max_depth=1,
        min_samples_leaf=1,
    )

    assert metrics["mean_mae"] == pytest.approx(0.0)
    assert metrics["mean_rmse"] == pytest.approx(0.0)

def test_cross_validate_random_forest_returns_mean_metrics() -> None:
    features = pd.DataFrame(
        {
            "sensor_1": [
                0.0, 1.0,
                0.0, 1.0,
                0.0, 1.0,
                0.0, 1.0,
                0.0, 1.0,
            ],
        }
    )

    targets = pd.Series(
        [
            10.0, 2.0,
            10.0, 2.0,
            10.0, 2.0,
            10.0, 2.0,
            10.0, 2.0,
        ]
    )

    engine_groups = pd.Series(
        [
            1, 1,
            2, 2,
            3, 3,
            4, 4,
            5, 5,
        ]
    )

    metrics = cross_validate_random_forest(
        features,
        targets,
        engine_groups,
        n_estimators=10,
        max_depth=2,
        min_samples_leaf=1,
        max_features=1.0,
    )

    assert metrics["mean_mae"] >= 0.0
    assert metrics["mean_rmse"] >= 0.0
    assert metrics["mean_rmse"] >= metrics["mean_mae"]