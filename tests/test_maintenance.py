"""Tests for maintenance timing and outcomes."""

import pytest

from src.evaluation.maintenance import (
    find_first_rul_trigger,
    score_maintenance_timing,
)

import numpy as np


@pytest.mark.parametrize(
    "trigger, scheduled, success, discarded",
    [
        (194, 199, True, 1),
        (195, 200, False, 0),
        (196, 201, False, 0),
        (None, None, False, 0),
    ],
)
def test_maintenance_timing_boundaries(
    trigger, scheduled, success, discarded,
):
    result = score_maintenance_timing(
        trigger_cycle=trigger,
        failure_cycle=200,
    )

    assert result == {
        "trigger_occurred": trigger is not None,
        "trigger_cycle": trigger,
        "maintenance_cycle": scheduled,
        "failure_cycle": 200,
        "preventive_maintenance_success": success,
        "ran_to_failure": not success,
        "discarded_useful_life": discarded,
    }

@pytest.mark.parametrize(
    "cycles, predictions, expected_trigger",
    [
        # Equality at the threshold triggers.
        ([30, 31, 32], [40, 30, 20], 31),

        # Predictions always above the threshold: no trigger.
        ([30, 31, 32], [40, 35, 31], None),

        # Ignore cycle 29; monitoring includes cycle 30.
        ([29, 30, 31], [10, 30, 20], 30),

        # A qualifying value before monitoring cannot trigger.
        ([29, 30, 31], [10, 40, 35], None),

        # First trigger wins, even if predictions rise and fall again.
        ([30, 31, 32, 33], [40, 25, 50, 10], 31),

        # History ends before monitoring begins.
        ([27, 28, 29], [10, 10, 10], None),
    ],
)
def test_first_rul_trigger(cycles, predictions, expected_trigger):
    trigger = find_first_rul_trigger(
        cycles=np.array(cycles),
        predicted_rul=np.array(predictions),
    )

    assert trigger == expected_trigger