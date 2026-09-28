"""Tests for maintenance timing and outcomes."""

import pytest

from src.evaluation.maintenance import (
    find_first_rul_trigger,
    score_maintenance_timing,
    simulate_rul_threshold_policy,
    simulate_run_to_failure,
    simulate_fixed_age,
    simulate_lower_bound_policy

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

def test_rul_policy_successful_maintenance():
    cycles = np.arange(30, 201)
    predictions = np.full(cycles.size, 50.0)
    predictions[cycles == 194] = 30.0

    result = simulate_rul_threshold_policy(
        cycles=cycles,
        predicted_rul=predictions,
        failure_cycle=200,
    )

    assert result == {
        "policy": "predicted_rul",
        "trigger_occurred": True,
        "trigger_cycle": 194,
        "maintenance_cycle": 199,
        "failure_cycle": 200,
        "preventive_maintenance_success": True,
        "ran_to_failure": False,
        "discarded_useful_life": 1,
    }


@pytest.mark.parametrize("last_cycle", [199, 201])
def test_rul_policy_rejects_wrong_history_endpoint(last_cycle):
    cycles = np.arange(30, last_cycle + 1)
    predictions = np.full(cycles.size, 50.0)

    with pytest.raises(ValueError, match="history must end at failure_cycle"):
        simulate_rul_threshold_policy(
            cycles=cycles,
            predicted_rul=predictions,
            failure_cycle=200,
        )

@pytest.mark.parametrize(
    "cycles, predictions",
    [
        ([], []),                          # Empty history
        ([30, 31], [40]),                  # Mismatched lengths
        ([30, 30], [40, 20]),              # Duplicate cycles
        ([31, 30], [40, 20]),              # Reversed cycles
        ([30, 32], [40, 20]),              # Missing cycle
        ([30, 31], [40, np.nan]),          # Missing prediction
        ([30, 31], [40, np.inf]),          # Infinite prediction
        ([30, 31], [40, -1]),              # Negative prediction
        ([31, 32], [40, 20]),              # Monitoring start missing
    ],
)
def test_first_rul_trigger_rejects_invalid_history(cycles, predictions):
    with pytest.raises(ValueError):
        find_first_rul_trigger(
            cycles=np.array(cycles),
            predicted_rul=np.array(predictions),
        )

def test_run_to_failure_policy():
    result = simulate_run_to_failure(failure_cycle=200)

    assert result == {
        "policy": "run_to_failure",
        "trigger_occurred": False,
        "trigger_cycle": None,
        "maintenance_cycle": None,
        "failure_cycle": 200,
        "preventive_maintenance_success": False,
        "ran_to_failure": True,
        "discarded_useful_life": 0,
    }

@pytest.mark.parametrize(
    "failure_cycle, trigger, scheduled, success, discarded",
    [
        (149, None, None, False, 0),  # Failure before trigger age
        (150, 150, 155, False, 0),    # Trigger at failure
        (154, 150, 155, False, 0),    # Maintenance after failure
        (155, 150, 155, False, 0),    # Maintenance exactly at failure
        (156, 150, 155, True, 1),     # Maintenance one cycle before failure
        (200, 150, 155, True, 45),    # Earlier manual example
    ],
)
def test_fixed_age_policy(
    failure_cycle, trigger, scheduled, success, discarded,
):
    result = simulate_fixed_age(failure_cycle=failure_cycle)

    assert result == {
        "policy": "fixed_age",
        "trigger_occurred": trigger is not None,
        "trigger_cycle": trigger,
        "maintenance_cycle": scheduled,
        "failure_cycle": failure_cycle,
        "preventive_maintenance_success": success,
        "ran_to_failure": not success,
        "discarded_useful_life": discarded,
    }

@pytest.mark.parametrize(
    "first_trigger, scheduled, success, discarded",
    [
        (180, 185, True, 15),
        (195, 200, False, 0),
        (None, None, False, 0),
    ],
)
def test_lower_bound_policy(
    first_trigger, scheduled, success, discarded,
):
    cycles = np.arange(30, 201)
    lower_bounds = np.full(cycles.size, 60.0)

    if first_trigger is not None:
        lower_bounds[cycles >= first_trigger] = 30.0

    result = simulate_lower_bound_policy(
        cycles=cycles,
        lower_bound=lower_bounds,
        failure_cycle=200,
    )

    assert result == {
        "policy": "lower_bound",
        "trigger_occurred": first_trigger is not None,
        "trigger_cycle": first_trigger,
        "maintenance_cycle": scheduled,
        "failure_cycle": 200,
        "preventive_maintenance_success": success,
        "ran_to_failure": not success,
        "discarded_useful_life": discarded,
    }