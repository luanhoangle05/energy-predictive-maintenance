"""Tests for hypothetical maintenance costs."""

import pytest

from src.evaluation.maintenance import score_maintenance_timing
from src.evaluation.maintenance_cost import calculate_maintenance_cost


@pytest.mark.parametrize(
    "success, discarded, expected",
    [
        (True, 0, 1.0),
        (True, 37, 1.37),
        (False, 0, 10.0),
        (False, 37, 10.0),
    ],
)
def test_cost_calculation(success, discarded, expected):
    cost = calculate_maintenance_cost(success, discarded)

    assert cost == pytest.approx(expected)


@pytest.mark.parametrize(
    "trigger",
    [
        None,  # No trigger
        196,   # Maintenance after failure
        195,   # Maintenance exactly at failure
    ],
)
def test_failure_outcomes_charge_only_failure_cost(trigger):
    outcome = score_maintenance_timing(
        trigger_cycle=trigger,
        failure_cycle=200,
    )

    cost = calculate_maintenance_cost(
        preventive_maintenance_success=outcome[
            "preventive_maintenance_success"
        ],
        discarded_useful_life=outcome["discarded_useful_life"],
    )

    assert cost == pytest.approx(10.0)


@pytest.mark.parametrize(
    "parameter",
    ["preventive_cost", "failure_cost", "discarded_cycle_cost"],
)
def test_negative_cost_parameter_is_rejected(parameter):
    with pytest.raises(ValueError, match=parameter):
        calculate_maintenance_cost(
            True,
            37,
            **{parameter: -1.0},
        )


@pytest.mark.parametrize("success", [True, False])
def test_negative_discarded_life_is_rejected(success):
    with pytest.raises(ValueError, match="discarded_useful_life"):
        calculate_maintenance_cost(success, -1)