"""Helpers for exploratory maintenance-timing simulations."""

import numpy as np


def find_first_rul_trigger(
    cycles: np.ndarray,
    predicted_rul: np.ndarray,
    *,
    monitoring_start: int = 30,
    rul_threshold: float = 30.0,
) -> int | None:
    """Return the first eligible trigger cycle, or None."""
    cycles = np.asarray(cycles, dtype=float)
    predicted_rul = np.asarray(predicted_rul, dtype=float)

    if cycles.ndim != 1 or predicted_rul.ndim != 1:
        raise ValueError("cycles and predicted_rul must be one-dimensional")

    if cycles.size == 0 or cycles.size != predicted_rul.size:
        raise ValueError("Inputs must have equal nonzero lengths")

    if not np.isfinite(cycles).all():
        raise ValueError("cycles must contain finite values")

    if np.any(cycles < 1) or np.any(cycles != np.floor(cycles)):
        raise ValueError("cycles must be positive integers")

    if np.any(np.diff(cycles) != 1):
        raise ValueError("cycles must be consecutive and increasing")

    if (
        not np.isfinite(predicted_rul).all()
        or np.any(predicted_rul < 0)
    ):
        raise ValueError("predicted_rul must be finite and non-negative")

    if (
        not np.isfinite(monitoring_start)
        or monitoring_start < 1
        or monitoring_start != int(monitoring_start)
    ):
        raise ValueError("monitoring_start must be a positive integer")

    if not np.isfinite(rul_threshold) or rul_threshold < 0:
        raise ValueError("rul_threshold must be finite and non-negative")

    if cycles[0] > monitoring_start:
        raise ValueError("The history must begin at or before monitoring_start")

    for cycle, prediction in zip(cycles, predicted_rul):
        if cycle >= monitoring_start and prediction <= rul_threshold:
            return int(cycle)

    return None

def score_maintenance_timing(
    trigger_cycle: int | None,
    failure_cycle: int,
    *,
    lead_time: int = 5,
) -> dict:
    """Score a previously selected trigger against actual failure."""
    if (
        not np.isfinite(failure_cycle)
        or failure_cycle < 1
        or failure_cycle != int(failure_cycle)
    ):
        raise ValueError("failure_cycle must be a positive integer")

    if (
        not np.isfinite(lead_time)
        or lead_time < 0
        or lead_time != int(lead_time)
    ):
        raise ValueError("lead_time must be a non-negative integer")

    if trigger_cycle is not None:
        if (
            not np.isfinite(trigger_cycle)
            or trigger_cycle < 1
            or trigger_cycle != int(trigger_cycle)
        ):
            raise ValueError("trigger_cycle must be a positive integer or None")

        if trigger_cycle > failure_cycle:
            raise ValueError("trigger_cycle cannot be after failure")

    maintenance_cycle = (
        int(trigger_cycle + lead_time)
        if trigger_cycle is not None
        else None
    )

    success = (
        maintenance_cycle is not None
        and maintenance_cycle < failure_cycle
    )

    return {
        "trigger_occurred": trigger_cycle is not None,
        "trigger_cycle": trigger_cycle,
        "maintenance_cycle": maintenance_cycle,
        "failure_cycle": int(failure_cycle),
        "preventive_maintenance_success": success,
        "ran_to_failure": not success,
        "discarded_useful_life": (
            int(failure_cycle - maintenance_cycle) if success else 0
        ),
    }

def simulate_rul_threshold_policy(
    cycles: np.ndarray,
    predicted_rul: np.ndarray,
    failure_cycle: int,
    *,
    monitoring_start: int = 30,
    lead_time: int = 5,
    rul_threshold: float = 30.0,
) -> dict:
    """Simulate one engine using a predicted-RUL threshold."""
    trigger_cycle = find_first_rul_trigger(
        cycles,
        predicted_rul,
        monitoring_start=monitoring_start,
        rul_threshold=rul_threshold,
    )

    result = score_maintenance_timing(
        trigger_cycle,
        failure_cycle,
        lead_time=lead_time,
    )

    # Retrospective input check: require history through failure.
    if np.asarray(cycles)[-1] != failure_cycle:
        raise ValueError("The history must end at failure_cycle")

    return {"policy": "predicted_rul", **result}

def simulate_run_to_failure(failure_cycle: int) -> dict:
    """Score an engine that receives no preventive maintenance."""
    result = score_maintenance_timing(
        trigger_cycle=None,
        failure_cycle=failure_cycle,
    )

    return {"policy": "run_to_failure", **result}

def simulate_fixed_age(failure_cycle: int) -> dict:
    """Simulate a frozen cycle-150 trigger with five-cycle lead time."""
    fixed_age = 150

    # Validate the failure cycle using the existing scorer.
    result = score_maintenance_timing(
        trigger_cycle=None,
        failure_cycle=failure_cycle,
    )

    # The trigger occurs only if the observed lifetime reaches cycle 150.
    if failure_cycle >= fixed_age:
        result = score_maintenance_timing(
            trigger_cycle=fixed_age,
            failure_cycle=failure_cycle,
            lead_time=5,
        )

    return {"policy": "fixed_age", **result}

def simulate_lower_bound_policy(
    cycles: np.ndarray,
    lower_bound: np.ndarray,
    failure_cycle: int,
    *,
    monitoring_start: int = 30,
    lead_time: int = 5,
    rul_threshold: float = 30.0,
) -> dict:
    """Apply an exploratory lower-bound threshold along one history.

    Snapshot calibration does not guarantee simultaneous coverage
    across the trajectory.
    """
    result = simulate_rul_threshold_policy(
        cycles=cycles,
        predicted_rul=lower_bound,
        failure_cycle=failure_cycle,
        monitoring_start=monitoring_start,
        lead_time=lead_time,
        rul_threshold=rul_threshold,
    )

    return {**result, "policy": "lower_bound"}