"""Hypothetical scenario costs, not validated maintenance economics."""

import numpy as np


def calculate_maintenance_cost(
    preventive_maintenance_success: bool,
    discarded_useful_life: int,
    *,
    preventive_cost: float = 1.0,
    failure_cost: float = 10.0,
    discarded_cycle_cost: float = 0.01,
) -> float:
    """Calculate hypothetical cost for one already-simulated outcome."""
    if not isinstance(preventive_maintenance_success, (bool, np.bool_)):
        raise ValueError("preventive_maintenance_success must be Boolean")

    if (
        not np.isfinite(discarded_useful_life)
        or discarded_useful_life < 0
        or discarded_useful_life != int(discarded_useful_life)
    ):
        raise ValueError(
            "discarded_useful_life must be a non-negative integer"
        )

    for name, value in (
        ("preventive_cost", preventive_cost),
        ("failure_cost", failure_cost),
        ("discarded_cycle_cost", discarded_cycle_cost),
    ):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")

    if preventive_maintenance_success:
        return float(
            preventive_cost
            + discarded_cycle_cost * discarded_useful_life
        )

    return float(failure_cost)