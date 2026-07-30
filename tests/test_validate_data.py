import pandas as pd
import pytest

from src.data.validate_data import (
    validate_cycle_sequence,
    validate_no_infinite_values,
    validate_unique_engine_cycles,
)


def test_duplicate_engine_cycle_is_rejected() -> None:
    """Duplicate engine-cycle records should raise an error."""
    duplicated_data = pd.DataFrame(
        {
            "unit_number": [1, 1],
            "time_in_cycles": [1, 1],
        }
    )

    with pytest.raises(ValueError, match="duplicate engine-cycle"):
        validate_unique_engine_cycles(duplicated_data)


def test_unique_engine_cycles_are_accepted() -> None:
    """Unique engine-cycle records should pass validation."""
    valid_data = pd.DataFrame(
        {
            "unit_number": [1, 1, 2, 2],
            "time_in_cycles": [1, 2, 1, 2],
        }
    )

    validate_unique_engine_cycles(valid_data)


def test_cycle_gap_is_rejected() -> None:
    """A missing cycle within an engine sequence should raise an error."""
    data_with_gap = pd.DataFrame(
        {
            "unit_number": [1, 1, 1],
            "time_in_cycles": [1, 2, 4],
        }
    )

    with pytest.raises(ValueError, match="invalid cycle step"):
        validate_cycle_sequence(data_with_gap)


def test_infinite_value_is_rejected() -> None:
    """Infinite numeric values should raise an error."""
    data_with_infinity = pd.DataFrame(
        {
            "unit_number": [1],
            "time_in_cycles": [1],
            "sensor_1": [float("inf")],
        }
    )

    with pytest.raises(ValueError, match="infinite numeric value"):
        validate_no_infinite_values(data_with_infinity)
