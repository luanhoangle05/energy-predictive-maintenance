"""Tests for Remaining Useful Life target construction."""

import pandas as pd
import pytest

from src.data.build_targets import (
    RUL_COLUMN,
    add_test_rul,
    add_training_rul,
)


def test_training_rul_counts_down_to_zero_for_each_engine() -> None:
    data = pd.DataFrame(
        {
            "unit_number": [1, 1, 1, 2, 2],
            "time_in_cycles": [1, 2, 3, 1, 2],
        }
    )

    result = add_training_rul(data)

    assert result[RUL_COLUMN].tolist() == [2, 1, 0, 1, 0]
    assert RUL_COLUMN not in data.columns


def test_test_rul_counts_back_from_endpoint_values() -> None:
    data = pd.DataFrame(
        {
            "unit_number": [1, 1, 1, 2, 2],
            "time_in_cycles": [1, 2, 3, 1, 2],
        }
    )
    endpoint_rul = pd.DataFrame(
        {
            "unit_number": [1, 2],
            RUL_COLUMN: [4, 10],
        }
    )

    result = add_test_rul(data, endpoint_rul)

    assert result[RUL_COLUMN].tolist() == [6, 5, 4, 11, 10]
    assert RUL_COLUMN not in data.columns


# If the endpoint file contains different engine IDs,
#. map() would silently produce missing RUL values.
# We want a clear error instead.
def test_test_rul_rejects_mismatched_engine_ids() -> None:
    data = pd.DataFrame(
        {
            "unit_number": [1, 2],
            "time_in_cycles": [1, 1],
        }
    )
    endpoint_rul = pd.DataFrame(
        {
            "unit_number": [1, 3],
            RUL_COLUMN: [4, 10],
        }
    )

    with pytest.raises(ValueError, match="engine IDs do not match"):
        add_test_rul(data, endpoint_rul)