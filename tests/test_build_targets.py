"""Tests for Remaining Useful Life target construction."""

import pandas as pd

from src.data.build_targets import RUL_COLUMN, add_training_rul


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