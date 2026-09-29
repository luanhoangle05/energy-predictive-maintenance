"""NASA upload parsing and history-preserving batching."""

from io import StringIO

import pandas as pd
import pytest

from src.data.load_data import CMAPSS_COLUMNS
from src.data.serving_upload import read_observations, history_batches


def test_original_nasa_whitespace_format():
    rows = [[engine, cycle] + [1.2] * 24
            for engine in (1, 2, 3) for cycle in range(1, 4)]
    text = "\n".join("  ".join(map(str, row)) + "  " for row in rows)
    frame = read_observations(StringIO(text), "train_FD001.txt")
    assert frame.shape == (9, 26)
    assert frame.unit_number.nunique() == 3


def test_labels_are_rejected():
    with pytest.raises(ValueError, match="labels"):
        read_observations(StringIO("112\n98\n"), "RUL_FD001.txt")


def test_batches_preserve_all_complete_engines():
    rows = [[engine, cycle] + [1.2] * 24
            for engine in (1, 2, 3) for cycle in range(1, 5001)]
    frame = pd.DataFrame(rows, columns=CMAPSS_COLUMNS)
    batches = list(history_batches(frame))
    assert [len(b) for b in batches] == [10000, 5000]
    assert pd.concat(batches).equals(frame)
    assert set(batches[0].unit_number).isdisjoint(batches[1].unit_number)
