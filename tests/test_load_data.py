"""Tests for loading C-MAPSS source files."""

from pathlib import Path

from src.data.load_data import RUL_COLUMN, load_cmapss_rul_file


def test_load_rul_file_assigns_one_based_engine_ids(
    tmp_path: Path,
) -> None:
    rul_file = tmp_path / "RUL_FD001.txt"
    rul_file.write_text("112\n98\n69\n", encoding="utf-8")

    result = load_cmapss_rul_file(rul_file)

    assert result.to_dict(orient="list") == {
        "unit_number": [1, 2, 3],
        RUL_COLUMN: [112, 98, 69],
    }