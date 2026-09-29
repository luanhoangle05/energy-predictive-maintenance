"""Parse dashboard uploads and batch complete histories for the API."""

import pandas as pd

from src.data.load_data import CMAPSS_COLUMNS
from src.models.bundle import validate_history

MAX_UPLOAD_ROWS = 50000
MAX_REQUEST_ROWS = 10000


def read_observations(upload, filename):
    if filename.lower().endswith(".txt"):
        frame = pd.read_csv(upload, sep=r"\s+", header=None, nrows=MAX_UPLOAD_ROWS + 1)
        if frame.shape[1] != len(CMAPSS_COLUMNS):
            raise ValueError("NASA trajectory files must contain 26 columns. "
                             "RUL_FD001.txt contains labels, not sensor histories.")
        frame.columns = CMAPSS_COLUMNS
    else:
        frame = pd.read_csv(upload, nrows=MAX_UPLOAD_ROWS + 1)
    if len(frame) > MAX_UPLOAD_ROWS:
        raise ValueError(f"Upload at most {MAX_UPLOAD_ROWS:,} observations.")
    return validate_history(frame)


def history_batches(frame):
    """Keep each engine together so rolling features remain correct."""
    groups, size = [], 0
    for _, history in frame.groupby("unit_number", sort=True):
        if len(history) > MAX_REQUEST_ROWS:
            raise ValueError("An individual engine exceeds the 10,000-row API limit.")
        if size + len(history) > MAX_REQUEST_ROWS:
            yield pd.concat(groups)
            groups, size = [], 0
        groups.append(history)
        size += len(history)
    if groups:
        yield pd.concat(groups)
