"""Development-only partial histories, selected before any inference."""

from pathlib import Path

from src.data.load_data import CMAPSS_COLUMNS, load_cmapss_file
from src.data.prepare_data import select_cutoff_indices
from src.models.bundle import validate_history

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_SOURCE = PROJECT_ROOT / "data/raw/cmapss/train_FD001.txt"
DEMO_SEED = 43


def truncate_demo(raw):
    """Reuse Milestone 7 snapshots; return only raw observations through cutoff.

    Full development trajectories establish eligible pre-failure cutoffs for
    this retrospective demonstration. Neither targets nor future rows are
    returned to the inference client. Seed is fixed independently of predictions.
    """
    raw = validate_history(raw.loc[:, CMAPSS_COLUMNS])
    indices = select_cutoff_indices(raw, minimum_cycle=30, random_state=DEMO_SEED)
    cutoffs = raw.loc[indices].set_index("unit_number")["time_in_cycles"]
    observed = raw.loc[raw.time_in_cycles <= raw.unit_number.map(cutoffs), CMAPSS_COLUMNS]
    return validate_history(observed)


def load_demo():
    # Fixed training filename; no arbitrary path and no fallback to official test data.
    return truncate_demo(load_cmapss_file(DEMO_SOURCE))
