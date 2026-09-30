"""Development-only partial histories, selected before any inference."""

from functools import lru_cache
import hashlib
from io import BytesIO
import os
from pathlib import Path
import urllib.request

from src.data.load_data import CMAPSS_COLUMNS, load_cmapss_file
from src.data.prepare_data import select_cutoff_indices
from src.data.serving_upload import read_observations
from src.models.bundle import validate_history

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_SOURCE = PROJECT_ROOT / "data/raw/cmapss/train_FD001.txt"
DEMO_SEED = 43


@lru_cache(maxsize=1)
def download_demo_source(url, checksum):
    """Cache only verified development training bytes; never fetch a test file."""
    if not url.startswith("https://") or not url.endswith("/train_FD001.txt"):
        raise ValueError("DEMO_SOURCE_URL must be an HTTPS URL ending in /train_FD001.txt")
    if len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
        raise ValueError("DEMO_SOURCE_SHA256 must contain 64 hexadecimal characters")
    limit = 8 * 1024 * 1024
    with urllib.request.urlopen(url, timeout=60) as response:
        content = response.read(limit + 1)
    if len(content) > limit:
        raise ValueError("Demo source exceeds 8 MiB")
    if hashlib.sha256(content).hexdigest() != checksum:
        raise ValueError("Demo source checksum mismatch")
    return content


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
    # Preserve local behavior; cloud download is enabled only by explicit settings.
    if DEMO_SOURCE.is_file():
        raw = load_cmapss_file(DEMO_SOURCE)
    elif os.environ.get("DEMO_SOURCE_URL"):
        content = download_demo_source(
            os.environ["DEMO_SOURCE_URL"].strip(),
            os.environ.get("DEMO_SOURCE_SHA256", "").strip().lower(),
        )
        raw = read_observations(BytesIO(content), "train_FD001.txt")
    else:
        raise FileNotFoundError("Configure the development training source to enable this demo")
    return truncate_demo(raw)
