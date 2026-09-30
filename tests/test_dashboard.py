"""Check dashboard startup without needing a running service."""

from pathlib import Path
import os
import subprocess
import sys
from streamlit.testing.v1 import AppTest


def test_dashboard_starts_and_discloses_scope():
    path = Path(__file__).resolve().parents[1] / "app/dashboard.py"
    app = AppTest.from_file(str(path)).run(timeout=30)
    assert not app.exception
    assert app.header[0].value == "Predictive Maintenance Dashboard"
    assert any("Not validated" in caption.value for caption in app.caption)


def test_dashboard_imports_without_repository_on_python_path(tmp_path):
    """Reproduce a launcher that does not expose the repository root to imports."""
    path = Path(__file__).resolve().parents[1] / "app/dashboard.py"
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    # Bare script execution renders no browser and starts no server or inference.
    result = subprocess.run(
        [sys.executable, "-I", str(path)], cwd=tmp_path, env=environment,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
