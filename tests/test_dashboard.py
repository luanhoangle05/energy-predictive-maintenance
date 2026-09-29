"""Check dashboard startup without needing a running service."""

from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_dashboard_starts_and_discloses_scope():
    path = Path(__file__).resolve().parents[1] / "app/dashboard.py"
    app = AppTest.from_file(str(path)).run(timeout=30)
    assert not app.exception
    assert app.header[0].value == "Predictive Maintenance Dashboard"
    assert any("Not validated" in caption.value for caption in app.caption)
