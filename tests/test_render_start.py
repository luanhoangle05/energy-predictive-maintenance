"""Lightweight startup checks: no real network, ML imports, or server processes."""

import hashlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.api import render_start


class RenderStartupTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.destination = Path(self.directory.name) / "model.joblib"
        self.payload = b"synthetic artifact bytes, never deserialized"
        self.environment = {
            "MODEL_URL": "https://example.com/model.joblib",
            "MODEL_SHA256": hashlib.sha256(self.payload).hexdigest(),
            "MODEL_PATH": str(self.destination),
            "PORT": "10000",
        }

    def test_verified_download_preserves_exact_bytes(self):
        with patch.object(render_start.urllib.request, "urlopen", return_value=io.BytesIO(self.payload)):
            render_start.prepare_model(self.environment)
        self.assertEqual(self.destination.read_bytes(), self.payload)

    def test_mismatched_download_does_not_replace_existing_model(self):
        self.destination.write_bytes(b"existing")
        with patch.object(render_start.urllib.request, "urlopen", return_value=io.BytesIO(b"wrong")):
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                render_start.prepare_model(self.environment)
        self.assertEqual(self.destination.read_bytes(), b"existing")

    def test_invalid_configuration_fails_before_network(self):
        for key, value in [("MODEL_URL", ""), ("MODEL_URL", "http://example.com"),
                           ("MODEL_SHA256", "bad"), ("MODEL_PATH", "")]:
            with self.subTest(key=key, value=value):
                with patch.object(render_start.urllib.request, "urlopen") as download:
                    with self.assertRaises(ValueError):
                        render_start.prepare_model(dict(self.environment, **{key: value}))
                    download.assert_not_called()

    def test_oversized_download_is_rejected(self):
        with patch.object(render_start, "MAX_MODEL_BYTES", 4):
            with patch.object(render_start.urllib.request, "urlopen", return_value=io.BytesIO(self.payload)):
                with self.assertRaisesRegex(ValueError, "exceeds"):
                    render_start.prepare_model(self.environment)
        self.assertFalse(self.destination.exists())

    def test_success_starts_existing_api_with_one_worker(self):
        with patch.dict(os.environ, self.environment, clear=True):
            with patch.object(render_start.urllib.request, "urlopen", return_value=io.BytesIO(self.payload)):
                with patch.object(render_start.os, "execv") as launch:
                    self.assertEqual(render_start.main(), 0)
        arguments = launch.call_args.args[1]
        self.assertIn("src.api.main:app", arguments)
        self.assertEqual(arguments[-4:], ["--port", "10000", "--workers", "1"])

    def test_failed_download_reports_error_and_does_not_start_api(self):
        with patch.dict(os.environ, self.environment, clear=True):
            with patch.object(render_start.urllib.request, "urlopen", side_effect=TimeoutError("download timed out")):
                with patch.object(render_start.os, "execv") as launch:
                    with patch("sys.stderr", new_callable=io.StringIO) as errors:
                        self.assertEqual(render_start.main(), 1)
                    self.assertIn("STARTUP FAILED: TimeoutError", errors.getvalue())
                    launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
