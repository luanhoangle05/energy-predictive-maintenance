"""Download a checksum-pinned trusted model, then start the unchanged API."""

import hashlib
import os
from pathlib import Path
import re
import sys
import urllib.request

MAX_MODEL_BYTES = 8 * 1024 * 1024


def prepare_model(environ):
    required = ("MODEL_URL", "MODEL_SHA256", "MODEL_PATH")
    missing = [key for key in required if not environ.get(key, "").strip()]
    if missing:
        raise ValueError("Missing environment variables: " + ", ".join(missing))
    url = environ["MODEL_URL"].strip()
    checksum = environ["MODEL_SHA256"].strip().lower()
    if not url.startswith("https://"):
        raise ValueError("MODEL_URL must use HTTPS")
    if not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise ValueError("MODEL_SHA256 must contain 64 hexadecimal characters")
    destination = Path(environ["MODEL_PATH"].strip())
    print("STARTUP: Downloading model", flush=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read(MAX_MODEL_BYTES + 1)
    if len(data) > MAX_MODEL_BYTES:
        raise ValueError("Model download exceeds 8 MiB")
    if hashlib.sha256(data).hexdigest() != checksum:
        raise ValueError("Model checksum mismatch; refusing to load artifact")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    print("STARTUP: Model downloaded and checksum verified", flush=True)
    return destination


def main():
    print("STARTUP: Python running", flush=True)
    try:
        port = int(os.environ.get("PORT", "10000"))
        if not 1 <= port <= 65535:
            raise ValueError("PORT must be between 1 and 65535")
        destination = prepare_model(os.environ)
        os.environ["MODEL_PATH"] = str(destination)
        print(f"STARTUP: Launching API on port {port}", flush=True)
        os.execv(sys.executable, [
            sys.executable, "-u", "-m", "uvicorn", "src.api.main:app",
            "--host", "0.0.0.0", "--port", str(port), "--workers", "1",
        ])
    except Exception as exc:
        print(f"STARTUP FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
