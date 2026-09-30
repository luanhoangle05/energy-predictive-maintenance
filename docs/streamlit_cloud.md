# Streamlit Community Cloud

Use `main`, entry point `app/dashboard.py`, and Python 3.10 where available.
`app/requirements.txt` provides the dashboard dependencies. The entry point
adds the repository root to the import path for cloud script execution.

Set the following top-level entries in Streamlit's Secrets settings:

```toml
API_URL = "https://energy-predictive-maintenance.onrender.com"
DEMO_SOURCE_URL = "https://github.com/luanhoangle05/energy-predictive-maintenance/releases/download/v0.9.0-demo/train_FD001.txt"
DEMO_SOURCE_SHA256 = "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8"
```

The demo settings require uploading the existing `data/raw/cmapss/train_FD001.txt`
as an asset on release `v0.9.0-demo`. This is a public dataset attachment, not
a Git source file; the `.gitignore` rules stay unchanged. It is the NASA C-MAPSS
FD001 simulated development training data, not original industrial telemetry.
Retain the project's NASA dataset attribution and source terms. Do not upload
or use the official test trajectories or RUL labels for the development demo.

The expected training file is 3,515,356 bytes. Cloud demo loading checks the
SHA-256 digest before parsing, caches verified bytes in process memory, then
uses the same seed-43 cutoff selection and observed-history truncation as local
use. A local training file takes precedence. Missing configuration, failed
download, or failed verification leaves compatible user uploads available.
No model training or historical RUL prediction is added.

Observed deployment evidence: Render `/ready` returned the expected model ID
`b85ac2f2-d912-4e35-979d-40c26d0adb87`; the user demonstrated a hosted upload with
three predictions and zero attention flags. Full-precision hosted parity and
the cloud demo after attaching the dataset still require verification.

Reference: [Community Cloud dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).
