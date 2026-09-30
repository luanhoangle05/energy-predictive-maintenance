# Render API deployment

The first cloud build succeeded, but deployment exited early with no Python
startup output in the supplied logs. The cause is not yet established; there
is no confirmed out-of-memory error. Replace the long inline Python command
with this repository entry point to simplify invocation and expose failures:

```text
python -u -m src.api.render_start
```

Push the entry point before selecting the new command in Render. Use Docker,
branch `main`, build context `.`, Dockerfile `Dockerfile`, and health check
`/ready`. The existing Dockerfile already copies `src`, so no image definition
change is required. Builds and API execution occur on Render, not the laptop.

Keep these environment variables:

```text
MODEL_URL=https://github.com/luanhoangle05/energy-predictive-maintenance/releases/download/v0.9.0-demo/development.joblib
MODEL_SHA256=2ae93b525c3fab7ba197a5e05012f26c7f8a7d70b620d26e0ec0c995545e73c4
MODEL_PATH=/tmp/development.joblib
```

The checksum matches the local 166,863-byte model and GitHub release digest.
Only use a trusted release: joblib deserialization can execute code. The launcher
downloads at each startup, verifies the bytes before writing the model, and
replaces itself with one Uvicorn worker using Render's `PORT`. It does not train
or change predictions. The existing API still controls artifact compatibility
and readiness. A successful build alone does not verify service readiness.

Lightweight offline tests (no ML loading, network, or server startup):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_render_start.py -v
```

After deployment, check `/ready` and model identity before testing predictions.
Public serving, numerical parity, and the separate Streamlit deployment still
require verification. Keep the Free compute selection during diagnosis.

References: [Render Docker commands](https://render.com/docs/docker) and
[deployment troubleshooting](https://render.com/docs/troubleshooting-deploys).
