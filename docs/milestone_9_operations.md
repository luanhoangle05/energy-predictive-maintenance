# Milestone 9: Development workflow operationalization

The latest tabbed dashboard, partial-history demo, model insights, and read-only
maintenance integration are documented in [the product-pass record](dashboard_product_pass.md).
Its current behavior supersedes the earlier dashboard-redesign notes below.

## Scope and status

This milestone adds a tracked training command, a versioned model bundle,
FastAPI inference, a Streamlit CSV dashboard, Docker Compose, and GitHub Actions.
The service exposes the existing exploratory Milestone 7 model: 64 fitting,
16 calibration, and 20 evaluation engines. It is not a final model fitted on all
training engines. NASA's official test trajectories and endpoint labels remain
untouched. Maintenance cost simulation remains a separate development analysis.

Local Python validation is recorded below. Docker execution and hosted deployment
remain pending: Docker is not installed in the implementation environment, and
no hosting destination has been specified. The roadmap remains unchecked until
deployment verification is complete. GitHub Actions will run after the code is
pushed; configuring a workflow does not establish a successful remote CI run.

## Reproduce training and inspect tracking

Run from the repository root with Python 3.10 and the virtual environment active:

```powershell
python -m pip install -r requirements-ops.txt
python -m pytest -q
python -m src.models.train_tracked
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

The training command reads only `data/raw/cmapss/train_FD001.txt`. It uses the
existing engine splits, five-cycle rolling features, fixed Gradient Boosting
parameters, and one cutoff per calibration/evaluation engine. Preprocessing is
fitted on training engines only. Predictions are clipped at zero before interval
calibration, matching Milestone 7. The evaluation cutoff seed is 43; the training,
split, and calibration cutoff seeds are 42. Repeatedly inspected development
metrics are exploratory and are not an unbiased final benchmark.

MLflow stores parameters, metrics, scope tags, and copies of the model and JSON
manifest. The manifest includes split engine IDs, source SHA-256, package version,
run ID, feature window, and calibration confidence. Local tracking uses SQLite
(`mlflow.db`) with local artifacts. Generated artifacts and tracking data are
ignored by Git. Use `--tracking-uri` to select a configured tracking backend and
`--output` to choose an artifact path. Each run has a unique model ID.

The default artifact is `models/development.joblib`, accompanied by a readable
`models/development.json`. It contains fitted preprocessing, the estimator, the
calibration radius, and metadata. Load only artifacts produced by a trusted
training process: joblib deserialization can execute Python code. The loader
rejects incompatible schema and scikit-learn versions. Dependencies with serving
APIs are pinned, but this is not a fully locked transitive environment.

## Local service and dashboard

In separate terminals with the same environment active:

```powershell
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
python -m streamlit run app/dashboard.py --server.address 127.0.0.1
```

Open the API documentation at http://127.0.0.1:8000/docs and the dashboard at
http://127.0.0.1:8501. `MODEL_PATH` selects the trusted local model file and
`API_URL` selects the dashboard's API (default http://127.0.0.1:8000).
The model loads once at startup. Restart the API after replacing its artifact.

Endpoints:

| Route | Contract |
| --- | --- |
| `GET /health` | Process liveness, independent of model availability |
| `GET /ready` | 200 and model ID when loaded; 503 otherwise |
| `GET /metadata` | Training metadata and development metrics; 503 if unavailable |
| `POST /predict` | `{"observations": [...]}` containing 1–10,000 raw observation objects |

Every observation must have exactly the 26 columns in `CMAPSS_COLUMNS`:
`unit_number`, `time_in_cycles`, `operational_setting_1` through
`operational_setting_3`, and `sensor_1` through `sensor_21`. Values must be finite
numbers. IDs and cycles must be positive integers. Include each engine's complete
observed history starting at cycle 1, in ascending consecutive cycle order.
Targets, precomputed features, duplicate cycles, missing values, and extra columns
are rejected with HTTP 422. The service does not retain history between requests.

Feature calculation uses each engine's own history and returns only its latest
observed cycle, ordered by engine ID. Outputs include non-negative point RUL,
lower/upper bounds, 30-cycle threshold status, and a lower-bound low-RUL flag.
Flags are not failure probabilities. The 90% calibration level does not establish
90% coverage for arbitrary uploaded equipment or for repeatedly queried cycles.
The dashboard accepts a CSV with the same named columns or an original NASA
whitespace-delimited trajectory `.txt` file without headers, and offers a result
CSV. Uploads may contain up to 50,000 rows; complete engine histories are batched
into API requests of at most 10,000 rows without splitting an engine. The preview
shows the first 20 rows; the displayed count reports all uploaded engines.
Use `data/raw/cmapss/train_FD001.txt` for a real-file development demo. Full training
trajectories end at failure, so their true endpoint RUL is zero; this is not an
unseen benchmark. `RUL_FD001.txt` is a label file and cannot be used as an input
history. Running `test_FD001.txt` begins the reserved official test evaluation.

## Container deployment

Install Docker with Compose separately. First generate the development artifact
using the pinned Python environment above, then run:

```powershell
docker compose up --build -d
docker compose ps
curl.exe --fail http://127.0.0.1:8000/ready
curl.exe --fail http://127.0.0.1:8501/_stcore/health
docker compose down
```

The API mounts `models/` read-only. The dashboard waits for API readiness. Both
containers run as a non-root user; published ports bind to the local machine.
The Docker build context excludes raw data, trained artifacts, tracking data,
and unrelated workspace files. Training is a separate step, never a server
startup side effect. Stop with `docker compose down`; model files remain on disk.

This is a local portfolio deployment without authentication. Hosted deployment
requires a selected provider, access controls/TLS and request-size limits at the
ingress, a compatible trusted artifact, and a smoke test of readiness and actual
predictions. No cloud resources or public endpoint are created by this change.

## Validation

On September 27, 2026, all 152 tests passed on Python 3.10 (one upstream
Starlette/httpx deprecation warning). A real development training run logged
successfully to local MLflow and exported the serving bundle. Its 20 evaluation
snapshots had MAE 24.09, RMSE 29.52, coverage 20/20, interval radius 74.92,
and mean interval width 137.30 cycles, reproducing the earlier interval results.
These snapshot metrics differ from the all-cycle Milestone 6 benchmark.
Live local smoke checks returned HTTP 200 for API readiness and dashboard health;
an API prediction for the first 30 cycles of development engine 1 exactly matched
direct inference from the saved artifact.

Automated tests cover artifact replay, per-engine feature isolation, API/direct
prediction parity, invalid history rejection, unavailable/corrupt models,
version mismatch, synthetic tracked training with artifact logging, and dashboard
startup. They use synthetic data and do not require NASA files. CI runs the full
suite, validates Compose configuration, and builds the container image.

## Dashboard redesign verification — September 28, 2026

The presentation now uses a native light Streamlit theme, bordered KPI cards,
a Plotly horizontal point/range chart for the 20 lowest predicted-RUL engines,
a compact priority table, and a persistent engine selector. Observed sensor
history uses the existing Altair rendering stack with an axis scaled to the
observed values. It is explicitly separate from the latest endpoint RUL and
exploratory bounds; no historical RUL estimates are generated. Requirements,
raw preview, metadata, dataset context, and limitations are collapsed by default.
The low-RUL definition remains lower bound <= 30 cycles, not failure probability.

Prediction state belongs to the upload's content hash and filename. Selecting
an engine or sensor does not submit inference again. Replacing or removing an
upload clears predictions, selections, metadata, and validation state, including
when the replacement is invalid. Metadata retrieval is optional and checks its
model ID against the prediction result. No ML or API behavior changed.

Validation completed:

- Full test suite: 161 passed, one existing Starlette/httpx deprecation warning.
- Browser: original NASA `train_FD001.txt` upload, validation, prediction for all
  100 engines, and selection of Engine 002 without losing the fleet results.
- All 100 predicted RUL values, both bounds, and flags exactly matched the
  pre-edit API baseline. Hashes of all `src/**/*.py`, the saved model, and all
  maintenance report CSVs were unchanged.
- Desktop 1280px and narrow 640px layouts inspected: cards stack, fleet labels
  remain readable, sensor histories render, and no page-level horizontal overflow.
- Maintenance integration was intentionally deferred; no policy or cost output
  is presented in the redesigned dashboard. The official test set was not used.

The reported `canvascanvas` artifact was investigated before edits using the
populated baseline. It was not present in application source, browser-rendered
body text, or inspected screenshots. Streamlit generates canvas elements for its
Glide dataframe and chart renderers; their existence does not establish the cause
of the reported text. The redesigned page also showed no `canvascanvas` text.
The original symptom remains unreproduced, with no CSS hiding or replacement
workaround applied. There is no custom CSS in this redesign.

Files changed specifically for this redesign:

| File | Purpose |
| --- | --- |
| `app/dashboard.py` | Page structure, upload state, existing prediction/metadata API calls, expanders |
| `app/__init__.py` | Importable presentation package |
| `app/ui/__init__.py` | Importable UI helper package |
| `app/ui/state.py` | Dataset identity, invalidation, existing request batching, ranking and KPIs |
| `app/ui/components.py` | Native cards/table/detail and fleet/sensor chart presentation |
| `.streamlit/config.toml` | Native light theme; no internal CSS selectors |
| `requirements-ops.txt` | Plotly dependency for the fleet interval chart |
| `Dockerfile` | Copy theme config into container working directory |
| `.dockerignore` | Allow that exact theme file into the otherwise allowlisted build context |
| `tests/test_dashboard.py` | Updated smoke assertions for the new header/disclaimer |
| `tests/test_dashboard_state.py` | State, invalidation, API parity, metadata failure, ranking, and interval tests |
| `docs/milestone_9_operations.md` | This verification and change record |

Docker packaging changes are necessary because the previous image copied only
`src/` and `app/`, and its build context excluded `.streamlit/`. Container execution
remains unverified locally because Docker is unavailable. No commits or pushes
were made. Local `tmp/redesign_baseline.json` and `tmp/redesign_hashes.json` retain
verification evidence; they are not runtime inputs.

### API references

- [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/)
- [FastAPI lifespan testing](https://fastapi.tiangolo.com/advanced/testing-events/)
- [MLflow tracking APIs](https://mlflow.org/docs/latest/ml/tracking/tracking-api)
- [Streamlit file uploader](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader)
- [Streamlit Docker deployment](https://docs.streamlit.io/deploy/tutorials/docker)
