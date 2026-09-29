# Streamlit product pass — September 28, 2026

## Current experience

Streamlit remains the presentation layer, FastAPI remains the inference layer,
and the existing saved preprocessing/model bundle produces every prediction.
The four tabs are Fleet Overview, Engine Explorer, Maintenance Analysis, and
Model Insights. Input selection stays above the tabs. The page uses native
Streamlit layout and theme settings, without custom HTML or CSS.

Start the existing API and dashboard from the repository root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/dashboard.py --server.address 127.0.0.1
```

Visit http://127.0.0.1:8501 and click **Try demo fleet**. The alternative
**Upload FD001-compatible telemetry** retains existing CSV/TXT validation.
Uploaded measurements must be compatible simulated FD001 histories; this is
not a universal model for industrial machinery. Predictions, bounds, and flags
retain full precision in the downloaded CSV; display rounding is presentation only.

## Demo construction and scope

The demo reads only `data/raw/cmapss/train_FD001.txt`. It reuses Milestone 7's
`select_cutoff_indices`, minimum cycle 30, with seed 43 fixed independently of
model predictions. One pre-failure cutoff is selected per development engine.
Histories are truncated **before** the existing inference request; only the 26
raw observation columns through each cutoff are sent. No targets, post-cutoff
rows, or remaining-life labels enter inference. Complete lifetimes are used only
to establish eligible retrospective demo cutoffs, as in Milestone 7.

The current source yields 100 engines and 11,913 observations. Retrospective
verification found 26 engines observed before 40% of their complete lifetime,
37 between 40% and 70%, and 37 at or beyond 70%, including 20 with <=30 actual
cycles remaining. These are descriptive verification counts, not risk categories
or inputs to inference. The unchanged model returns 55 attention flags for this
demo. Neither cutoffs nor predictions were tuned to achieve that count.

The demo is not an independent benchmark; it reuses development data. The
official NASA test set was not read or used. Missing training data gives an
explicit demo-unavailable message and never falls back to official test files.
Normal compatible uploads remain usable when the demo source is absent.

Source changes, same-name file replacements, invalid replacements, and removal
clear prediction and selection state. Engine/sensor selection and tab navigation
preserve existing predictions without resubmitting inference.

## Model and maintenance information

The additive read-only `GET /insights` endpoint returns the loaded bundle's
feature names, existing split-based importance values, retained feature count,
and model identifier. It does not fit, transform, or predict. `/predict` and
`/metadata` retain their previous behavior. The UI checks the returned model ID
against the prediction result and can still show predictions if insights fail.

Model Insights displays the serving model's recorded snapshot metrics from
metadata, explains each measure, and separately labels the earlier Milestone 6
all-cycle results (including near-failure MAE) as a different development fit
and evaluation scope. Importance does not imply causality or effect direction.
The uncertainty explanation uses 16 calibration engines, a 90% target, rank
ceil(17 * 0.9) = 16, and the existing 74.915095237546-cycle radius. Development
coverage is exploratory, with no real-world guarantee.

Maintenance Analysis reads the existing `cost_summary.csv` and
`timing_summary.csv` from `reports/maintenance/`. Its comparison is the saved
20-engine retrospective Milestone 8 study, **not** a simulation for the active
upload or demo. No policy or cost calculation runs in the dashboard. Displayed
mean costs are the saved values, with labels rounded for readability. Frozen
trigger, lead-time, and arbitrary cost assumptions remain documented in an
expander. The tab is omitted if the saved summaries are unavailable or malformed.

Compose now mounts the specific development training file and the maintenance
report directory read-only into the dashboard. Dataset contents are not baked
into the image. The training file must exist before Compose startup; its bind
mount disables automatic host-path creation. Docker execution remains unverified
locally because Docker is unavailable. Native upload-only use needs no demo file.

## Verification

- Full suite: 170 tests passed; one existing upstream Starlette/httpx warning.
- Tests cover deterministic prefixes, absence of future rows/targets in inference,
  unchanged output under altered future sensor values, exact request contents,
  ranking/top-N, formatting, <=30 attention semantics, source changes, selectors,
  missing-demo handling, fitted importance, and saved maintenance-value parity.
- Before/after live regression on `dashboard_sample.csv`: all three predictions,
  lower/upper bounds, and Boolean flags matched exactly, including full precision.
- Hashes confirmed the saved model, all preprocessing/feature/evaluation modules,
  and maintenance CSVs were unchanged. The only previously existing `src` file
  changed was `src/api/main.py` to add the read-only endpoint.
- Browser checks: demo button, compatible CSV upload, upload analysis, engine
  selection for both sources, tab navigation, and return from upload to demo.
- Desktop 1280px and narrow 640px checks: readable chart labels/reference line,
  responsive cards, usable tabs/selectors/tables, and no page-level horizontal
  overflow. Sensor charts show observed values only, with cycle/value hover.
- `canvascanvas` did not appear in inspected screenshots or rendered page text,
  either before or after this pass. No hiding workaround was introduced.
- No retraining, official test evaluation, commit, merge, or push was performed.

## Files changed in this pass

| File | Reason |
| --- | --- |
| `app/dashboard.py` | Compact header, demo/upload paths, tabs, state integration, optional insights requests |
| `app/ui/components.py` | Plain-language metrics, point/range fleet chart, priority table, endpoint cards, Plotly sensor hover |
| `app/ui/state.py` | New presentation labels and invalidation of fetched insights |
| `app/ui/demo.py` | Fixed development source and reuse of existing cutoff helper before inference |
| `app/ui/formatting.py` | Read-only status labels, range strings, and readable table columns |
| `app/ui/insights.py` | Existing feature importance, scoped performance results, uncertainty and limitations |
| `app/ui/maintenance.py` | Read-only saved retrospective maintenance comparison |
| `src/api/main.py` | Additive `/insights` endpoint; existing inference and metadata handlers unchanged |
| `.streamlit/config.toml` | Smaller native metric text to keep range/status cards readable |
| `compose.yaml` | Read-only mounts needed by demo and saved-report presentation |
| `tests/test_dashboard.py` | Compact-header smoke check |
| `tests/test_dashboard_state.py` | Updated labels, upload path, and insight response mock |
| `tests/test_dashboard_product.py` | Demo isolation, presentation semantics, source-switch, missing-data and saved-report tests |
| `tests/test_serving.py` | Fitted importance and unchanged metadata response check |
| `docs/dashboard_product_pass.md` | This usage, scope, and verification record |
| `docs/milestone_9_operations.md` | Link to the current product-pass documentation |

Local `tmp/product_baseline.json` and `tmp/product_hashes.json` are verification
evidence only, not application inputs. Existing unrelated working-tree changes
were preserved. Dockerfile, .dockerignore, and requirements were not changed in
this pass; the already installed Plotly dependency is reused.

Deferred: interactive cost controls, live maintenance simulation for uploaded
fleets, historical RUL inference, public hosting, and official endpoint evaluation.
These are not needed to present the existing model and its limitations accurately.
