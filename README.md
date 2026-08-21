# Industrial Equipment Predictive Maintenance

An end-to-end machine-learning project for estimating the remaining useful life
(RUL) of rotating machinery from multivariate time-series sensor data.

The current development scope uses the FD001 subset of NASA's C-MAPSS dataset.
The project begins with reproducible RUL prediction and will later extend into
uncertainty estimation, risk assessment, maintenance-policy simulation,
explainability, and deployment.

## Business problem

Unexpected failures of turbines, compressors, pumps, and generators can cause
production losses, safety risks, and expensive repairs. RUL prediction estimates
how many operating cycles remain before an asset reaches a defined failure
point. A reliable estimate can support condition-based maintenance by helping
teams prioritize inspections, plan outages, and replace components before
failure while avoiding unnecessary early maintenance.

This portfolio project focuses on the technical workflow behind that decision
support. It does not prescribe real maintenance actions. Any operational use
would require equipment-specific data, engineering review, uncertainty analysis,
and independent validation.

## Dataset disclaimer

This project uses NASA's C-MAPSS simulated turbofan engine degradation dataset.
The original assets are **simulated aircraft turbofan engines**, not oil and gas,
Canadian energy-sector, or real industrial equipment. Results from this project
must not be interpreted as validated performance for turbines, compressors,
pumps, generators, or other real-world machinery.

The project presents the methodology as potentially transferable to other
sensor-equipped rotating machinery while keeping that application framing
separate from the dataset's actual origin.

The dataset is downloaded separately and is not committed to this repository.
The current modeling scope is **FD001**, which contains one simulated operating
condition and one simulated fault mode.

## Current status

- **Milestones 1-4 are complete:** project setup, dataset documentation,
  validation and exploration, leakage-safe targets, temporal features, and
  preprocessing.
- **Milestone 5 is in progress:** the mean baseline, Linear Regression,
  Decision Tree, and Random Forest models are implemented and tested.
- Decision Tree and Random Forest complexity settings are compared using
  five-fold cross-validation grouped by engine ID.
- Preprocessing is fitted separately inside each cross-validation training
  fold before transforming its validation engines.
- The controlled Random Forest is the strongest current development candidate:
  MAE 23.74 cycles, RMSE 30.52 cycles, and R-squared 0.7838.
- These are row-level development-validation results. NASA's official FD001
  test data remains untouched.
- **Next task:** evaluate Gradient Boosting using the same leakage-safe
  protocol.
- The project currently contains 28 passing automated tests.

## Current machine-learning workflow

1. Load the C-MAPSS text files and assign documented column names.
2. Validate required columns, missing and infinite values, duplicate
   engine-cycle records, and cycle order.
3. Explore FD001 engine lifetimes, operating settings, sensor variability, and
   degradation patterns.
4. Calculate an **uncapped RUL target** for every training cycle.
5. Split the development data by `unit_number` so complete engine trajectories
   remain together.
6. Create temporal features independently for each engine using only the
   current and previous cycles.
7. Fit variance filtering and feature scaling on training engines only, then
   reuse the fitted preprocessing pipeline for validation data.
8. Train the mean, Linear Regression, Decision Tree, and Random Forest models.
9. Tune tree complexity using five-fold grouped cross-validation with
   fold-local preprocessing.
10. Fit selected configurations using all training engines and evaluate them
    on the held-out development-validation engines.

```text
Raw FD001 training data
        |
        v
Validate and add uncapped RUL
        |
        v
Split complete engines into training and validation
        |
        v
Build per-engine temporal features
        |
        v
Fit outer preprocessing on training engines only
        |
        v
Train baselines and candidate models
        |
        v
Tune with engine-grouped folds and fold-local preprocessing
        |
        v
Evaluate selected configurations on held-out validation engines
```

## Leakage-safe feature preparation

The current feature workflow:

- Retains the current cycle, operational settings, and raw sensor values.
- Adds rolling sensor means over the current and previous four cycles by
  default.
- Adds rolling sensor standard deviations over the same window.
- Adds cycle-to-cycle sensor differences.
- Restarts every rolling calculation and difference at each engine boundary.
- Excludes `unit_number` and `remaining_useful_life` from model inputs.
- Removes features that are constant in the training split.
- Standardizes retained features using statistics learned from training data
  only.

The feature definitions and FD001 verification are documented in
[docs/milestone_4_features.md](docs/milestone_4_features.md).

## Evaluation protocol

The current evaluation protocol follows these rules:

- Use uncapped RUL as the first target definition.
- Keep complete engines separated between training and validation.
- Keep NASA's official FD001 test trajectories and `RUL_FD001.txt` untouched
  while choosing features and models.
- Evaluate predictions across all validation cycles using MAE, RMSE, and
  R-squared.
- Tune Decision Tree and Random Forest complexity using five-fold grouped
  cross-validation within the 80 training engines.
- Fit variance filtering and scaling separately inside each cross-validation
  training fold before transforming that fold's validation engines.
- Fit selected configurations using all 80 training engines, then evaluate
  once on the 20 held-out development-validation engines.
- Add NASA's asymmetric score, near-failure performance, and engine-level
  diagnostics after the current model comparison is complete.

The last row of each complete run-to-failure validation engine is not a useful
endpoint benchmark because every such row has true RUL equal to zero. Official
endpoint evaluation will instead use the last observed row of each truncated
NASA test trajectory together with `RUL_FD001.txt`. If endpoint-like validation
is needed during development, leakage-safe artificial cutoff snapshots will be
created later.

Future prediction-uncertainty calibration will also use engine-separated
calibration data. The final NASA test set will not be reused for calibration.

## Repository structure

```text
energy-predictive-maintenance/
|-- app/                    # Future Streamlit presentation application
|-- configs/                # Future reproducible project configuration
|-- data/
|   |-- raw/                # Original, immutable source files (ignored)
|   |-- interim/            # Intermediate transformed data (ignored)
|   `-- processed/          # Model-ready data (ignored)
|-- docs/                   # Dataset and milestone documentation
|-- models/                 # Trained model artifacts (ignored)
|-- notebooks/              # Exploratory analysis notebooks
|-- reports/
|   `-- figures/            # Generated evaluation figures
|-- sql/                    # Future reusable SQL queries
|-- src/
|   |-- data/               # Loading, validation, targets, splitting, preparation
|   |-- evaluation/         # Evaluation metrics and diagnostics
|   |-- features/           # Temporal feature engineering and preprocessing
|   `-- models/             # Baseline and candidate-model training and prediction
|-- tests/                  # Automated unit and workflow tests
|-- .gitignore
|-- LICENSE
|-- README.md
`-- requirements.txt
```

## Setup (Windows PowerShell, Python 3.10)

From the repository root, confirm that the Python launcher can find Python 3.10:

```powershell
py -3.10 --version
```

Create and activate a local virtual environment:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks the activation script, allow locally created scripts for
your user and retry:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Run the tests:

```powershell
python -m pytest
```

Deactivate the environment when finished:

```powershell
deactivate
```

## Roadmap

- [x] Milestone 1: Create the project scaffold and development foundation
- [x] Milestone 2: Acquire and document the C-MAPSS dataset
- [x] Milestone 3: Load, validate, clean, and explore FD001
- [x] Milestone 4: Build leakage-safe RUL targets, features, and preprocessing
- [ ] Milestone 5: Establish trustworthy baselines and train candidate models
  - Mean, Linear Regression, controlled Decision Tree, and controlled Random
    Forest implemented
  - MAE, RMSE, and R-squared implemented
  - Engine-grouped cross-validation uses fold-local preprocessing
  - Random Forest is the strongest current development candidate
  - Gradient Boosting is next
- [ ] Milestone 6: Evaluate, interpret, and compare models
  - Core regression metrics
  - NASA asymmetric score
  - Near-failure and engine-level diagnostics
  - Explainability and failure-case analysis
- [ ] Milestone 7: Add engine-separated uncertainty calibration and risk
  assessment
- [ ] Milestone 8: Simulate maintenance timing and hypothetical cost policies
- [ ] Milestone 9: Operationalize the workflow with MLflow, FastAPI, Streamlit,
  Docker, automated testing, CI, and deployment
- [ ] Milestone 10: Finalize documentation, limitations, and portfolio narrative

Future maintenance and cost outputs will be explicitly presented as hypothetical
decision-support simulations, not validated maintenance recommendations.

## License

Project code is available under the [MIT License](LICENSE). The NASA dataset is
distributed separately and remains subject to its source terms.
