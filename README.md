# Industrial Equipment Predictive Maintenance

An end-to-end data science project for predicting the remaining useful life (RUL) of rotating machinery from multivariate sensor data.

## Business problem

Unexpected failures of turbines, compressors, pumps, and generators can cause production losses, safety risks, and expensive repairs. RUL prediction estimates how many operating cycles remain before an asset reaches a defined failure point. A reliable estimate can support condition-based maintenance: teams can prioritize inspections, plan outages, and replace components before failure while avoiding unnecessary early maintenance.

This portfolio project focuses on the technical workflow behind that decision support. It does not prescribe maintenance actions, and model predictions would require engineering review and validation before operational use.

## Dataset disclaimer

This project uses NASA's C-MAPSS simulated turbofan engine degradation dataset. The original assets are **aircraft turbofan engines**, not oil and gas or Canadian energy-sector equipment, and the data is not real industrial operating data. The project presents the methodology as transferable to other sensor-equipped rotating machinery—such as turbines, compressors, pumps, and generators—while clearly separating that application framing from the dataset's actual origin.

The dataset is not included in this repository and is not downloaded during Milestone 1.

## Planned machine-learning pipeline

1. Ingest the C-MAPSS text files and assign documented column names.
2. Validate schema, data types, missing values, engine identifiers, and cycle order.
3. Calculate the RUL target for each engine cycle without leaking future information into model inputs.
4. Explore operating conditions and sensor degradation patterns.
5. Engineer reproducible time-series features using training data only where fitting is required.
6. Train baseline and tree-based regression models.
7. Evaluate models with regression metrics, NASA's asymmetric scoring function, and engine-level diagnostics.
8. Save selected artifacts and present results through clear figures and a lightweight application.

No models or analysis results are produced in this milestone.

## Repository structure

```text
energy-predictive-maintenance/
|-- app/                    # Future lightweight presentation application
|-- configs/                # Future project configuration files
|-- data/
|   |-- raw/                # Original, immutable dataset files (ignored)
|   |-- interim/            # Intermediate transformed data (ignored)
|   `-- processed/          # Model-ready data (ignored)
|-- models/                 # Trained model artifacts (ignored)
|-- notebooks/              # Exploratory analysis notebooks
|-- reports/
|   `-- figures/            # Generated report figures
|-- sql/                    # Reusable SQL queries
|-- src/
|   |-- data/               # Data loading and validation
|   |-- evaluation/         # Model evaluation
|   |-- features/           # Feature engineering
|   `-- models/             # Training and prediction code
|-- tests/                  # Automated tests
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

If PowerShell blocks the activation script, allow locally created scripts for your user and retry:

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
- [x] Milestone 3: Load, validate, and explore the data
- [x] Milestone 4: Build leakage-safe RUL targets and features
- Milestone 4 feature definitions and FD001 verification are documented in
[docs/milestone_4_features.md](docs/milestone_4_features.md).
- [ ] Milestone 5: Train baseline and candidate regression models
- [ ] Milestone 6: Evaluate, interpret, and compare models
- [ ] Milestone 7: Package the prediction workflow and presentation app
- [ ] Milestone 8: Finalize documentation and portfolio narrative

## License

Project code is available under the [MIT License](LICENSE). The NASA dataset is distributed separately and remains subject to its source terms.
