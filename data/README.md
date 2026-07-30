# C-MAPSS Data Documentation

## Source

This project uses the **C-MAPSS Jet Engine Simulated Data** published through the [NASA Open Data Portal](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data).

C-MAPSS stands for Commercial Modular Aero-Propulsion System Simulation. The dataset was generated using a simulation of commercial aircraft turbofan engine degradation.

The dataset was accessed in July 2026. It is downloaded separately and is not included in this Git repository.

## Dataset description

C-MAPSS contains multivariate time-series data representing simulated aircraft turbofan engines operating over multiple cycles.

Each row represents one engine during one operating cycle. The training trajectories continue until simulated failure. The test trajectories stop before failure, and separate RUL files provide the true remaining cycle counts for their final observations.

Each training and test row contains 26 columns:

- One engine unit number
- One operating-cycle number
- Three operational settings
- Twenty-one sensor measurements

The dataset contains four subsets:

| Subset | Operating conditions | Fault modes |
|---|---:|---:|
| FD001 | 1 | 1 |
| FD002 | 6 | 1 |
| FD003 | 1 | 2 |
| FD004 | 6 | 2 |

The initial analysis and modeling scope of this project focuses on **FD001**, the simplest subset with one operating condition and one simulated fault mode.

## Raw file structure

The raw files are stored locally using the following structure:

```text
data/raw/cmapss/
|-- train_FD001.txt
|-- test_FD001.txt
|-- RUL_FD001.txt
|-- train_FD002.txt
|-- test_FD002.txt
|-- RUL_FD002.txt
|-- train_FD003.txt
|-- test_FD003.txt
|-- RUL_FD003.txt
|-- train_FD004.txt
|-- test_FD004.txt
|-- RUL_FD004.txt
|-- readme.txt
`-- Damage Propagation Modeling.pdf
```

The file prefixes have the following meanings:

- `train_`: complete simulated run-to-failure engine trajectories
- `test_`: trajectories that end before simulated failure
- `RUL_`: true remaining useful life values for the final observations in the test set

## Storage policy

Files under `data/raw/` are treated as original, immutable source data. They must not be manually edited or overwritten.

The raw dataset is excluded from Git because downloaded datasets should not be stored directly in the source-code repository.

The project uses these directories for different data stages:

- `data/raw/`: original downloaded NASA files
- `data/interim/`: intermediate data created during processing
- `data/processed/`: model-ready datasets created by reproducible code

The FD001 cleaning assessment found no missing values, infinite values, duplicate engine-cycle records, invalid identifiers, or cycle-sequence problems. Therefore, no cleaned copy was created in `data/interim/`.

## Important disclaimer

The C-MAPSS dataset represents **simulated aircraft turbofan engines**. It is not real Canadian energy-sector, oil and gas, power-generation, or industrial operating data.

This project presents Remaining Useful Life prediction as a methodology that may be relevant to other sensor-equipped rotating machinery, including turbines, compressors, pumps, and generators. However, results obtained from C-MAPSS must not be interpreted as validated performance on those types of equipment.

Using this methodology in a real industrial environment would require equipment-specific operating data, engineering domain knowledge, appropriate failure definitions, and independent operational validation.
