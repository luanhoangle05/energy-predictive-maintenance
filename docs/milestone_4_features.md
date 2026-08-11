# Milestone 4: Leakage-Safe RUL Targets and Features

## Scope

Milestone 4 constructs Remaining Useful Life targets, creates
engine-level time-series features, and prepares training and
validation data without information leakage.

The current scope uses the NASA C-MAPSS FD001 subset

## Engine-level splitting

Training and validation data are split by engine ID rather than individual rows

This ensures that all cycles belonging to one engine remain in the same dataset
No engine appears in both training and validation data

The default split uses:
- Validation size: 20%
- Random state = 42
- Shuffling: enabled at the engine-ID level

## Feature construction

The following columns are excluded from model inputs:
- unit number
- remaining_useful_life

The current cycle, operational setting, and raw sensor values remain available as model inputs

For every sensor, the workflow creates:
- Rolling mean over the current and previous four cycles
- Rolling std over the same five-cycle window
- Difference between the current and previous cycle

All calculation are grouped by engine ID
Rolling windows and differences restart when a new engine begins

The first cycle std and difference are set to 0 because no earlier observations are available

## Train-fitted preprocessing
The preprocessing pipeline contains:
- VarianceThreshold: to remove columns that are constant in the training data
- StandardScaler to center and scale retained features


The preprocessor has:
- .fit_transform(): used for training engines to calculate std+ mean then learn the pattern
- .transform(): used the same calculation and apply


