# Milestone 7: Uncertainty Calibration and Risk Assessment

## Development scope

This experiment uses NASA C-MAPSS FD001 simulated turbofan data.
The official NASA test set remains untouched.

The 100 development engines are separated into:
- 64 model-fitting engines: 13,160 observations.
- 16 calibration engines: one snapshot per engine.
- 20 evaluation engines: one snapshot per engine.

The outer engine split uses random_state=42. The calibration split
takes 20% of the remaining 80 engines, also using random_state=42.

## Features and model

Temporal features use current and previous cycles only.
Variance filtering and scaling are fitted on model-fitting engines
and reused for calibration and evaluation. There are 63 retained
features in this split.

Gradient Boosting uses the previously selected configuration:
- n_estimators=200
- learning_rate=0.05
- max_depth=2
- min_samples_leaf=30
- random_state=42

Point predictions are clipped to zero.

## Snapshot policy

Select one observed cycle per calibration and evaluation engine:
- At cycle 30 or later.
- Strictly before the engine's final cycle.
- Uniformly sampled from eligible rows.
- Calibration seed: 42; evaluation seed: 43.

Original row indexes preserve feature-target alignment.
Targets use the complete engine lifetime; features use only
information available at the selected cycle.

## Calibration method

The target coverage is 90%.

Calculate one absolute prediction error per calibration engine.
Select the sorted error at rank ceil((n + 1) * 0.90).

For 16 calibration engines, the rank is 16, so the radius is the
largest calibration error: approximately 74.92 cycles.

Intervals are:
- Lower bound: max(prediction - radius, 0).
- Upper bound: prediction + radius.

## Evaluation results

| Metric | Result |
|---|---:|
| Evaluation snapshots | 20 |
| Covered snapshots | 20/20 |
| Empirical coverage | 100% |
| Mean interval width | 137.30 cycles |
| Actual RUL below lower bound | 0 |
| Actual RUL above upper bound | 0 |

## Low-RUL threshold indicators

The illustrative threshold is 30 cycles.

Interval status:
- Entirely at or below threshold: 0 engines.
- Crosses threshold: 12 engines.
- Entirely above threshold: 8 engines.

A snapshot is flagged when its lower bound is at or below 30.

| Flag outcome | Count |
|---|---:|
| True positives | 4 |
| False positives | 8 |
| False negatives | 0 |
| True negatives | 8 |

Observed recall is 100% and precision is 33.3%.
These indicators describe interval positions, not failure probabilities
or validated maintenance recommendations.

## Limitations

- All development engines influenced earlier model fitting or selection.
  Separating engines now does not undo that historical reuse.
- Results are exploratory; 100% observed coverage across 20 snapshots
  does not establish coverage on new engines.
- Standard split-conformal guarantees require appropriate exchangeability
  and independence of calibration data from model fitting and selection.
  This experiment does not establish those conditions.
- Artificial cutoffs may differ from NASA's official test truncation.
- The intervals are broad. With this radius and non-negative predictions,
  no interval can lie entirely below the 30-cycle threshold.
- Four near-failure snapshots provide limited evidence about warning
  performance.
- No settings were changed in response to the reported interval results.

## Verification and remaining work

The user confirmed 89 automated tests passing across the full suite.

Tests cover engine splitting, cutoff selection and reproducibility,
calibration ranks, interval construction, threshold boundaries,
and invalid inputs.

Milestone 7 is complete for its exploratory development scope. The workflow
is implemented, reviewed, and tested. The full runner has been checked through
manual development runs; automated tests cover its component helpers rather
than the complete runner integration.
Official endpoint evaluation and final model fitting remain pending.
