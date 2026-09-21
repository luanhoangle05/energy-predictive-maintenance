# Milestone 6: Model Evaluation and Interpretation

## Evaluation scope

Results use 20 held-out development-validation engines from NASA
C-MAPSS FD001: 4,070 observations, including 620 observations with
true remaining useful life at or below 30 cycles.

The official NASA FD001 test set remains untouched.
These findings describe simulated turbofan data, not validated
performance on real industrial equipment.

## Current model decision

Retain Gradient Boosting with time_in_cycles as the leading
development candidate.

It achieved the strongest results among the five evaluated model
families across overall validation, near-failure evaluation, and
comparisons giving each engine equal weight.

| Metric | Gradient Boosting |
|---|---:|
| Overall MAE | 22.04 cycles |
| Overall RMSE | 29.03 cycles |
| Overall R² | 0.8045 |
| Overall NASA score (sum) | 137,156.12 |
| Near-failure MAE | 6.82 cycles |
| Near-failure RMSE | 8.84 cycles |
| Near-failure NASA score (sum) | 891.46 |
| Mean engine MAE | 22.23 cycles |
| Mean engine NASA penalty | 35.26 |
| Worst engine MAE | 47.09 cycles |

Overall and near-failure NASA totals cover different numbers of
observations and should not be compared directly.

## Explainability and ablation

time_in_cycles accounts for 56.89% of split-based feature importance.
Most other highly ranked features are five-cycle rolling sensor means.

Removing time_in_cycles while retaining the same model settings
increased validation MAE from 22.04 to 25.23 cycles and NASA score
from 137,156.12 to 894,648.17.

This supports retaining the feature under the current configuration
and development split. Split-based importance does not establish
causation or show the direction of a feature's effect.

## Failure cases and limitations

- Engine 45 was predominantly overestimated.
- Engine 84 was predominantly underestimated.
- Near-failure predictions had a positive mean error of 4.60 cycles.
- Performance varied substantially across engines and RUL ranges.
- Repeated inspection of the development-validation set means these
  results are exploratory; final performance requires the untouched
  official test set.
- The ablation used fixed hyperparameters and one development split.

## Verification

49 automated tests passed.

Coverage includes equal weighting across engines, endpoint selection,
preservation of temporal feature history, and prediction-target
alignment by engine ID.

## Evaluation readiness

Gradient Boosting with time_in_cycles remains the selected development
candidate. Removing time_in_cycles worsened performance under the
fixed-settings ablation.

The endpoint preparation and alignment helpers are implemented and
tested with synthetic data. Official NASA test evaluation is pending.

Before final fitting and official test evaluation, define the
engine-separated calibration split for Milestone 7 and freeze the
model configuration and evaluation procedure.

## Final NASA test protocol

Before accessing official test results, freeze the feature definitions,
model settings, preprocessing procedure, and prediction clipping rule.

1. Define the model-fitting and engine-separated calibration partitions.
   Reserve calibration engines for Milestone 7 uncertainty estimation.
   Never use official test engines for fitting or calibration.
2. Fit preprocessing and Gradient Boosting using model-fitting engines only.
3. Build temporal features using each test engine's complete observed
   history, ordered by cycle.
4. Apply the fitted preprocessing without refitting it.
5. Select the last observed feature row for each test engine.
6. Predict one RUL value per engine and clip negative predictions to zero.
7. Match predictions to RUL_FD001.txt by engine ID.
8. Verify that every test engine has exactly one prediction and one target.
9. Report endpoint MAE, RMSE, R², and summed NASA score.

These endpoint results measure a different evaluation setting from
the development metrics calculated across complete trajectories.

Official test results must not guide further feature selection,
hyperparameter tuning, or uncertainty calibration.