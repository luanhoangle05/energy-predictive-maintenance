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

43 automated tests passed, including a test confirming that
the engine-level model summary gives each engine equal weight.