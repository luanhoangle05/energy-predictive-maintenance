# Milestone 5: Model Evaluation Protocol

## Initial target

The initial modeling target is uncapped Remaining Useful Life

## Development split

Complete engine trajectories are split by engine ID

- Training engines are used to fit preprocessing and models
- Validation engines are used to evaluate model performance
- No engine may appear in both datasets

## Initial evaluation unit

- Initial development evaluation uses every cycle from the held-out validation engines
- This measures how well the model predicts RUL across complete engine trajectories.
- The final row alone is not used as an endpoint metric because every complete run-to-failure validation engine has a final RUL of zero.

## Final test protection

The NASA FD001 test trajectories and `RUL_FD001.txt` are kept untouched during development

They will be used only for final endpoint evaluation after model and feature decisions are complete

## Initial evaluation metrics

The first model comparison uses predictions from every cycle belonging to the held-out validation engines

### Mean Absolute Error

MAE measures the average absolute difference between predicted and actual RUL

It is expressed in operating cycles and is easy to interpret

Lower values indicate better predictions

### Root Mean Squared Error

RMSE calculates the square root of the average squared prediction error

It penalizes large prediction errors more strongly than MAE

Lower values indicate better predictions

### R-squared

R² measures model performance relative to predicting a constant mean value

Higher values indicate better performance

An R² value can be negative when a model performs worse than the reference mean prediction

## Initial evaluation limitation

Metrics calculated across every validation row give more influence to engines with longer trajectories because they contribute more cycles.

Engine-level diagnostics and near-failure analysis will be added later to ensure that overall row-level metrics do not hide poor performance on particular engines or important RUL ranges.

## Mean baseline results

The mean `DummyRegressor` was fitted using the training engines and evaluated across every cycle from the held-out validation engines.

| Setting | Value |
|---|---:|
| Validation size | 20% of engine IDs |
| Random state | 42 |
| Training rows | 16,561 |
| Validation rows | 4,070 |
| Retained features | 63 |
| Constant prediction | 108.38 cycles |

| Metric | Result |
|---|---:|
| MAE | 55.36 cycles |
| RMSE | 65.72 cycles |
| R² | -0.0019 |

The mean baseline does not use sensor information to vary its predictions. It establishes a reference that future models should improve upon by achieving lower MAE and RMSE and a higher R².

These are development-validation results, not final NASA FD001 test results.

### Linear Regression baseline results

Linear Regression was fitted using the same training engines and evaluated on the same validation cycles as the mean baseline

The raw model produced 162 negative predictions from 4,070 validation rows, approx 4.0% of its predictions
Its prediction range was -57.84 to 256.17 cycles
while the true validation target range was 0 to 268 cycles.

Because Remaining Useful Life (RUL) can not be negative, a non-negative post-processing rule was also evaluated by cutting predictions below 0 to 0
This threshold constraint rather than a value tuned from validation performance

| Model output | MAE | RMSE | R² |
|---|---:|---:|---:|
| Raw Linear Regression | 25.18 cycles | 31.68 cycles | 0.7672 |
| Clipped at zero | 24.47 cycles | 31.25 cycles | 0.7734 |

For transparency, both raw and clipped results are retained
The clipped result represents the domain-valid Linear Regression output for later model comparison

Compared with the mean baseline (Dummy Regressor), the clipped Linear Regression result reduced MAE by approx 55.8% and RMSE by approx 52.4%

These remain development-validation results and do not use the NASA FD001 test set.

## Unrestricted Decision Tree baseline

An unrestricted `DecisionTreeRegressor` was fitted using the same training engines and evaluated one the same validation engines

| Diagnostic | Result |
|---|---:|
| Maximum fitted depth | 31 |
| Leaf nodes | 14,300 |
| Training rows | 16,561 |

| Dataset | MAE | RMSE | R² |
|---|---:|---:|---:|
| Training | 0.00 cycles | 0.00 cycles | 1.0000 |
| Validation | 34.30 cycles | 48.11 cycles | 0.4630 |

The unrestricted tree is overfitting ( MAE, RMSE =0 and R^2=1) but performed kinda worse on unseen validation engines

The tree performed better than the mean baseline but worse than Linear Regression on validation data. This provides evidence that tree complexity should be controlled before evaluating more advanced tree ensembles.

The first tuning investigation will focus on `max_depth` and `min_samples_leaf`, which directly limit tree complexity. Hyperparameter choices will not use the NASA FD001 test data.