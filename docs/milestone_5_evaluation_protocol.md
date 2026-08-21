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

## Decision Tree tuning protocol

Decision Tree hyperparameters will be selected using only the training engines

Five-fold group cross-validation will be used. Engine ID will be the grouping variable so that every cycle belonging to one engine remains entirely within 1 fold

The held-out development-validation engines will not be used to select the Decision Tree hyperparameters
After selection, the chosen configuration will be fited using all training engines and evaluated once on the held-out validation engines

The NASA FD001 test trajectories remain untouched

### Initial hyperparameter grid

The initial experiment will control only two parameters:

| Hyperparameter | Candidate values |
|---|---|
| `max_depth` | 4, 8, 12 |
| `min_samples_leaf` | 1, 10, 30 |

This creates nine configurations. With five cross-validation folds, the experiment requires 45 model fits.

`max_depth` limits how many sequential splits the tree can create. The unrestricted baseline reached a depth of 31.

`min_samples_leaf` requires every terminal leaf to contain a minimum number of training observations. The unrestricted baseline created 14,300 leaves from 16,561 training rows.

Other tree hyperparameters will remain at their defaults during this initial experiment so that the comparison stays small and interpretable.

### Selection metric

Mean cross-validation RMSE will be the primary selection metric because it penalizes large RUL errors strongly.

Mean cross-validation MAE will also be reported for interpretability.

If configurations perform similarly, the simpler tree with lower depth and/or larger leaves will be preferred.

## Controlled Decision Tree results

Nine Decision Tree configurations were compared using five-fold cross-validation grouped by engine ID. Only the 80 training engines participated in hyperparameter selection.

The lowest mean cross-validation RMSE was obtained by all three depth-4 configurations:

| Max depth | Minimum leaf size | Mean CV MAE | Mean CV RMSE |
|---:|---:|---:|---:|
| 4 | 1 | 29.33 | 40.80 |
| 4 | 10 | 29.33 | 40.80 |
| 4 | 30 | 29.33 | 40.80 |

Deeper configurations produced higher cross-validation errors. Because the depth-4 configurations tied at the displayed precision, `max_depth=4` and `min_samples_leaf=30` were selected using the documented preference for the more strongly constrained configuration.

The selected tree was then fitted using all 80 training engines and evaluated once on the 20 outer validation engines.

| Dataset | MAE | RMSE | R² |
|---|---:|---:|---:|
| Training | 27.49 cycles | 38.37 cycles | 0.6964 |
| Validation | 25.36 cycles | 32.38 cycles | 0.7568 |

The selected tree had a fitted depth of 4 and 16 leaves, compared with depth 31 and 14,300 leaves for the unrestricted tree.

Complexity control substantially improved validation performance and removed the extreme training-data memorization. The selected Decision Tree performed slightly worse than clipped Linear Regression, so Linear Regression remains the strongest development baseline at this stage.

No NASA FD001 test trajectories were used for tuning or evaluation.

## Initial Random Forest experiment

Random Forest is the next candidate model because it averages predictions from many randomized Decision Trees. Averaging can reduce the high variance and overfitting observed in the unrestricted single-tree baseline.

The first experiment will use a reproducible, otherwise default Random Forest configuration:

| Hyperparameter | Initial value |
|---|---:|
| `n_estimators` | 100 |
| `max_depth` | None |
| `min_samples_leaf` | 1 |
| `max_features` | 1.0 |
| `bootstrap` | True |
| `random_state` | 42 |

`n_estimators=100` means that predictions are averaged across 100 trees. It primarily controls ensemble stability and computational cost.

The initial Random Forest will be evaluated using the same five-fold cross-validation grouped by engine ID. Only the 80 training engines will participate.

Mean cross-validation RMSE remains the primary comparison metric, with mean MAE also reported.

The outer 20 validation engines will remain untouched until an initial Random Forest configuration has been assessed and any controlled tuning choices have been made.

The NASA FD001 test data will not be used.

## Initial Random Forest cross-validation

The initial 100-tree Random Forest used default tree complexity and was evaluated using the same five engine-grouped training folds.

| Model | Mean CV MAE | Mean CV RMSE |
|---|---:|---:|
| Selected Decision Tree | 29.33 cycles | 40.80 cycles |
| Initial Random Forest | 28.57 cycles | 41.28 cycles |

The initial Random Forest slightly improved MAE but produced a slightly higher RMSE. It therefore did not improve the primary selection metric.

A controlled Random Forest experiment will investigate:

| Hyperparameter | Candidate values |
|---|---|
| `max_depth` | 4, 8, None |
| `min_samples_leaf` | 1, 10, 30 |

`n_estimators` will remain fixed at 100, and `max_features` will remain fixed at 1.0. This isolates tree-complexity effects and creates nine configurations.

The outer validation engines and NASA test data remain untouched during tuning.