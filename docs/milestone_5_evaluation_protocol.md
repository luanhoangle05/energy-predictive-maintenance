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
