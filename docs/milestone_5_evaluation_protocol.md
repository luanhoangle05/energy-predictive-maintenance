# Milestone 5: Model Evaluation Protocol

## Initial target

The initial modeling target is uncapped Remaining Useful Life

# Development split

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