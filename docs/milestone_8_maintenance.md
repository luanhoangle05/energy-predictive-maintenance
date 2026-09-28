# Milestone 8: Maintenance Timing and Hypothetical Cost Simulation

## Status and scope

Milestone 8 is in progress. Timing/outcome logic is implemented for four
policies, tested with synthetic examples, and applied to 20 development
evaluation engines. The first hypothetical cost scenario is implemented
and applied to the saved timing outcomes without changing policy decisions.

This experiment uses NASA C-MAPSS FD001 simulated turbofan data.
Results illustrate decision-support methodology, not real maintenance
recommendations.

The official NASA FD001 test set remains untouched.

## Frozen development workflow

The Milestone 7 engine split is retained:

- 64 model-fitting engines: 13,160 rows.
- 16 calibration engines: 3,401 complete-history rows.
- 20 evaluation engines: 4,070 rows.

Features use current and previous observations only. Preprocessing is
fitted on the 64 model-fitting engines and reused for the other partitions.

The Gradient Boosting configuration remains unchanged:

- n_estimators=200
- learning_rate=0.05
- max_depth=2
- min_samples_leaf=30
- random_state=42

Predictions are clipped to zero. Calibration uses one snapshot per
calibration engine, selected with seed 42 at cycle 30 or later and
strictly before failure. The 90% target interval radius is approximately
74.92 cycles.

## Frozen timing rules

- Monitoring begins at cycle 30, inclusive.
- Maintenance lead time is 5 cycles.
- The RUL trigger threshold is 30 cycles.
- Threshold comparison uses <=.
- Only the first eligible trigger is used.
- Scheduled maintenance cycle = trigger cycle + lead time.
- Preventive maintenance succeeds only when maintenance occurs strictly
  before failure.
- Maintenance at or after failure counts as failure.
- No trigger means the engine runs to failure.
- Actual failure time is used for retrospective scoring, not to select
  the RUL trigger or adjust the predefined fixed age.
- Discarded useful life equals failure cycle minus maintenance cycle
  after successful preventive maintenance; otherwise it is zero.

A trigger on the failure cycle may be recorded, but cannot prevent failure.
The maintenance_cycle output represents the scheduled cycle, even when
failure prevents maintenance from occurring.

## Policies

1. Run to failure: never triggers preventive maintenance.
2. Fixed age: triggers at cycle 150 if the engine reaches that cycle;
   maintenance is scheduled for cycle 155. This age was frozen before
   inspecting evaluation-policy outcomes.
3. Predicted RUL: triggers at the first eligible prediction <= 30.
4. Lower bound: triggers at the first eligible interval lower bound <= 30.

Each policy is simulated independently for each engine. The simulation
ends at successful preventive maintenance or failure. Replacement lives
and repeated repairs are not modeled.

## Development timing results

| Policy | Engines | Preventive successes | Failures | Total discarded cycles | Mean discarded cycles, all engines | Mean discarded cycles, successes only |
|---|---:|---:|---:|---:|---:|---:|
| Run to failure | 20 | 0 | 20 | 0 | 0.00 | N/A |
| Fixed age | 20 | 18 | 2 | 991 | 49.55 | 55.06 |
| Predicted RUL | 20 | 20 | 0 | 422 | 21.10 | 21.10 |
| Lower bound | 20 | 20 | 0 | 1932 | 96.60 | 96.60 |

Under these frozen settings, predicted RUL produced zero failures and
discarded less useful life than the other preventive policies on this
development sample.

The lower-bound policy also produced zero failures but discarded an
additional 75.50 cycles per engine compared with predicted RUL.

These observations do not establish performance on new engines or
economic savings.

## Saved outputs and reproduction

From the repository root:

    python -m src.evaluation.run_maintenance

This reruns fitting with the frozen configuration and overwrites:

- reports/maintenance/engine_timing_outcomes.csv
- reports/maintenance/timing_summary.csv

The outcome file contains 80 rows: 20 engines times four policies.
The summary contains four policy rows. The cost runner uses these saved
outcomes without refitting the model.

## Verification

The user reported:

- 32 maintenance timing tests passing.
- 12 hypothetical cost tests passing.
- 133 tests passing across the full project suite.

Tests cover first-trigger selection, threshold equality, monitoring start,
timing boundaries, no-trigger outcomes, all four policies, and selected
invalid-input cases. Cost tests cover successful maintenance charges,
failure-only charges, and rejection of negative costs and discarded life.

The development runner was executed manually. Saved CSV checks confirmed
80 outcome rows, four summary rows, no duplicate engine-policy pairs,
and consistent success/failure and discarded-life calculations.

The complete model-to-CSV runner does not yet have an automated
integration test.

## Limitations and remaining work

- Development engines influenced earlier fitting or model selection;
  the current split does not undo that historical reuse.
- Twenty evaluation engines provide limited evidence.
- Snapshot calibration does not establish simultaneous trajectory
  coverage or coverage at a policy-selected stopping time.
- Broad intervals can cause substantially earlier maintenance.
- Preventive success means maintenance precedes failure; it does not
  establish that the timing is economically preferable.
- Lead time, monitoring start, and trigger settings are illustrative
  assumptions, not validated operational choices.
- This is a single-life simulation without fleet availability,
  replacement operation, or maintenance-resource constraints.
- Cost sensitivity covers nine predefined illustrative scenarios.
  It does not establish validated economics or an optimal policy.
- Final fitting and official NASA endpoint evaluation remain pending.

## Initial hypothetical cost scenario

Frozen weights:

- Successful preventive maintenance: 1.0 cost unit.
- Failure: 10.0 cost units for the entire failure event.
- Discarded useful life: 0.01 cost units per cycle, charged only after
  successful preventive maintenance.
- Trigger/alarm: 0.0 cost units.

Successful maintenance costs 1.0 + 0.01 × discarded useful-life cycles.
Failure costs 10.0, with no additional preventive maintenance charge.

These are arbitrary decision-support scenario weights, not real prices,
estimated savings, downtime or safety costs, or validated maintenance
economics. They do not change predictions or timing decisions.

| Policy | Engines | Total cost units | Mean cost units per engine |
|---|---:|---:|---:|
| Run to failure | 20 | 200.00 | 10.0000 |
| Fixed age | 20 | 47.91 | 2.3955 |
| Predicted RUL | 20 | 24.22 | 1.2110 |
| Lower bound | 20 | 39.32 | 1.9660 |

Predicted RUL has the lowest hypothetical cost under these weights on
this development sample. This does not establish real economic savings.

Run the cost calculation from the repository root:

    python -m src.evaluation.run_maintenance_cost

It reads the saved timing outcomes without retraining and overwrites:

- reports/maintenance/engine_cost_outcomes.csv
- reports/maintenance/cost_summary.csv

Verification confirmed 80 cost outcomes, four summary rows, correct
per-outcome costs and summary calculations, and unchanged timing fields.

## Hypothetical cost sensitivity

Nine scenarios were evaluated using the same 80 saved timing outcomes:

- Preventive maintenance cost: fixed at 1.0.
- Failure cost: 5.0, 10.0, or 20.0.
- Discarded-cycle cost: 0.0, 0.01, or 0.05.
- Trigger/alarm cost: fixed at zero.

No model refitting, timing changes, or policy optimization was performed.

Predicted RUL had the lowest hypothetical cost in six scenarios and tied
with the lower-bound policy in the three zero-discarded-cost scenarios.
The ties occur because both policies had 20 preventive successes and
zero failures, while early replacement carried no charge.

With failure cost 5.0 and discarded-cycle cost 0.05, the lower-bound
policy cost 116.60 units, exceeding run-to-failure at 100.00 units.
This illustrates how assumptions about discarded life affect comparisons.

These findings apply only to the selected weights and development
outcomes. They do not establish real savings or future policy performance.

The cost runner also saves:

- reports/maintenance/cost_sensitivity_summary.csv

The saved file was checked against the timing outcomes: 36 rows,
nine scenarios, four policies per scenario, no duplicate scenario-policy
pairs, and matching total and mean costs.
