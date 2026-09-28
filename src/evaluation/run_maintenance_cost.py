"""Apply hypothetical cost weights to saved maintenance outcomes."""

from pathlib import Path

import pandas as pd

from src.evaluation.maintenance_cost import calculate_maintenance_cost


def main() -> None:
    input_path = Path(
        "reports/maintenance/engine_timing_outcomes.csv"
    )

    outcomes = pd.read_csv(
        input_path,
        dtype={"preventive_maintenance_success": "string"},
    )

    # Parse explicitly: bool("False") would incorrectly produce True.
    success_values = outcomes["preventive_maintenance_success"]

    if not success_values.isin(["True", "False"]).all():
        raise ValueError("Success flags must contain True or False")

    success_flags = success_values.map({
        "True": True,
        "False": False,
    })

    costed_outcomes = outcomes.copy()
    costed_outcomes["total_cost"] = [
        calculate_maintenance_cost(
            preventive_maintenance_success=success,
            discarded_useful_life=discarded,
            preventive_cost=1.0,
            failure_cost=10.0,
            discarded_cycle_cost=0.01,
        )
        for success, discarded in zip(
            success_flags,
            outcomes["discarded_useful_life"],
        )
    ]

    print("Costed outcome rows:", len(costed_outcomes))
    print("\nEngine 1 hypothetical costs:")
    print(
        costed_outcomes.loc[
            costed_outcomes["unit_number"] == 1,
            ["unit_number", "policy", "total_cost"],
        ].to_string(index=False)
    )

    cost_summary = costed_outcomes.groupby("policy").agg(
        engines=("unit_number", "nunique"),
        total_cost=("total_cost", "sum"),
        mean_cost_per_engine=("total_cost", "mean"),
    )

    print("\nHypothetical cost summary (arbitrary cost units):")
    print(cost_summary.round(4).to_string())

    output_directory = input_path.parent

    costed_outcomes.to_csv(
        output_directory / "engine_cost_outcomes.csv",
        index=False,
    )

    cost_summary.to_csv(
        output_directory / "cost_summary.csv",
        index=True,
        index_label="policy",
    )

    print(f"\nCost results saved to: {output_directory.resolve()}")

    scenario_summaries = []

    for failure_cost in (5.0, 10.0, 20.0):
        for discarded_cycle_cost in (0.0, 0.01, 0.05):
            scenario_outcomes = outcomes.copy()

            scenario_outcomes["total_cost"] = [
                calculate_maintenance_cost(
                    preventive_maintenance_success=success,
                    discarded_useful_life=discarded,
                    preventive_cost=1.0,
                    failure_cost=failure_cost,
                    discarded_cycle_cost=discarded_cycle_cost,
                )
                for success, discarded in zip(
                    success_flags,
                    outcomes["discarded_useful_life"],
                )
            ]

            scenario_summary = (
                scenario_outcomes.groupby("policy")
                .agg(
                    engines=("unit_number", "nunique"),
                    total_cost=("total_cost", "sum"),
                    mean_cost_per_engine=("total_cost", "mean"),
                )
                .reset_index()
            )

            scenario_summary["preventive_cost"] = 1.0
            scenario_summary["failure_cost"] = failure_cost
            scenario_summary["discarded_cycle_cost"] = discarded_cycle_cost

            scenario_summaries.append(scenario_summary)

    sensitivity_summary = pd.concat(
        scenario_summaries,
        ignore_index=True,
    )

    print("\nSensitivity summary rows:", len(sensitivity_summary))
    print(sensitivity_summary.round(4).to_string(index=False))

    sensitivity_path = output_directory / "cost_sensitivity_summary.csv"

    sensitivity_summary.to_csv(
        sensitivity_path,
        index=False,
    )

    print(f"\nSensitivity results saved to: {sensitivity_path.resolve()}")


if __name__ == "__main__":
    main()