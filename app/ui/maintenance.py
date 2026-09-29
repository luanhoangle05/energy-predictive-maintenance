"""Read-only presentation of saved Milestone 8 summaries, not live policy logic."""

from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPORTS = Path(__file__).resolve().parents[2] / "reports/maintenance"
POLICIES = {"run_to_failure": "Run to failure", "fixed_age": "Fixed age",
            "predicted_rul": "Predicted RUL", "lower_bound": "Lower-bound exploratory policy"}


def load_maintenance_reports(root=REPORTS):
    if not all((root / name).is_file() for name in ("cost_summary.csv", "timing_summary.csv")):
        return None
    cost = pd.read_csv(root / "cost_summary.csv")
    timing = pd.read_csv(root / "timing_summary.csv")
    merged = cost.merge(timing, on=["policy", "engines"], validate="one_to_one")
    if set(merged.policy) != set(POLICIES) or len(merged) != 4:
        raise ValueError("Saved maintenance summaries must contain the four documented policies.")
    return merged.set_index("policy").loc[list(POLICIES)].reset_index()


def render_maintenance(report):
    st.subheader("Hypothetical Maintenance Simulation")
    st.caption("Saved Milestone 8 study · 20 development-evaluation engines · independent of your selected fleet")
    st.write("These retrospective simulations compare frozen maintenance rules. Timing assumptions "
             "are hypothetical; costs are arbitrary scenario weights, not money or real savings.")
    fig = go.Figure(go.Bar(
        x=report.mean_cost_per_engine.tolist(), y=[POLICIES[p] for p in report.policy], orientation="h",
        marker_color="#2563EB", text=[f"{v:.2f}" for v in report.mean_cost_per_engine], textposition="auto",
        hovertemplate="%{y}<br>%{x:.3f} illustrative cost units per engine<extra></extra>",
    ))
    fig.update_layout(height=300, margin=dict(l=0, r=10, t=10, b=0),
                      xaxis_title="Illustrative cost units per engine", yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption("Outputs are not operational maintenance recommendations. No policies are rerun for uploads or the demo.")
    with st.expander("Frozen assumptions and saved outcomes"):
        st.write("Monitoring starts at cycle 30; lead time is 5 cycles. Fixed age triggers at "
                 "150; predicted-RUL and lower-bound policies trigger at their first value ≤ 30. "
                 "Maintenance must occur strictly before failure to succeed. Initial costs: "
                 "successful preventive action 1, failure 10, discarded useful cycle 0.01.")
        st.dataframe(report.rename(columns={"policy": "Policy"}), hide_index=True)
