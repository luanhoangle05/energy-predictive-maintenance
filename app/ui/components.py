"""Plain-language fleet and engine presentation; numerical results stay untouched."""
import plotly.graph_objects as go
import streamlit as st
from app.ui.state import fleet_metrics, priority_engines
from app.ui.formatting import LIFE_HELP, RANGE_HELP, ATTENTION_HELP, attention_label, range_label, priority_table

ACCENT = "#2563EB"
WARNING = "#B45309"


def render_metrics(results):
    for column, (label, value) in zip(st.columns(4), fleet_metrics(results).items()):
        column.metric(label, f"{value:,.1f}" if isinstance(value, float) else str(value),
                      help=ATTENTION_HELP if label == "Attention flags" else LIFE_HELP if "life" in label else None,
                      border=True)


def fleet_figure(results):
    priority = priority_engines(results)
    labels = [f"Engine {int(value):03d}" for value in priority.unit_number]
    custom = [[float(r.lower_bound), float(r.upper_bound), int(r.time_in_cycles),
               attention_label(r.low_rul_flag)] for r in priority.itertuples()]
    figure = go.Figure(go.Scatter(
        x=priority.predicted_rul.tolist(), y=labels, mode="markers",
        marker=dict(size=11, color=ACCENT, line=dict(color="white", width=1)),
        error_x=dict(type="data", symmetric=False,
                     array=(priority.upper_bound - priority.predicted_rul).tolist(),
                     arrayminus=(priority.predicted_rul - priority.lower_bound).tolist(),
                     color="#A8B6CA", thickness=1.5, width=4),
        customdata=custom,
        hovertemplate=("%{y}<br>Estimated remaining life: %{x:.2f} cycles"
                       "<br>Exploratory range: %{customdata[0]:.2f}–%{customdata[1]:.2f} cycles"
                       "<br>Current observed cycle: %{customdata[2]}"
                       "<br>Status: %{customdata[3]}<extra></extra>"),
    ))
    figure.add_vline(x=30, line_dash="dash", line_color=WARNING,
                     annotation_text="30-cycle attention reference", annotation_position="top right")
    figure.update_layout(
        height=max(290, len(priority) * 23 + 85), margin=dict(l=0, r=12, t=30, b=0),
        paper_bgcolor="white", plot_bgcolor="white", font=dict(color="#334155", size=12),
        xaxis=dict(title="Estimated remaining life (cycles)", gridcolor="#E2E8F0",
                   range=[-3, max(65, float(priority.upper_bound.max()) * 1.05)]),
        yaxis=dict(categoryorder="array", categoryarray=labels, autorange="reversed", automargin=True),
        showlegend=False,
    )
    return figure


def render_fleet(results):
    render_metrics(results)
    st.caption("Estimated life is measured in operating cycles. Attention flag: " + ATTENTION_HELP)
    st.markdown("#### Engines with the lowest estimated life")
    st.caption(f"Showing {min(20, len(results))} of {len(results)} engines, lowest estimate first.")
    st.plotly_chart(fleet_figure(results), width="stretch", config={"displayModeBar": False})
    st.caption("Blue dot: model estimate · gray line: exploratory uncertainty range. " + RANGE_HELP)
    st.markdown("#### Priority engines")
    st.dataframe(priority_table(priority_engines(results)), hide_index=True, width="stretch", height=230)
    st.caption("Attention is a single documented flag, not a failure probability or an instruction to service an engine.")


def sensor_figure(history, sensor):
    figure = go.Figure(go.Scatter(
        x=history.time_in_cycles.tolist(), y=history[sensor].tolist(), mode="lines",
        line=dict(color=ACCENT, width=2),
        hovertemplate="Observed cycle: %{x}<br>Sensor value: %{y:.3f}<extra></extra>",
    ))
    figure.update_layout(height=280, margin=dict(l=0, r=10, t=10, b=0),
                          xaxis_title="Observed cycle", yaxis_title="Observed sensor value",
                          hovermode="x", paper_bgcolor="white", plot_bgcolor="white")
    figure.update_xaxes(gridcolor="#E2E8F0")
    figure.update_yaxes(gridcolor="#E2E8F0")
    return figure


def render_engine_detail(results, history):
    ids = sorted(results.unit_number.astype(int).tolist())
    engine = st.selectbox("Engine", ids, format_func=lambda value: f"Engine {value:03d}", key="selected_engine")
    row = results.loc[results.unit_number == engine].iloc[0]
    columns = st.columns(4)
    columns[0].metric("Current observed cycle", str(int(row.time_in_cycles)), border=True)
    columns[1].metric("Estimated remaining life", f"{row.predicted_rul:.2f}", help=LIFE_HELP, border=True)
    columns[2].metric("Exploratory range", f"{row.lower_bound:.1f}–{row.upper_bound:.1f}", help=RANGE_HELP, border=True)
    columns[3].metric("Attention status", attention_label(row.low_rul_flag), help=ATTENTION_HELP, border=True)
    st.caption("Life estimate and range are in cycles, at the latest observation only. Attention flag: " + ATTENTION_HELP)
    st.markdown("#### Observed sensor history")
    sensor = st.selectbox("Sensor", [f"sensor_{i}" for i in range(1, 22)], index=3,
                          key="selected_sensor", format_func=lambda value: value.replace("_", " ").title())
    engine_history = history.loc[history.unit_number == engine]
    st.plotly_chart(sensor_figure(engine_history, sensor), width="stretch", config={"displayModeBar": False})
    st.caption("Observed measurements, not remaining-life predictions. The 30-cycle reference applies "
               "to remaining life, not sensor values. No historical prediction curve is inferred.")
    with st.expander("Endpoint technical details"):
        st.write({"predicted_rul": float(row.predicted_rul), "lower_bound": float(row.lower_bound),
                  "upper_bound": float(row.upper_bound), "low_rul_flag": bool(row.low_rul_flag)})
