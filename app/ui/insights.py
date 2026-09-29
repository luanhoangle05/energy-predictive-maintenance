"""Explain existing fitted importance and recorded development evaluation."""

import math
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_insights(metadata, insights, notice=None):
    st.markdown("#### What the model relies on")
    if insights and metadata and insights.get("model_id") == metadata.get("model_id"):
        features = pd.DataFrame(insights["feature_importances"]).nlargest(12, "importance")
        figure = go.Figure(go.Bar(
            x=features.importance.tolist(), y=features.feature.tolist(), orientation="h", marker_color="#2563EB",
            hovertemplate="%{y}<br>Share of fitted split importance: %{x:.1%}<extra></extra>",
        ))
        figure.update_layout(height=360, margin=dict(l=0, r=10, t=10, b=0),
                              xaxis=dict(title="Share of split-based importance", tickformat=".0%"),
                              yaxis=dict(autorange="reversed", automargin=True))
        st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    else:
        st.caption("Run an analysis to retrieve importance from the currently loaded model. "
                   "If unavailable, inference can still be used.")
    st.caption("Importance reflects split/impurity usage by this fitted Gradient Boosting model. "
               "It does not show causality or the direction of a feature’s effect.")
    st.markdown("#### Model performance")
    metrics = metadata.get("metrics", {}) if metadata else {}
    if all(k in metrics for k in ("mae", "rmse", "r2", "nasa_score")):
        st.caption("Serving model · one artificial snapshot per development-evaluation engine · exploratory results")
        columns = st.columns(4)
        for col, label, value, help_text in zip(columns,
            ["Average absolute error", "Large-error measure", "Variation explained", "Asymmetric error score"],
            [f"{metrics['mae']:.2f}", f"{metrics['rmse']:.2f}", f"{metrics['r2']:.3f}", f"{metrics['nasa_score']:.2f}"],
            ["MAE: mean absolute error in cycles; lower is better.",
             "RMSE: error in cycles that weighs larger mistakes more heavily; lower is better.",
             "R²: fraction of target variation explained relative to predicting its mean; higher is better.",
             "NASA asymmetric score: overestimating remaining life is penalized more. Lower is better; totals depend on sample count."]):
            col.metric(label, value, help=help_text, border=True)
    else:
        st.caption("Recorded metrics will appear after a successful analysis and metadata request.")
    with st.expander("Earlier all-cycle development evaluation"):
        st.write("Milestone 6 evaluated 4,070 observations from 20 development-validation engines "
                 "using the earlier 80-engine fit. These are not endpoint metrics for the serving model.")
        st.table(pd.DataFrame({"Measure": ["Average absolute error (MAE)", "Root mean square error (RMSE)",
                                            "Variation explained (R²)", "NASA score (sum)", "Near-failure MAE"],
                               "Recorded result": ["22.04 cycles", "29.03 cycles", "0.8045", "137,156.12", "6.82 cycles"]}))
        st.caption("Source: Milestone 6 evaluation report. Near-failure MAE covers 620 observations "
                   "with actual remaining life ≤ 30 cycles. Repeated development inspection makes "
                   "these results exploratory. NASA totals across different sample counts are not directly comparable.")
    st.markdown("#### Understanding the exploratory range")
    if metadata and "interval_radius" in metrics:
        count = len(metadata.get("engine_ids", {}).get("calibration", []))
        confidence = metadata.get("confidence", .9)
        radius = metrics["interval_radius"]
        if count:
            rank = math.ceil((count + 1) * confidence)
            st.info(f"{count} calibration engines → {confidence:.0%} target → error rank {rank} of {count} → {radius:.2f}-cycle radius")
            st.write("With this small calibration sample, the finite-sample rule selects the largest "
                     "observed absolute calibration error. The resulting radius is broad. "
                     "The lower end is clipped at zero, so displayed range widths can differ.")
        if "coverage" in metrics:
            st.caption(f"Recorded snapshot coverage: {metrics['coverage']:.0%}. "
                       "Coverage is exploratory; development data was reused during earlier model development.")
    st.write("An uncertainty range communicates development-calibrated error around an estimate. "
             "It is not a failure probability or a guarantee for real equipment or repeated lifecycle queries.")
    with st.expander("Model details"):
        st.write("Gradient Boosting · uncapped remaining useful life · five-cycle rolling features")
        if metadata:
            if insights:
                st.write(f"Retained features: {insights['retained_feature_count']}")
            st.json(metadata, expanded=False)
        if notice:
            st.caption(notice)


def render_limitations():
    with st.expander("Dataset information"):
        st.write("NASA C-MAPSS FD001 contains simulated turbofan degradation trajectories with one "
                 "operating condition and one fault mode. These are not current aircraft telemetry "
                 "and this is not a NASA operational maintenance system.")
        st.write("This trained model has only been evaluated in the NASA C-MAPSS FD001 development "
                 "setting. Applying the workflow to other machinery requires equipment-specific data, "
                 "retraining, validation, and uncertainty calibration. It is not a universal machine predictor.")
    with st.expander("Methodology and limitations"):
        st.write("Observed history so far → validation and causal temporal features → fitted preprocessing "
                 "and Gradient Boosting → current remaining-life estimate and exploratory range. "
                 "Inference runs through FastAPI; the dashboard does not fit models.")
        st.write("The serving development fit uses 64 engines, 16 calibration engines, and 20 evaluation "
                 "engines. Calibration uses one pre-failure snapshot per engine. Historical development "
                 "reuse limits generalization claims; the official NASA test benchmark remains reserved.")
        st.write("The demo reuses Milestone 7 pre-failure cutoffs with a fixed seed, starting at cycle 30. "
                 "Full development lifetimes are used only to construct eligible retrospective cutoffs. "
                 "Truncation happens before inference; no future rows or true remaining-life targets "
                 "enter the model. Demo results are not an independent accuracy benchmark.")
        st.write("Attention means the lower range endpoint is ≤ 30 cycles. It is not a diagnosis, failure "
                 "probability, or recommendation. Sensor history is observed input, not historical RUL predictions.")
