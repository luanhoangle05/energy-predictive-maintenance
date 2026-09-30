"""Streamlit product view over the unchanged FD001 inference workflow."""
from io import BytesIO
import os
from pathlib import Path
import sys

# Script launchers may put only app/ on sys.path, excluding sibling packages.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import httpx
import pandas as pd
import streamlit as st
from app.ui.components import render_fleet, render_engine_detail
from app.ui.demo import load_demo
from app.ui.insights import render_insights, render_limitations
from app.ui.maintenance import load_maintenance_reports, render_maintenance
from app.ui.state import dataset_identity, sync_dataset, clear_predictions, request_predictions
from src.data.serving_upload import read_observations

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
st.header("Predictive Maintenance Dashboard")
st.markdown("**Remaining Useful Life Monitoring**")
st.caption("NASA C-MAPSS FD001 • Simulated turbofan degradation • Development model  \n"
           "Research/development demonstration. Not validated for real maintenance decisions.")
api = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
source = st.radio("Choose your fleet", ["Try demo fleet", "Upload FD001-compatible telemetry"],
                  horizontal=True, key="source_mode", label_visibility="collapsed")
if st.session_state.get("previous_source_mode") != source:
    sync_dataset(st.session_state, None)
    st.session_state.previous_source_mode = source

run_analysis = False
if source == "Try demo fleet":
    st.caption("Explore simulated development engines observed at different points before failure.")
    if st.button("Try demo fleet", type="primary", key="demo_button"):
        clear_predictions(st.session_state)
        try:
            frame = load_demo()
            identity = "demo:" + dataset_identity(frame.to_csv(index=False).encode(), "development-demo")
            sync_dataset(st.session_state, identity)
            st.session_state.uploaded_frame = frame
            st.session_state.pop("upload_error", None)
            run_analysis = True
        except (OSError, ValueError, TypeError) as exc:
            sync_dataset(st.session_state, None)
            st.error(f"Development demo unavailable: {exc}. You can still upload compatible telemetry.")
else:
    uploaded = st.file_uploader("Upload compatible multivariate engine histories", type=["csv", "txt"])
    identity = "upload:" + dataset_identity(uploaded.getvalue(), uploaded.name) if uploaded is not None else None
    sync_dataset(st.session_state, identity)
    if uploaded is not None:
        if uploaded.name.lower() == "test_fd001.txt":
            st.warning("Reserved official test file: generating predictions begins final benchmark evaluation.")
        if "uploaded_frame" not in st.session_state and "upload_error" not in st.session_state:
            try:
                st.session_state.uploaded_frame = read_observations(BytesIO(uploaded.getvalue()), uploaded.name)
            except (ValueError, TypeError) as exc:
                st.session_state.upload_error = str(exc)
        if "upload_error" in st.session_state:
            st.error(f"Validation failed: {st.session_state.upload_error}")
        else:
            run_analysis = st.button("Analyze telemetry", type="primary", key="analyze_button")

with st.expander("Data requirements"):
    st.write("This model supports FD001-compatible simulated turbofan histories, not arbitrary machinery. "
             "Use a CSV with the 26 named raw columns or a NASA-style whitespace-delimited .txt file without a header. "
             "Supply each engine’s complete observations from cycle 1 through its current observed cycle, "
             "in consecutive order. IDs and cycles must be positive integers; all measurements must be finite. "
             "No targets or derived features. Maximum: 50,000 uploaded rows and 10,000 rows per engine.")
    st.code("unit_number, time_in_cycles, operational_setting_1–3, sensor_1–21", language=None)
    st.write("Full train_FD001.txt histories end at failure and therefore describe end-of-life engines. "
             "Use the demo to explore partial histories. RUL_FD001.txt contains labels, not telemetry.")
    st.warning("Using reserved test_FD001.txt begins final benchmark evaluation.")

if "uploaded_frame" in st.session_state:
    frame = st.session_state.uploaded_frame
    label = "Development demo fleet · artificially truncated histories" if source == "Try demo fleet" else "Uploaded compatible telemetry"
    st.caption(f"{label} · Validated · {frame.unit_number.nunique():,} engines · {len(frame):,} observations")
    with st.expander("Preview observations"):
        st.dataframe(frame.head(20), hide_index=True, width="stretch")
        st.caption("First 20 rows only. The model receives every observed row through each engine’s cutoff.")

if run_analysis:
    clear_predictions(st.session_state)
    try:
        with st.spinner("Estimating remaining life from observed histories…"):
            with httpx.Client(timeout=60) as client:
                result = request_predictions(client, api, st.session_state.uploaded_frame)
                st.session_state.prediction_result = result
                for route, state_key in (("/metadata", "model_metadata"), ("/insights", "model_insights")):
                    try:
                        response = client.get(api + route)
                        response.raise_for_status()
                        details = response.json()
                        if details.get("model_id") != result["model_id"]:
                            raise ValueError("Model changed; details are not shown for this analysis.")
                        st.session_state[state_key] = details
                    except (httpx.HTTPError, ValueError) as exc:
                        st.session_state.metadata_notice = f"Some model details are unavailable: {exc}"
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        st.error(f"Unable to generate predictions: {exc}")

try:
    maintenance = load_maintenance_reports()
except (OSError, ValueError, KeyError):
    maintenance = None

names = ["Fleet Overview", "Engine Explorer"]
if maintenance is not None:
    names.append("Maintenance Analysis")
names.append("Model Insights")
tabs = dict(zip(names, st.tabs(names)))
result = st.session_state.get("prediction_result")
results = pd.DataFrame(result["predictions"]) if result else None
with tabs["Fleet Overview"]:
    if results is not None:
        render_fleet(results)
        st.download_button("Download all predictions", results.to_csv(index=False),
                           "predictions.csv", "text/csv")
    else:
        st.info("Try the development demo or upload compatible telemetry to analyze a fleet.")
with tabs["Engine Explorer"]:
    if results is not None:
        render_engine_detail(results, st.session_state.uploaded_frame)
    else:
        st.info("Analyze a fleet first, then inspect an engine’s current estimate and observed sensors.")
if maintenance is not None:
    with tabs["Maintenance Analysis"]:
        render_maintenance(maintenance)
with tabs["Model Insights"]:
    render_insights(st.session_state.get("model_metadata"), st.session_state.get("model_insights"),
                    st.session_state.get("metadata_notice"))
    render_limitations()
