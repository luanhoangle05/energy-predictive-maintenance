"""Functional dashboard state, ranking, API parity, and error handling."""

from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.ui.components import fleet_figure
from app.ui.state import dataset_identity, sync_dataset, priority_engines, fleet_metrics, request_predictions
from src.data.load_data import CMAPSS_COLUMNS


def histories():
    return pd.DataFrame([[engine, cycle] + [float(cycle)] * 24
                         for engine in (1, 2) for cycle in (1, 2, 3)], columns=CMAPSS_COLUMNS)


def prediction(engine, rul=20.123456789, lower=0.0):
    return dict(unit_number=engine, time_in_cycles=3, predicted_rul=rul,
                lower_bound=lower, upper_bound=rul + 40, low_rul_flag=lower <= 30,
                threshold_status="crosses_threshold" if lower <= 30 else "entirely_above")


def test_dataset_identity_and_invalidation():
    identity = dataset_identity(b"first", "same.csv")
    state = dict(dataset_identity=identity, prediction_result={"predictions": [1]},
                 selected_engine=2, model_metadata={}, uploaded_frame="data")
    original = state.copy()
    sync_dataset(state, identity)
    assert state == original
    sync_dataset(state, dataset_identity(b"changed", "same.csv"))
    assert set(state) == {"dataset_identity"}
    state["prediction_result"] = "new"
    sync_dataset(state, None)
    assert state == {"dataset_identity": None}
    assert dataset_identity(b"same", "x.csv") != dataset_identity(b"same", "x.txt")


def test_rank_metrics_and_interval_chart_preserve_values():
    results = pd.DataFrame([prediction(i, i + .123456789, max(0, i - 10)) for i in range(30, 0, -1)])
    original = results.copy(deep=True)
    top = priority_engines(results)
    assert top.unit_number.tolist() == list(range(1, 21))
    assert fleet_metrics(results)["Engines analyzed"] == 30
    assert fleet_metrics(results)["Median estimated life"] == results.predicted_rul.median()
    fig = fleet_figure(results)
    trace = fig.data[0]
    np.testing.assert_array_equal(trace.x, top.predicted_rul)
    np.testing.assert_allclose(np.array(trace.x) - trace.error_x.arrayminus, top.lower_bound)
    np.testing.assert_allclose(np.array(trace.x) + trace.error_x.array, top.upper_bound)
    assert fig.layout.shapes[0].x0 == 30
    pd.testing.assert_frame_equal(results, original)


def test_api_payload_is_not_rounded():
    payload = {"model_id": "test", "predictions": [prediction(1), prediction(2)]}
    def respond(request):
        assert request.url.path == "/predict"
        return httpx.Response(200, json=payload)
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        assert request_predictions(client, "http://test", histories()) == payload


def test_mixed_models_are_rejected():
    from app.ui import state
    with patch.object(state, "history_batches", return_value=[histories(), histories()]):
        count = 0
        def respond(request):
            nonlocal count
            count += 1
            return httpx.Response(200, json={"model_id": str(count), "predictions": [prediction(count)]})
        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            with pytest.raises(ValueError, match="Model changed"):
                request_predictions(client, "http://test", histories())


@pytest.mark.parametrize("metadata_status", [200, 503])
def test_upload_prediction_selector_and_replacement(metadata_status):
    upload = BytesIO(histories().to_csv(index=False).encode())
    upload.name = "same.csv"
    payload = {"model_id": "test", "predictions": [prediction(1), prediction(2, 70, 35)]}
    calls = []
    def respond(request):
        calls.append(request.url.path)
        if request.url.path == "/predict":
            return httpx.Response(200, json=payload)
        if request.url.path == "/insights":
            return httpx.Response(200, json={"model_id": "test", "retained_feature_count": 1,
                                            "feature_importances": [{"feature": "sensor_4", "importance": 1.0}]})
        return httpx.Response(metadata_status, json={"model_id": "test", "confidence": .9,
                                                    "metrics": {"interval_radius": 40}, "parameters": {}})
    client_type = httpx.Client
    path = Path(__file__).resolve().parents[1] / "app/dashboard.py"
    with patch("streamlit.file_uploader", return_value=upload) as uploader, patch(
        "httpx.Client", side_effect=lambda **kwargs: client_type(transport=httpx.MockTransport(respond), **kwargs)
    ):
        app = AppTest.from_file(str(path)).run()
        assert not app.exception
        app.radio(key="source_mode").set_value("Upload FD001-compatible telemetry").run()
        app.button(key="analyze_button").click().run()
        assert not app.exception
        assert app.session_state["prediction_result"] == payload
        app.selectbox(key="selected_engine").select(2).run()
        assert not app.exception
        assert app.session_state["selected_engine"] == 2
        assert app.session_state["prediction_result"] == payload
        assert calls.count("/predict") == 1
        assert any(metric.value == "70.00" for metric in app.metric)
        app.selectbox(key="selected_sensor").select("sensor_3").run()
        assert app.session_state["prediction_result"] == payload
        # Changed contents with identical filename must invalidate results even when invalid.
        replacement = BytesIO(b"invalid\nfile\n")
        replacement.name = "same.csv"
        uploader.return_value = replacement
        app.run()
        assert not app.exception
        assert app.error
        assert "prediction_result" not in app.session_state
        assert len(app.selectbox) == 0
        uploader.return_value = None
        app.run()
        assert not app.error
        assert "uploaded_frame" not in app.session_state
