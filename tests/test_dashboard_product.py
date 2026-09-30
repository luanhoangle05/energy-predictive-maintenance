"""Demo isolation, display semantics, and source-switch interaction tests."""

from pathlib import Path
from unittest.mock import patch
import json
import hashlib
from io import BytesIO

import httpx
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.ui.demo import truncate_demo
from app.ui import demo
from app.ui.formatting import attention_label, range_label, priority_table
from app.ui.maintenance import load_maintenance_reports
from app.ui.state import request_predictions
from src.data.load_data import CMAPSS_COLUMNS
from src.data.prepare_data import select_cutoff_indices


def full_development_histories():
    return pd.DataFrame([[engine, cycle] + [float(cycle + engine)] * 24
                         for engine in range(1, 21) for cycle in range(1, 101 + engine)],
                        columns=CMAPSS_COLUMNS)


def test_demo_is_exact_prefix_and_excludes_targets_and_future():
    raw = full_development_histories()
    labeled = raw.assign(remaining_useful_life=999)
    observed = truncate_demo(labeled)
    cutoffs = raw.loc[select_cutoff_indices(raw, random_state=43)].set_index("unit_number").time_in_cycles
    assert list(observed.columns) == CMAPSS_COLUMNS
    for engine, group in observed.groupby("unit_number"):
        expected = raw.loc[(raw.unit_number == engine) & (raw.time_in_cycles <= cutoffs[engine])]
        pd.testing.assert_frame_equal(group.reset_index(drop=True), expected.reset_index(drop=True))
        assert 30 <= group.time_in_cycles.max() < raw.loc[raw.unit_number == engine].time_in_cycles.max()
    changed = raw.copy()
    future = changed.time_in_cycles > changed.unit_number.map(cutoffs)
    changed.loc[future, "sensor_4"] = -99999
    pd.testing.assert_frame_equal(truncate_demo(changed), observed)
    pd.testing.assert_frame_equal(truncate_demo(raw), observed)


def test_demo_request_contains_only_truncated_raw_histories():
    observed = truncate_demo(full_development_histories())
    sent = []
    def respond(request):
        sent.extend(json.loads(request.content)["observations"])
        return httpx.Response(200, json={"model_id": "test", "predictions": []})
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        request_predictions(client, "http://test", observed)
    pd.testing.assert_frame_equal(pd.DataFrame(sent), observed)


@pytest.mark.parametrize("lower,expected", [(0, "Attention"), (30, "Attention"), (30.0001, "No attention flag")])
def test_plain_language_status_preserves_exact_criterion(lower, expected):
    assert attention_label(lower <= 30) == expected
    data = pd.DataFrame([dict(unit_number=18, time_in_cycles=142, predicted_rul=43.123456,
                              lower_bound=lower, upper_bound=118.98765, low_rul_flag=lower <= 30)])
    original = data.copy(deep=True)
    table = priority_table(data)
    assert table.Status.iloc[0] == expected
    assert table["Estimated life"].iloc[0] == "43.1 cycles"
    assert table["Exploratory range"].iloc[0] == range_label(lower, 118.98765)
    pd.testing.assert_frame_equal(data, original)


def test_saved_maintenance_values_are_read_without_recalculation(tmp_path):
    policies = ["fixed_age", "run_to_failure", "predicted_rul", "lower_bound"]
    costs = pd.DataFrame(dict(policy=policies, engines=[20]*4, mean_cost_per_engine=[2.3,10,1.2,1.9]))
    timing = pd.DataFrame(dict(policy=policies, engines=[20]*4, failures=[2,20,0,0]))
    costs.to_csv(tmp_path / "cost_summary.csv", index=False)
    timing.to_csv(tmp_path / "timing_summary.csv", index=False)
    result = load_maintenance_reports(tmp_path)
    for row in result.itertuples():
        assert row.mean_cost_per_engine == costs.set_index("policy").loc[row.policy, "mean_cost_per_engine"]
        assert row.failures == timing.set_index("policy").loc[row.policy, "failures"]


def test_demo_button_and_mode_switch_clear_results():
    observed = truncate_demo(full_development_histories())
    calls = []
    def respond(request):
        if request.url.path == "/predict":
            calls.append(request)
            rows = [dict(unit_number=int(engine), time_in_cycles=int(group.time_in_cycles.max()),
                         predicted_rul=60.0, lower_bound=0.0, upper_bound=135.0,
                         low_rul_flag=True, threshold_status="crosses_threshold")
                    for engine, group in observed.groupby("unit_number")]
            return httpx.Response(200, json={"model_id":"test", "predictions":rows})
        return httpx.Response(503)
    client_type = httpx.Client
    with patch("app.ui.demo.load_demo", return_value=observed), patch("httpx.Client", side_effect=lambda **kw:
        client_type(transport=httpx.MockTransport(respond), **kw)):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app/dashboard.py")).run()
        app.button(key="demo_button").click().run()
        assert not app.exception
        assert len(app.session_state["prediction_result"]["predictions"]) == 20
        app.selectbox(key="selected_engine").select(2).run()
        assert not app.exception
        assert len(calls) == 1
        app.radio(key="source_mode").set_value("Upload FD001-compatible telemetry").run()
        assert "prediction_result" not in app.session_state
        assert "uploaded_frame" not in app.session_state
        app.radio(key="source_mode").set_value("Try demo fleet").run()
        assert "prediction_result" not in app.session_state


def test_demo_missing_data_does_not_fall_back_to_test_data():
    with patch("app.ui.demo.load_demo", side_effect=FileNotFoundError("training data missing")):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app/dashboard.py")).run()
        app.button(key="demo_button").click().run()
        assert not app.exception
        assert "Development demo unavailable" in app.error[0].value
        assert "prediction_result" not in app.session_state


def test_cloud_demo_matches_local_prefix_and_caches_verified_bytes(tmp_path, monkeypatch):
    raw = full_development_histories()
    content = raw.to_csv(sep=" ", header=False, index=False).encode()
    source = tmp_path / "train_FD001.txt"
    source.write_bytes(content)
    monkeypatch.setattr(demo, "DEMO_SOURCE", source)
    local = demo.load_demo()
    monkeypatch.setattr(demo, "DEMO_SOURCE", tmp_path / "missing.txt")
    monkeypatch.setenv("DEMO_SOURCE_URL", "https://example.com/train_FD001.txt")
    monkeypatch.setenv("DEMO_SOURCE_SHA256", hashlib.sha256(content).hexdigest())
    demo.download_demo_source.cache_clear()
    try:
        with patch("app.ui.demo.urllib.request.urlopen", return_value=BytesIO(content)) as download:
            pd.testing.assert_frame_equal(demo.load_demo(), local)
            pd.testing.assert_frame_equal(demo.load_demo(), local)
            download.assert_called_once()
    finally:
        demo.download_demo_source.cache_clear()


def test_cloud_demo_rejects_mismatched_bytes():
    demo.download_demo_source.cache_clear()
    with patch("app.ui.demo.urllib.request.urlopen", return_value=BytesIO(b"wrong")):
        with pytest.raises(ValueError, match="checksum mismatch"):
            demo.download_demo_source("https://example.com/train_FD001.txt", "0" * 64)
    assert demo.download_demo_source.cache_info().currsize == 0


@pytest.mark.parametrize("filename", ["test_FD001.txt", "RUL_FD001.txt"])
def test_cloud_demo_rejects_official_test_urls_before_download(filename):
    with patch("app.ui.demo.urllib.request.urlopen") as download:
        with pytest.raises(ValueError, match="train_FD001"):
            demo.download_demo_source("https://example.com/" + filename, "0" * 64)
        download.assert_not_called()
