"""Synthetic integration checks: persisted features, API contract, and failures."""

import numpy as np
import pandas as pd
import pytest
import sklearn
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.data.load_data import CMAPSS_COLUMNS
from src.features.build_features import build_features
from src.features.preprocess_features import create_feature_preprocessor
from src.models.bundle import ModelBundle, load_bundle
from src.models.train_model import train_gradient_boosting_regressor


@pytest.fixture
def history():
    rows = []
    for engine in (1, 2):
        for cycle in range(1, 8):
            rows.append([engine, cycle] + [float(engine * cycle + j) for j in range(24)])
    return pd.DataFrame(rows, columns=CMAPSS_COLUMNS)


@pytest.fixture
def artifact(tmp_path, history):
    prep = create_feature_preprocessor()
    x = prep.fit_transform(build_features(history))
    model = train_gradient_boosting_regressor(x, 10 - history.time_in_cycles, n_estimators=5)
    bundle = ModelBundle(prep, model, 4.0,
                         {"model_id": "synthetic", "sklearn_version": sklearn.__version__})
    path = tmp_path / "model.joblib"
    bundle.save(path)
    return path


def test_round_trip_and_independent_engines(artifact, history):
    bundle = load_bundle(artifact)
    combined = bundle.predict(history)["predictions"]
    separate = [bundle.predict(group)["predictions"][0]
                for _, group in history.groupby("unit_number")]
    assert combined == separate
    assert [row["time_in_cycles"] for row in combined] == [7, 7]
    expected = bundle.model.predict(bundle.preprocessor.transform(build_features(history)))
    np.testing.assert_allclose([row["predicted_rul"] for row in combined], expected[[6, 13]])


def test_api_matches_bundle(artifact, history):
    with TestClient(create_app(artifact)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").json()["model_id"] == "synthetic"
        assert client.get("/metadata").json()["sklearn_version"] == sklearn.__version__
        result = client.post("/predict", json={"observations": history.to_dict("records")})
        assert result.status_code == 200
        assert result.json() == load_bundle(artifact).predict(history)


@pytest.mark.parametrize("problem", ["missing", "extra", "duplicate", "gap", "order",
                                     "fractional_id", "negative_id", "text", "boolean", "null"])
def test_invalid_history_rejected(artifact, history, problem):
    if problem == "missing":
        history = history.drop(columns="sensor_1")
    elif problem == "extra":
        history["remaining_useful_life"] = 10
    elif problem == "duplicate":
        history = pd.concat([history, history.iloc[[0]]])
    elif problem == "gap":
        history = history.drop(index=2)
    elif problem == "order":
        history = history.iloc[::-1]
    elif problem == "fractional_id":
        history["unit_number"] = 1.5
    elif problem == "negative_id":
        history["unit_number"] = -1
    else:
        history["sensor_1"] = {"text": "12", "boolean": True, "null": None}[problem]
    with TestClient(create_app(artifact)) as client:
        result = client.post("/predict", json={"observations": history.to_dict("records")})
        assert result.status_code == 422


def test_empty_and_oversized_requests(artifact, history):
    with TestClient(create_app(artifact)) as client:
        assert client.post("/predict", json={"observations": []}).status_code == 422
        rows = [history.iloc[0].to_dict()] * 10001
        assert client.post("/predict", json={"observations": rows}).status_code == 422


@pytest.mark.parametrize("corrupt", [False, True])
def test_missing_or_corrupt_artifact_is_unready(tmp_path, corrupt):
    path = tmp_path / "missing.joblib"
    if corrupt:
        path.write_text("not an artifact")
    with TestClient(create_app(path)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.post("/predict", json={"observations": [{}]}).status_code == 503


def test_version_mismatch_rejected(artifact):
    bundle = load_bundle(artifact)
    bundle.metadata["sklearn_version"] = "incompatible"
    bundle.save(artifact)
    with pytest.raises(ValueError, match="version"):
        load_bundle(artifact)


def test_nonfinite_history_rejected(artifact, history):
    history.loc[0, "sensor_1"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        load_bundle(artifact).predict(history)


def test_insights_expose_existing_fitted_importance(artifact):
    bundle = load_bundle(artifact)
    with TestClient(create_app(artifact)) as client:
        response = client.get("/insights")
        assert response.status_code == 200
        details = response.json()
        assert details["model_id"] == bundle.metadata["model_id"]
        assert details["retained_feature_count"] == len(bundle.model.feature_importances_)
        assert [r["feature"] for r in details["feature_importances"]] == list(bundle.preprocessor.get_feature_names_out())
        np.testing.assert_array_equal([r["importance"] for r in details["feature_importances"]],
                                      bundle.model.feature_importances_)
        assert client.get("/metadata").json() == bundle.metadata
