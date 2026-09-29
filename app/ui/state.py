"""Dataset-scoped presentation state and existing API orchestration."""

import hashlib
from src.data.serving_upload import history_batches

RESULT_KEYS = ("prediction_result", "model_metadata", "metadata_notice", "model_insights", "selected_engine", "selected_sensor")


def dataset_identity(content, filename):
    # Name selects the parser and can carry the reserved benchmark warning.
    return hashlib.sha256(filename.encode() + b"\0" + content).hexdigest()


def sync_dataset(state, identity):
    """Invalidate dependent state on replacement/removal, including invalid files."""
    if state.get("dataset_identity") != identity:
        for key in (*RESULT_KEYS, "uploaded_frame", "upload_error"):
            state.pop(key, None)
        state["dataset_identity"] = identity


def clear_predictions(state):
    for key in RESULT_KEYS:
        state.pop(key, None)


def request_predictions(client, api, frame):
    """Preserve existing batching, values, and mixed-model protection."""
    predictions, model_id = [], None
    for batch in history_batches(frame):
        response = client.post(api + "/predict", json={"observations": batch.to_dict("records")})
        if response.status_code != 200:
            raise ValueError(f"Prediction unavailable: {response.json().get('detail')}")
        payload = response.json()
        if model_id is not None and model_id != payload["model_id"]:
            raise ValueError("Model changed during prediction; please retry.")
        model_id = payload["model_id"]
        predictions.extend(payload["predictions"])
    return {"predictions": predictions, "model_id": model_id}


def priority_engines(results, limit=20):
    return results.sort_values(["predicted_rul", "unit_number"]).head(limit).copy()


def fleet_metrics(results):
    return {"Engines analyzed": len(results),
            "Attention flags": int(results.low_rul_flag.sum()),
            "Median estimated life": float(results.predicted_rul.median()),
            "Lowest estimated life": float(results.predicted_rul.min())}
