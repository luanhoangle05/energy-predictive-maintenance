"""Exercise training, tracking, and artifact replay without NASA data."""

import json
import sys

import mlflow
import numpy as np
import pandas as pd

from src.data.load_data import CMAPSS_COLUMNS
from src.models.bundle import load_bundle
from src.models.train_tracked import main


def test_tracked_training_exports_replayable_bundle(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    rows = []
    for engine in range(1, 61):
        for cycle in range(1, 36 + engine % 5):
            rows.append([engine, cycle, *rng.normal(cycle, 1, size=24)])
    frame = pd.DataFrame(rows, columns=CMAPSS_COLUMNS)
    source = tmp_path / "data/raw/cmapss/train_FD001.txt"
    source.parent.mkdir(parents=True)
    frame.to_csv(source, sep=" ", header=False, index=False)
    tracking_uri = "sqlite:///" + (tmp_path / "tracking.db").as_posix()
    previous_uri = mlflow.get_tracking_uri()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["train_tracked", "--tracking-uri", tracking_uri])
    try:
        main()
        bundle = load_bundle("models/development.joblib")
        manifest = json.loads((tmp_path / "models/development.json").read_text())
        splits = [set(ids) for ids in manifest["engine_ids"].values()]
        assert len(set.union(*splits)) == 60
        assert sum(map(len, splits)) == 60
        run = mlflow.get_run(manifest["mlflow_run_id"])
        assert run.info.status == "FINISHED"
        assert run.data.metrics == manifest["metrics"]
        assert run.data.tags["official_test_used"] == "false"
        artifacts = mlflow.MlflowClient().list_artifacts(run.info.run_id, "bundle")
        assert {a.path for a in artifacts} == {"bundle/development.joblib", "bundle/development.json"}
        history = frame.loc[(frame.unit_number == 1) & (frame.time_in_cycles <= 30)]
        assert len(bundle.predict(history)["predictions"]) == 1
    finally:
        mlflow.set_tracking_uri(previous_uri)
