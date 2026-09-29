"""Reproduce the development split, track evaluation, and export a serving bundle."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import uuid

import numpy as np
import sklearn

from src.data.load_data import load_cmapss_file, RUL_COLUMN
from src.data.prepare_data import prepare_uncertainty_splits, select_cutoff_indices
from src.evaluation.evaluate_model import regression_metrics
from src.evaluation.uncertainty import calibrate_interval_radius, build_prediction_intervals
from src.features.build_features import build_features
from src.features.preprocess_features import create_feature_preprocessor
from src.models.train_model import train_gradient_boosting_regressor
from src.models.bundle import ModelBundle

PARAMETERS = dict(n_estimators=200, learning_rate=0.05, max_depth=2,
                  min_samples_leaf=30, random_state=42)


def fit_development_bundle(raw):
    """Fit only model-training engines; reserve calibration and evaluation engines."""
    training, calibration, evaluation = prepare_uncertainty_splits(raw)
    preprocessor = create_feature_preprocessor()
    train_x = preprocessor.fit_transform(build_features(training))
    model = train_gradient_boosting_regressor(train_x, training[RUL_COLUMN], **PARAMETERS)
    predictions, actuals = [], []
    for frame, seed in ((calibration, 42), (evaluation, 43)):
        indices = select_cutoff_indices(frame, random_state=seed)
        features = preprocessor.transform(build_features(frame).loc[indices])
        predictions.append(np.maximum(model.predict(features), 0.0))
        actuals.append(frame.loc[indices, RUL_COLUMN].to_numpy())
    radius = calibrate_interval_radius(actuals[0], predictions[0], confidence=0.90)
    lower, upper = build_prediction_intervals(predictions[1], radius)
    metrics = regression_metrics(actuals[1], predictions[1])
    metrics.update(coverage=float(np.mean((actuals[1] >= lower) & (actuals[1] <= upper))),
                   mean_interval_width=float(np.mean(upper - lower)), interval_radius=radius)
    metadata = dict(model_id=str(uuid.uuid4()), sklearn_version=sklearn.__version__,
                    python_version=platform.python_version(), dataset="FD001 development",
                    confidence=0.90, threshold=30.0, parameters=PARAMETERS,
                    rolling_window=5, target="uncapped RUL", metrics=metrics,
                    split_seed=42, calibration_cutoff_seed=42, evaluation_cutoff_seed=43,
                    engine_ids={name: sorted(frame.unit_number.unique().tolist())
                                for name, frame in (("training", training),
                                                    ("calibration", calibration),
                                                    ("evaluation", evaluation))})
    return ModelBundle(preprocessor, model, radius, metadata), metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="models/development.joblib")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    args = parser.parse_args()
    # Deliberately no test-set input or arbitrary data path in this development runner.
    source = Path("data/raw/cmapss/train_FD001.txt")
    import mlflow
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("fd001-development")
    with mlflow.start_run() as run:
        bundle, metrics = fit_development_bundle(load_cmapss_file(source))
        bundle.metadata.update(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                               mlflow_run_id=run.info.run_id)
        mlflow.log_params({**PARAMETERS, "rolling_window": 5, "confidence": 0.90,
                           "split_seed": 42, "target": "uncapped", "dataset": "FD001"})
        mlflow.set_tags({"scope": "exploratory development", "official_test_used": "false"})
        mlflow.log_metrics(metrics)
        output = Path(args.output)
        bundle.save(output)
        manifest = output.with_suffix(".json")
        manifest.write_text(json.dumps(bundle.metadata, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(output), "bundle")
        mlflow.log_artifact(str(manifest), "bundle")
        print(json.dumps({"artifact": str(output), "run_id": run.info.run_id,
                          "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
