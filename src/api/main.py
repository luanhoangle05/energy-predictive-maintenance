"""FastAPI service with one loaded artifact per process."""

from contextlib import asynccontextmanager
import logging
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
import pandas as pd

from src.models.bundle import load_bundle


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observations: list[dict[str, object]] = Field(min_length=1, max_length=10000)


def create_app(model_path=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.bundle = None
        try:
            app.state.bundle = load_bundle(
                model_path or os.environ.get("MODEL_PATH", "models/development.joblib")
            )
        except Exception:
            logging.getLogger(__name__).exception("Model unavailable; readiness disabled")
        yield
        app.state.bundle = None

    app = FastAPI(title="FD001 development RUL", version="1.0.0", lifespan=lifespan)

    def ready_bundle():
        if app.state.bundle is None:
            raise HTTPException(503, "Model unavailable. Train or mount a compatible artifact.")
        return app.state.bundle

    @app.get("/health")
    def health():
        return {"status": "alive"}

    @app.get("/ready")
    def ready():
        return {"status": "ready", "model_id": ready_bundle().metadata["model_id"]}

    @app.get("/metadata")
    def metadata():
        return ready_bundle().metadata

    @app.get("/insights")
    def insights():
        """Read existing fitted importance; never fit, transform, or predict here."""
        bundle = ready_bundle()
        names = bundle.preprocessor.get_feature_names_out()
        weights = bundle.model.feature_importances_
        return {"model_id": bundle.metadata["model_id"],
                "retained_feature_count": len(names),
                "feature_importances": [{"feature": str(name), "importance": float(weight)}
                                        for name, weight in zip(names, weights)]}

    @app.post("/predict")
    def predict(request: PredictionRequest):
        bundle = ready_bundle()
        try:
            return bundle.predict(pd.DataFrame(request.observations))
        except (ValueError, TypeError, OverflowError) as exc:
            raise HTTPException(422, str(exc)) from exc

    return app


app = create_app()
