"""
FastAPI backend for 001_linear_regression_engine
Serves prediction endpoints backed by the trained scikit-learn model.

Endpoints:
  GET  /health          Liveness check
  GET  /schema          Returns feature names and types
  POST /predict         Single-row inference
  POST /predict/batch   Batch inference (up to 100 rows)
  GET  /metrics         Evaluation metrics from training
"""

from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator, model_validator

# ─── Configuration ────────────────────────────────────────────────────────────
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/model.pkl"))
METRICS_PATH = Path("models/metrics.json")
FEATURES: list[str] = %%FEATURES%%    # injected by generate_deploy.py
REPO_NAME: str = "001_linear_regression_engine"
MAX_BATCH_SIZE = 100

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "info").upper())

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title=REPO_NAME.replace("_", " ").title(),
    description=f"AIML Engineering Lab — {REPO_NAME} inference API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict in production to your frontend domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Model cache ──────────────────────────────────────────────────────────────
_model: Any = None


def get_model() -> Any:
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                "Run the training notebook to generate it."
            )
        _model = joblib.load(MODEL_PATH)
        logger.info("Model loaded from %s", MODEL_PATH)
    return _model


# ─── Schemas ──────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    features: dict[str, float]

    @model_validator(mode="after")
    def validate_features(self) -> "PredictRequest":
        missing = set(FEATURES) - set(self.features.keys())
        if missing:
            raise ValueError(f"Missing features: {sorted(missing)}")
        return self


class PredictResponse(BaseModel):
    prediction: float | str
    probability: float | None = None
    input_features: dict[str, float]


class BatchPredictRequest(BaseModel):
    rows: list[dict[str, float]]

    @field_validator("rows")
    @classmethod
    def cap_batch_size(cls, v: list) -> list:
        if len(v) > MAX_BATCH_SIZE:
            raise ValueError(f"Batch size exceeds maximum of {MAX_BATCH_SIZE}")
        return v


class BatchPredictResponse(BaseModel):
    predictions: list[float | str]
    count: int


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health", tags=["ops"])
def health() -> dict:
    return {"status": "ok", "model": MODEL_PATH.name, "repo": REPO_NAME}


@app.get("/schema", tags=["ops"])
def schema() -> dict:
    return {
        "features": FEATURES,
        "feature_count": len(FEATURES),
        "model_file": MODEL_PATH.name,
    }


@app.get("/metrics", tags=["ops"])
def metrics() -> dict:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return {"message": "No metrics file found. Run the training notebook."}


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(req: PredictRequest) -> PredictResponse:
    model = get_model()
    x = np.array([[req.features[f] for f in FEATURES]])

    try:
        pred = model.predict(x)[0]
        prob: float | None = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(x)[0]
            prob = float(max(probs))
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))

    return PredictResponse(
        prediction=float(pred) if isinstance(pred, (int, float, np.number)) else str(pred),
        probability=prob,
        input_features=req.features,
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["inference"])
def predict_batch(req: BatchPredictRequest) -> BatchPredictResponse:
    model = get_model()
    rows = [{f: row.get(f, 0.0) for f in FEATURES} for row in req.rows]
    X = np.array([[row[f] for f in FEATURES] for row in rows])

    try:
        preds = model.predict(X).tolist()
    except Exception as e:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=str(e))

    return BatchPredictResponse(predictions=preds, count=len(preds))
