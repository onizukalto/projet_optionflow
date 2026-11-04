"""API d'inférence.  Lancement : uvicorn api.main:app --reload"""
import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.joblib"))

app = FastAPI(title="QQQ Options Flow API",
              description="Probabilité que QQQ soit plus haut dans 5 minutes.")
_bundle = None


def get_bundle():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise HTTPException(503, f"Modèle introuvable ({MODEL_PATH}) : lance `python train.py`.")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


class Features(BaseModel):
    return_1m: float
    return_5m: float
    return_15m: float
    call_share_vol_5m: float = Field(ge=0, le=1)
    net_premium_share_5m: float = Field(ge=-1, le=1)
    total_volume_5m: float = Field(ge=0)
    call_share_vol_15m: float = Field(ge=0, le=1)
    net_premium_share_15m: float = Field(ge=-1, le=1)
    total_volume_15m: float = Field(ge=0)
    call_share_vol_30m: float = Field(ge=0, le=1)
    net_premium_share_30m: float = Field(ge=-1, le=1)
    total_volume_30m: float = Field(ge=0)
    volume_vs_day_avg: float = Field(ge=0)


class Prediction(BaseModel):
    proba_up: float
    direction: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict", response_model=Prediction)
def predict(features: Features):
    bundle = get_bundle()
    X = pd.DataFrame([features.model_dump()])[bundle["features"]]
    proba = float(bundle["model"].predict_proba(X)[0, 1])
    return Prediction(proba_up=round(proba, 4), direction="UP" if proba > 0.5 else "DOWN")
