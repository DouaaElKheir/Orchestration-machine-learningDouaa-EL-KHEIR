"""API d'inference FastAPI pour le projet Adult Income."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from mlproject.config import MODEL_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ml: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    model_path = MODEL_DIR / "model.joblib"
    if not model_path.exists():
        logger.warning("Modele introuvable: %s", model_path)
    else:
        ml["model"] = joblib.load(model_path)
        logger.info("Modele charge depuis %s", model_path)

    yield

    ml.clear()
    logger.info("Ressources API liberees")


app = FastAPI(title="Adult Income API", version="0.1.0", lifespan=lifespan)


class Features(BaseModel):
    age: float = Field(..., ge=0)
    fnlwgt: float = Field(..., ge=0)
    education_num: float = Field(..., alias="education.num", ge=0)
    capital_gain: float = Field(..., alias="capital.gain", ge=0)
    capital_loss: float = Field(..., alias="capital.loss", ge=0)
    hours_per_week: float = Field(..., alias="hours.per.week", ge=0)
    workclass: str = Field(..., min_length=1)
    education: str = Field(..., min_length=1)
    marital_status: str = Field(..., alias="marital.status", min_length=1)
    occupation: str = Field(..., min_length=1)
    relationship: str = Field(..., min_length=1)
    race: str = Field(..., min_length=1)
    sex: str = Field(..., min_length=1)
    native_country: str = Field(..., alias="native.country", min_length=1)

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "examples": [
                {
                    "age": 39,
                    "fnlwgt": 77516,
                    "education.num": 13,
                    "capital.gain": 2174,
                    "capital.loss": 0,
                    "hours.per.week": 40,
                    "workclass": "State-gov",
                    "education": "Bachelors",
                    "marital.status": "Never-married",
                    "occupation": "Adm-clerical",
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": "Male",
                    "native.country": "United-States",
                }
            ]
        },
    }


class PredictionOut(BaseModel):
    prediction: int
    probability: float


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOut)
def predict(features: Features) -> PredictionOut:
    model = ml.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Modele non charge")

    row = pd.DataFrame([features.model_dump(by_alias=True)])
    proba = float(model.predict_proba(row)[0, 1])  # type: ignore[attr-defined]
    return PredictionOut(prediction=int(proba >= 0.5), probability=round(proba, 4))


@app.get("/model-info")
def model_info() -> dict[str, str]:
    return {"version": os.environ.get("MODEL_VERSION", "unknown")}
