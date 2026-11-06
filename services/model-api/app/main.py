from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from app.loader import load_model, make_s3

MODEL_ENV = [
    "MODEL_URI",
    "MODEL_RUN_ID",
    "S3_ENDPOINT",
    "S3_ACCESS_KEY",
    "S3_SECRET_KEY",
]


def log(level: str, msg: str, **fields) -> None:
    record = {
        "time": datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z"),
        "level": level,
        "service": "model-api",
        "msg": msg,
        **fields,
    }
    print(json.dumps(record), flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    missing = [name for name in MODEL_ENV if not os.environ.get(name)]
    if missing:
        log("ERROR", "model_load_failed", error=f"missing env {missing}")
    else:
        model_uri = os.environ["MODEL_URI"]
        try:
            s3 = make_s3(
                os.environ["S3_ENDPOINT"],
                os.environ["S3_ACCESS_KEY"],
                os.environ["S3_SECRET_KEY"],
            )
            model = load_model(s3, model_uri)

            app.state.run_id = os.environ["MODEL_RUN_ID"]
            app.state.model_uri = model_uri
            app.state.loaded_at = datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            app.state.model = model

            log("INFO", "model_loaded", run_id=app.state.run_id, model_uri=model_uri)
        except Exception as e:
            log("ERROR", "model_load_failed", model_uri=model_uri, error=str(e))
    yield


FEATURES = [f"f{i}" for i in range(10)]

app = FastAPI(title="model-api", lifespan=lifespan)

app.state.model = None
app.state.run_id = None
app.state.model_uri = None
app.state.loaded_at = None


class PredictRequest(BaseModel):
    features: list[float] = Field(min_length=10, max_length=10)


class PredictResponse(BaseModel):
    prediction: int
    proba: float
    run_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready"}


@app.get("/model-info")
def model_info():
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {
        "run_id": app.state.run_id,
        "model_uri": app.state.model_uri,
        "loaded_at": app.state.loaded_at,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    X = pd.DataFrame([req.features], columns=FEATURES)

    proba = app.state.model.predict_proba(X)[0][1]
    prediction = int(proba >= 0.5)

    return PredictResponse(prediction=prediction, proba=proba, run_id=app.state.run_id)
