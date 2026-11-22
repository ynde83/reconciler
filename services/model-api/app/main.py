import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.forward import forward_to_evaluator
from app.loader import load_model, make_s3
from app.logging_json import log, setup_logging

setup_logging()

FEATURES = [f"f{i}" for i in range(10)]

MODEL_ENV = [
    "MODEL_URI",
    "MODEL_RUN_ID",
    "S3_ENDPOINT",
    "S3_ACCESS_KEY",
    "S3_SECRET_KEY",
]

EVALUATOR_URL = os.environ.get("EVALUATOR_URL", "")

EVALUATOR_TIMEOUT = 0.3


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(timeout=EVALUATOR_TIMEOUT)

    missing = [name for name in MODEL_ENV if not os.environ.get(name)]
    if missing:
        log("ERROR", "model_load_failed", error=f"missing env: {missing}")
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

    log("INFO", "evaluator_forwarding", enabled=bool(EVALUATOR_URL), url=EVALUATOR_URL)

    yield

    await app.state.http.aclose()


app = FastAPI(title="model-api", lifespan=lifespan)

app.state.model = None
app.state.run_id = None
app.state.model_uri = None
app.state.loaded_at = None
app.state.http = None


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
def predict(req: PredictRequest, background_tasks: BackgroundTasks):
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    started = time.perf_counter()
    X = pd.DataFrame([req.features], columns=FEATURES)
    proba = float(app.state.model.predict_proba(X)[0][1])
    prediction = int(proba >= 0.5)
    log(
        "INFO",
        "predict_ok",
        run_id=app.state.run_id,
        n_features=len(req.features),
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
    )

    background_tasks.add_task(
        forward_to_evaluator, app.state.http, EVALUATOR_URL, req.features
    )

    return PredictResponse(prediction=prediction, proba=proba, run_id=app.state.run_id)
