from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

FEATURES = [f"f{i}" for i in range(10)]

app = FastAPI(title="model-api")

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
