import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.main import FEATURES, app

client = TestClient(app)
GOOD = {"features": [0.0] * 10}     #correct response


def make_model():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size = (200,10)), columns = FEATURES)
    y = (X["f0"] > 0 ).astype(int)
    return make_pipeline(StandardScaler(), LogisticRegression()).fit(X, y)

@pytest.fixture
def loaded():
    app.state.model = make_model()
    app.state.run_id = "run-test"
    app.state.model_uri = "s3://models/runs/run-test/model.joblib"
    app.state.loaded_at = "2026-11-05T00:00:00Z"

    yield

    app.state.model = None
    app.state.run_id = None
    app.state.model_uri = None
    app.state.loaded_at = None

def test_health_always_ok():
    assert client.get("/health").status_code == 200

def test_without_loaded_model():
    assert client.get("/ready").status_code == 503
    assert client.get("/model-info").status_code == 503
    assert client.post("/predict", json=GOOD).status_code == 503

def test_wrong_lenth():
    assert client.post("/predict", json={"features": [1, 2, 3]}).status_code == 422

def test_predict_with_model(loaded):
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0 <= body["proba"] <= 1
    assert body["run_id"] == "run-test"


def test_ready_and_info_with_model(loaded):
    assert client.get("/ready").status_code == 200
    assert client.get("/model-info").json()["run_id"] == "run-test"


def test_too_many_features():
    body = {"features": [0.0] * 11}
    r = client.post("/predict", json = body)
    assert r.status_code == 422

def test_string_failure():
    body = {"features": ["abc"] + [0.0] * 9}
    r = client.post("predict", json = body)
    assert r.status_code == 422