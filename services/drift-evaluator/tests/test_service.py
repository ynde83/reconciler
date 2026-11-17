import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))             
sys.path.insert(0, str(HERE.parents[3] / "scripts")) 
sys.path.insert(0, str(HERE.parents[0]))              

from evaluator import main  
from datagen import FEATURE_NAMES, generate 
from test_psi import make_baseline_df  

client = TestClient(main.app)


@pytest.fixture(autouse=True)
def clean():
    main.window.clear()
    main.state["baseline"] = make_baseline_df(generate(5000, seed=42))
    main.state["baseline_run_id"] = "run-test"
    main.detector.__init__(cooldown_sec=300)
    yield
    main.window.clear()


def fill(df):
    for row in df[FEATURE_NAMES].to_numpy():
        assert client.post("/ingest", json={"features": row.tolist()}).status_code == 202


def test_ingest_validation():
    assert client.post("/ingest", json={"features": [1, 2]}).status_code == 422


def test_warmup_status():
    fill(generate(50, seed=1))
    body = client.get("/status").json()
    assert body["state"] == "WARMUP"
    assert body["window_fill"] == 50
    assert main.run_check() is None


def test_window_is_capped():
    fill(generate(600, seed=1))
    assert client.get("/status").json()["window_fill"] == 500


def test_normal_traffic_no_drift():
    fill(generate(500, seed=1))
    assert main.handle_report(main.run_check(), now=0) is False
    assert main.handle_report(main.run_check(), now=15) is False
    assert client.get("/status").json()["state"] == "OK"


def test_drift_confirmed_after_two_checks():
    fill(generate(500, shift=1.5, concept="v2", seed=2))
    assert main.handle_report(main.run_check(), now=0) is False
    assert main.handle_report(main.run_check(), now=15) is True
    assert set(client.get("/status").json()["drifted_features"]) == {"f3", "f7", "f9"}