import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import main
from test_api import make_model
from test_forward import FakeClient

GOOD = {"features": [0.5] * 10}


@pytest.fixture
def forwarding(monkeypatch):
    monkeypatch.setattr(main, "EVALUATOR_URL", "http://evaluator/ingest")
    main.app.state.model = make_model()
    main.app.state.run_id = "run-test"
    yield
    main.app.state.model = None
    main.app.state.run_id = None
    main.app.state.http = None


def test_predict_forwards_features(forwarding):
    fake = FakeClient()
    main.app.state.http = fake

    r = TestClient(main.app).post("/predict", json=GOOD)

    assert r.status_code == 200
    assert fake.calls == [GOOD]


def test_predict_ok_when_evaluator_down(forwarding):
    main.app.state.http = FakeClient(fail=True)

    r = TestClient(main.app).post("/predict", json=GOOD)

    assert r.status_code == 200
