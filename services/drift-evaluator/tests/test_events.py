import json
import sys
from pathlib import Path

import jsonschema

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

from evaluator.drift import DriftReport
from evaluator.events import build_event

SCHEMA = json.loads((HERE.parents[3] / "contracts" / "drift-detected.schema.json").read_text())


REPORT = DriftReport(
    psi_by_feature={"f3": 1.3, "f7": 1.2, "f9": 1.4},
    drifted_features=["f3", "f7", "f9"],
    drift_score=1.3,
    is_drift=True,
)

def make_event():
    return build_event(REPORT, "run-0001", "s3://datasets/incoming/latest.parquet", 0.2, 500)

def test_event_matches_contract():
    jsonschema.validate(make_event(), SCHEMA)

def test_event_is_json_serializable():
    json.dumps(make_event())

def test_event_ids_are_unique():
    first = make_event()
    second = make_event()

    assert first["id"] != second["id"]

def test_time_ends_with_z():
    event = make_event()
    assert event["time"].endswith("Z")