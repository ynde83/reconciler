import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from traffic_generator import make_rows

def test_rows_shape():
    rows = make_rows(5, "none", seed=0)
    assert len(rows) == 5
    assert all(len(row) == 10 for row in rows)

def test_shift_moves_drift_features():
    normal = make_rows(500, "none", seed=0)
    drifted = make_rows(500, "shift", seed=0)
    mean_normal = sum(row[3] for row in normal) / len(normal)
    mean_drifted = sum(row[3] for row in drifted) / len(drifted)
    assert mean_drifted - mean_normal > 1

    