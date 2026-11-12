import numpy as np
import sys
from pathlib import Path


HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))              # services/drift-evaluator -> пакет evaluator
sys.path.insert(0, str(HERE.parents[3] / "scripts"))
from evaluator.drift import bin_shares, psi, check_drift
from datagen import FEATURE_NAMES, generate

def make_baseline(train):
    edges = np.quantile(train, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    probs = bin_shares(train, edges)
    return edges, probs

def test_same_distribution_psi():   #tests how psi func calculate normal drift value
    rng = np.random.default_rng(42)
    train = rng.normal(0, 1, 5000)
    window = rng.normal(0, 1, 500)
    edges, probs = make_baseline(train)
    assert psi(probs, window, edges) < 0.1

def test_shift_high_psi():  #tests how psi func calculate high drift value
    rng = np.random.default_rng(42)
    train = rng.normal(0, 1, 5000)
    window = rng.normal(1.5 * np.std(train), 1, 500)
    edges, probs = make_baseline(train)
    assert psi(probs, window, edges) > 0.2

def make_baseline_df(df):   #baseline to df format
    features = {}
    for name in FEATURE_NAMES:
        values = df[name].to_numpy()
        edges = np.unique(np.quantile(values, np.linspace(0.1, 0.9, 9)))
        features[name] = {"edges": edges.tolist(), "props": bin_shares(values, edges).tolist()}
    return {"features": features}

def to_window(df):
    return {name: df[name].to_numpy() for name in FEATURE_NAMES}

BASELINE = make_baseline_df(generate(5000, seed=42))

def test_no_drift_on_normal_window():
    report = check_drift(to_window(generate(500, seed=1)), BASELINE)
    assert report.is_drift is False
    assert report.drift_score == 0.0

def test_drift_on_shifted_window():
    report = check_drift(to_window(generate(500, shift=1.5, concept="v2", seed=2)), BASELINE)
    assert report.is_drift is True
    assert set(report.drifted_features) == {"f3", "f7", "f9"}


def test_constant_feature():
    train = np.full(1000, 3.0)
    edges = np.unique(np.quantile(train, np.linspace(0.1, 0.9, 9)))
    probs = bin_shares(train, edges)
    assert np.isfinite(psi(probs, np.full(200, 3.0), edges))



def test_empty_bins():
    rng = np.random.default_rng(0)
    train = rng.normal(0, 1, 5000)
    edges = np.quantile(train, np.linspace(0.1, 0.9, 9))
    value = psi(bin_shares(train, edges), np.full(500, 10.0), edges)
    assert np.isfinite(value) and value > 1