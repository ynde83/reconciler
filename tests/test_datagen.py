"""Проверяем главное требование M1.1: дрейф реально роняет F1 старой модели."""
import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from datagen import generate

FEATS = [f"f{i}" for i in range(10)]


def _fit(df):
    return make_pipeline(StandardScaler(), LogisticRegression()).fit(df[FEATS], df["y"])


def test_deterministic():
    assert generate(100, seed=1).equals(generate(100, seed=1))


def test_drift_hurts_old_model():
    model = _fit(generate(5000, seed=42))
    clean = generate(2000, seed=1)                      
    drifted = generate(2000, shift=1.5, concept="v2", seed=2)  

    f1_clean = f1_score(clean["y"], model.predict(clean[FEATS]))
    f1_drift = f1_score(drifted["y"], model.predict(drifted[FEATS]))

    assert f1_clean > 0.9
    assert f1_drift < 0.75 


def test_retrain_recovers():
    new = generate(3000, shift=1.5, concept="v2", seed=7)
    model = _fit(new.iloc[:2400])
    hold = new.iloc[2400:]
    assert f1_score(hold["y"], model.predict(hold[FEATS])) > 0.85