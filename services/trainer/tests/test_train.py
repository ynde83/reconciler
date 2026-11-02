import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

HERE = Path(__file__).resolve()

sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[3] / "scripts"))

from trainer.train import train, FEATURES
from datagen import generate

def test_deterministic():
    df = generate(2000)

    _, m1, _ = train(df, seed=42)
    _, m2, _ = train(df, seed=42)

    assert m1["candidate"] == m2["candidate"]

def test_small_data():
    df = generate(500) #prepare small dataset

    _, metrics, _ = train(df)

    assert 0 <= metrics["candidate"] <= 1
    assert metrics["holdout_size"] == 100

def test_baseline_format():
    df = generate(2000)

    _, _, baseline = train(df)
    features = baseline["features"]

    assert list(features) == FEATURES

    for name, stats in features.items():
        assert len(stats["edges"]) == 9, name
        assert len(stats["props"]) == 10, name
        assert sum(stats["props"]) == pytest.approx(1.0), name

def test_different_seeds():
    df = generate(2000)

    _, m1, _ = train(df, seed=1)
    _, m2, _ = train(df, seed=2)

    assert m1["candidate"] != m2["candidate"]