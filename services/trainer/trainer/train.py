import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

FEATURES = [f"f{i}" for i in range(10)]
N_BINS = 10

def compute_baseline(X_train: pd.DataFrame) -> dict:
    levels = np.linspace(0.1, 0.9, N_BINS - 1)
    features = {}
    for col in FEATURES:
        values = X_train[col].to_numpy()
        edges = np.unique(np.quantile(values, levels))
        idx = np.searchsorted(edges, values, side="right") 
        counts = np.bincount(idx, minlength=len(edges) + 1)
        props = counts / len(values)
        features[col] = {"edges": edges.tolist(), "props": props.tolist()}
    return {"features": features}


def train(df: pd.DataFrame, seed: int = 42) -> tuple[Pipeline, dict, dict]:
    X = df[FEATURES]
    y = df["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=seed, 
        stratify=y,

    )

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(random_state=seed))
    ]
    )

    pipe.fit(X_train, y_train)

    f1 = f1_score(y_test, pipe.predict(X_test))

    metrics = {
        "metric": "f1",
        "candidate": float(f1),
        "champion": None,
        "holdout_size": int(len(X_test)),
        "seed": seed,
    }
    baseline = compute_baseline(X_train)
    return pipe, metrics, baseline

if __name__ == "__main__":
    import json
    import sys

    sys.path.insert(0, "scripts")
    from datagen import generate

    model, metrics, baseline = train(generate(5000))
    print(metrics)
    print(baseline["features"]["f0"])
    print(round(sum(baseline["features"]["f0"]["props"]), 6))
    json.dumps(baseline)
    print("json ok")