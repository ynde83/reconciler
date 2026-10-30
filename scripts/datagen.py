import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_FEATURES = 10
FEATURE_NAMES = [f"f{i}" for i in range(N_FEATURES)]

DRIFT_FEATURES = [3, 7, 9]    #drift parameters
DRIFT_SCALE = 1.3
NOISE_STD = 0.5


W_BASE = np.array([1.5, -1.0, 0.8, 1.2, 0.0, 0.5, -0.7, 1.0, 0.3, -1.3])      #hidden function weights
W_V2 = W_BASE * np.array([1, 1, 1, -1, 1, 1, 1, 1, 1, -1])

WEIGHTS = {"base": W_BASE, "v2": W_V2}

def generate(n: int, shift: float = 0.0, concept: str = "base", seed: int = 42) -> pd.DataFrame:    # generate data
  if concept not in WEIGHTS:
    raise ValueError(f"concept must be one of {list(WEIGHTS)}, received: {concept}")
  rng = np.random.default_rng(seed)
  X = rng.normal(0.0, 1.0, size=(n, N_FEATURES))

  if shift:
        X[:, DRIFT_FEATURES] = X[:, DRIFT_FEATURES] * DRIFT_SCALE + shift   #make data drift

  logit = (X * WEIGHTS[concept]).sum(axis=1) + rng.normal(0.0, NOISE_STD, size=n)
  y = (logit > 0).astype(int)

  df = pd.DataFrame(X, columns=FEATURE_NAMES)
  df["y"] = y
  return df

def main() -> None:
  p = argparse.ArgumentParser(description="generation of parquet dataset")
  p.add_argument("--n", type=int, default=5000)
  p.add_argument("--shift", type=float, default=0.0)
  p.add_argument("--concept", choices=list(WEIGHTS), default="base")
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--out", type=Path, default=Path("data/base/train.parquet"))
  args = p.parse_args()

  df = generate(args.n, args.shift, args.concept, args.seed)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  df.to_parquet(args.out, index=False)
  print(f"saved {len(df)} rows -> {args.out}, part of y=1: {df['y'].mean():.2f}")

if __name__ == "__main__":
   main()