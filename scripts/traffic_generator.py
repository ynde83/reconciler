import argparse
import httpx
import time
from collections import Counter

from datagen import FEATURE_NAMES, generate


# drift parameters
SHIFT = 1.5
CONCEPT = "v2"

REPORT_EVERY_SEC = 10


def parse_args():
    p = argparse.ArgumentParser(description="Traffic generator for model-api")
    p.add_argument("--rps", type=float, default=5)
    p.add_argument("--duration", type=int, default=120)
    p.add_argument("--target", default="http://localhost:8000")
    p.add_argument("--drift", choices=["drift", "none"], default="none")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def make_rows(n, drift, seed):
    if drift == "shift":
        df = generate(n, shift=SHIFT, concept=CONCEPT, seed=seed)
    else:
        df = generate(n, seed)
    return df[FEATURE_NAMES].values.tolist()


def run(rows, target, rps):
    counts = Counter()
    last_report = time.monotonic()

    with httpx.Client(timeout=2) as http:
        for row in rows:
            try:
                r = http.post(f"{target}/predict", json={"features": row})
                counts[f"{r.status_code // 100}xx"] += 1
            except httpx.HTTPError:
                counts["error"] += 1

            now = time.monotonic()
            if now - last_report >= REPORT_EVERY_SEC:
                print(dict(counts), flush=True)
                last_report = now

            time.sleep(1 / rps)

    print("done:", dict(counts), flush=True)
    return counts


def main():
    args = parse_args()
    n = int(args.rps * args.duration)
    rows = make_rows(n, args.drift, args.seed)
    print(f"sending {n} requests to {args.target}, drift={args.drift}", flush=True)
    run(rows, args.target, args.rps)


if __name__ == "__main__":
    main()
