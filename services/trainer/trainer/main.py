import os
import sys
import json
import io
import time
import traceback
from datetime import datetime, timezone

import boto3
import joblib
import pandas as pd
from botocore.config import Config
from botocore.exceptions import ClientError

from trainer.train import train


REQUIRED = ["RUN_ID", "DATASET_URI", "S3_ENDPOINT", "S3_ACCESS_KEY", "S3_SECRET_KEY"]

def log(level: str, msg: str, **fields) -> None:
    record = {
        "time": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "level": level,
        "service": "trainer",
        "msg": msg,
        **fields,
    }
    print(json.dumps(record), flush=True)

def now_utc() -> str:
    """Текущее время UTC в формате RFC3339: 2026-11-09T10:19:12Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")



def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"bad s3 uri, expected s3:// : {uri}")

    rest = uri.removeprefix("s3://")

    if "/" not in rest:
        raise ValueError(f"no key in s3 uri: {uri}")

    bucket, key = rest.split("/", 1)
    if not bucket or not key:
        raise ValueError(f"empty bucket or key in s3 uri: {uri}")
    return bucket, key

def load_config() -> dict:
    missing = [name for name in REQUIRED if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"missing env: {missing}")
    return {
        "run_id": os.environ["RUN_ID"],
        "dataset_uri": os.environ["DATASET_URI"],
        "s3_endpoint": os.environ["S3_ENDPOINT"],
        "s3_access_key": os.environ["S3_ACCESS_KEY"],
        "s3_secret_key": os.environ["S3_SECRET_KEY"],
        "models_bucket": os.environ.get("MODELS_BUCKET", "models"),
        "bootstrap": os.environ.get("BOOTSTRAP", "false").lower() == "true",
        "seed": int(os.environ.get("SEED", "42")),
    }

def make_s3(cfg: dict): 
    return boto3.client(
        "s3",
        endpoint_url = cfg["s3_endpoint"],
        aws_access_key_id=cfg["s3_access_key"],
        aws_secret_access_key=cfg["s3_secret_key"],
        region_name="us-east-1",
        config=Config(s3={"addressing_style": "path"}),
    )

def object_exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise

def put_json(s3, bucket: str, key: str, data: dict) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data, indent=2).encode(),
        ContentType="application/json",
    )


def run(cfg: dict, s3) -> None:
    run_id = cfg["run_id"]
    bucket = cfg["models_bucket"]
    prefix = f"runs/{run_id}"

    if object_exists(s3, bucket, f"{prefix}/metrics.json"):
        log("INFO", "run_already_completed", run_id=run_id)
        return

    started = time.monotonic()

    data_bucket, data_key = parse_s3_uri(cfg["dataset_uri"])
    body = s3.get_object(Bucket=data_bucket, Key=data_key)["Body"].read()
    df = pd.read_parquet(io.BytesIO(body))
    log("INFO", "dataset_loaded", run_id=run_id, rows=len(df))

    model, metrics, baseline = train(df, seed=cfg["seed"])
    log("INFO", "model_trained", run_id=run_id, candidate=metrics["candidate"])

    buf = io.BytesIO()
    joblib.dump(model, buf)
    s3.put_object(Bucket=bucket, Key=f"{prefix}/model.joblib", Body=buf.getvalue())
    put_json(s3, bucket, f"{prefix}/baseline_stats.json", baseline)

    metrics.update({
        "run_id": run_id,
        "trained_on": cfg["dataset_uri"],
        "duration_sec": round(time.monotonic() - started, 1),
        "finished_at": now_utc(),
    })
    put_json(s3, bucket, f"{prefix}/metrics.json", metrics)
    log("INFO", "artifacts_uploaded", run_id=run_id, prefix=f"s3://{bucket}/{prefix}/")

    if cfg["bootstrap"]:
        put_json(s3, bucket, "current.json", {
            "run_id": run_id,
            "model_uri": f"s3://{bucket}/{prefix}/model.joblib",
            "baseline_uri": f"s3://{bucket}/{prefix}/baseline_stats.json",
            "updated_at": now_utc(),
        })
        log("INFO", "current_pointer_written", run_id=run_id)


def main() -> int:
    try:
        cfg = load_config()
        log("INFO", "trainer_start", run_id=cfg["run_id"], dataset_uri=cfg["dataset_uri"],
            bootstrap=cfg["bootstrap"], seed=cfg["seed"])
        run(cfg, make_s3(cfg))
        log("INFO", "trainer_done", run_id=cfg["run_id"])
        return 0
    except Exception as e:
        log("ERROR", "trainer_failed", error=str(e), exc_info=traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())


