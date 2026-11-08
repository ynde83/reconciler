# trainer

Batch job. Reads a parquet dataset from S3, trains the model, uploads artifacts to MinIO and exits. In k8s it runs as a Job.

## Env

| Variable | Required | Default | Comment |
|---|---|---|---|
| `RUN_ID` | yes | | set by the controller |
| `DATASET_URI` | yes | | `s3://bucket/key.parquet` |
| `S3_ENDPOINT` | yes | | with scheme, `http://minio:9000` |
| `S3_ACCESS_KEY` | yes | | from Secret |
| `S3_SECRET_KEY` | yes | | from Secret |
| `MODELS_BUCKET` | no | `models` | |
| `BOOTSTRAP` | no | `false` | `true` also writes `models/current.json` |
| `SEED` | no | `42` | |

## Output

Artifacts go to `s3://<MODELS_BUCKET>/runs/<RUN_ID>/`:

- `model.joblib`
- `baseline_stats.json`
- `metrics.json`, uploaded last. If it exists, the run is complete.

Running again with the same `RUN_ID` does nothing: logs `run_already_completed` and exits 0.

Exit code 0 means artifacts are written. 1 means something broke (missing env, S3 unreachable, etc.), check the `trainer_failed` log line.

## Build and run

```bash
docker build -t selfheal/trainer:0.1.0 services/trainer

docker run --rm --network selfheal \
  -e RUN_ID=run-0001 -e DATASET_URI=s3://datasets/base/train.parquet -e BOOTSTRAP=true \
  -e S3_ENDPOINT=http://minio-dev:9000 -e S3_ACCESS_KEY=... -e S3_SECRET_KEY=... \
  selfheal/trainer:0.1.0
```

## Notes

- runs as uid 10001, doesn't listen on any port
- image is ~760 MB, about 590 MB of that is pip packages (pyarrow, scipy, pandas, botocore). Left as is for now
- keep `scikit-learn` and `joblib` pinned to the same versions as in model-api, otherwise the model may not load there
- drift-evaluator has to bin values the same way: `np.searchsorted(edges, x, side="right")`