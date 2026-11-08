# model-api

FastAPI service. Downloads the model from S3 on startup and serves predictions. One worker, the model lives in process memory.

## Env

| Variable | Required | Comment |
|---|---|---|
| `MODEL_URI` | yes | `s3://models/runs/<run_id>/model.joblib` |
| `MODEL_RUN_ID` | yes | returned in `/predict` and `/model-info` |
| `S3_ENDPOINT` | yes | with scheme, `http://minio:9000` |
| `S3_ACCESS_KEY` | yes | from Secret |
| `S3_SECRET_KEY` | yes | from Secret |

If the model can't be loaded the process keeps running, `/ready` returns 503, and the reason is in the `model_load_failed` log line.

## Endpoints

Port 8000.

| Endpoint | |
|---|---|
| `POST /predict` | `{"features": [10 floats]}` returns `{"prediction", "proba", "run_id"}`, 422 on bad input |
| `GET /health` | always 200, use for liveness |
| `GET /ready` | 200 after the model is loaded, 503 before, use for readiness |
| `GET /model-info` | `{"run_id", "model_uri", "loaded_at"}` |

## Build and run

```bash
docker build -t selfheal/model-api:0.1.0 services/model-api

docker run -d --name model-api --network selfheal -p 8000:8000 \
  -e MODEL_URI=s3://models/runs/run-0001/model.joblib -e MODEL_RUN_ID=run-0001 \
  -e S3_ENDPOINT=http://minio-dev:9000 -e S3_ACCESS_KEY=... -e S3_SECRET_KEY=... \
  selfheal/model-api:0.1.0

curl -s localhost:8000/ready
```

Tests run without MinIO:

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Notes

- runs as uid 10001
- logs are one JSON object per line, uvicorn access lines too (`"msg": "http_request"`)