import asyncio
import os
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
import nats

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from evaluator.detector import DriftDetector
from evaluator.drift import check_drift
from evaluator.logging_json import log, setup_logging
from evaluator.events import build_event
from evaluator.publisher import publish_event

setup_logging()

FEATURES = [f"f{i}" for i in range(10)]

WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", "500"))
MIN_SAMPLES = int(os.environ.get("MIN_SAMPLES", "200"))
CHECK_INTERVAL = float(os.environ.get("CHECK_INTERVAL", "15"))
PSI_THRESHOLD = float(os.environ.get("PSI_THRESHOLD", "0.2"))
MIN_DRIFTED_FEATURES = int(os.environ.get("MIN_DRIFTED_FEATURES", "3"))
COOLDOWN_SEC = float(os.environ.get("COOLDOWN_SEC", "300"))
NATS_URL = os.environ.get("NATS_URL", "nats://localhost:4222")
DATASET_URI = os.environ.get("DATASET_URI", "s3://datasets/incoming/latest.parquet")
window = deque(maxlen=WINDOW_SIZE)
lock = threading.Lock()
detector = DriftDetector(cooldown_sec=COOLDOWN_SEC)
state = {"baseline": None, "baseline_run_id": None, "last_psi": {}, "drifted_features": [], "js": None, "pending_event": None}

def run_check():
    with lock:
        snapshot = list(window)

    if len(snapshot) < MIN_SAMPLES or state["baseline"] is None:
        return None

    arr = np.array(snapshot)
    by_feature = {name: arr[:, i] for i, name in enumerate(FEATURES)}
    return check_drift(by_feature, state["baseline"], PSI_THRESHOLD, MIN_DRIFTED_FEATURES)

def handle_report(report, now: float) -> bool:
    state["last_psi"] = {k: round(v, 4) for k, v in report.psi_by_feature.items()}
    state["drifted_features"] = report.drifted_features
    return detector.update(report.is_drift, now)

async def on_nats_error(e: Exception):
    log("WARNING", "nats_error", error=str(e))

async def flush_pending(js):
    event = state["pending_event"]
    if event is None:
        return
    sent = await publish_event(js, event)
    if sent:
        state["pending_event"] = None


async def check_loop():
    loop = asyncio.get_running_loop()
    while True:
        await asyncio.sleep(CHECK_INTERVAL)
        try:
            await flush_pending(state["js"])

            report = await loop.run_in_executor(None, run_check)
            if report is None:
                continue

            fire = handle_report(report, time.time())

            log("INFO", "drift_check", state=detector.state, window_fill=len(window),
                max_psi=max(state["last_psi"].values()))
            if fire:
                log("WARNING", "drift_confirmed", drifted_features=report.drifted_features,
                    drift_score=round(report.drift_score, 4))
                state["pending_event"] = build_event(
                    report,
                    current_run_id=state["baseline_run_id"],
                    dataset_uri=DATASET_URI,
                    threshold=PSI_THRESHOLD,
                    window_size=WINDOW_SIZE,
                )

                await flush_pending(state["js"])
        except Exception as e:
            log("ERROR", "drift_check_failed", error=str(e))
@asynccontextmanager
async def lifespan(app: FastAPI):
   try:
       nc = await asyncio.wait_for(
           nats.connect(NATS_URL, error_cb = on_nats_error, max_reconnect_attempts=-1),
           timeout=5,
        )
   except Exception as e:
        log("ERROR", "nats_connect_failed", url=NATS_URL, error=repr(e))
        raise

   state["js"] = nc.jetstream()
   log("INFO", "nats_connected", url=NATS_URL)

   task = asyncio.create_task(check_loop())
   yield
   task.cancel()
   await nc.close()

app = FastAPI(title="drift-evaluator", lifespan=lifespan)

class IngestRequest(BaseModel):
    features: list[float] = Field(min_length=10, max_length=10)

@app.post("/ingest", status_code=202)
def ingest(req: IngestRequest):
    with lock:
        window.append(req.features)
    return {"accepted": True}

@app.get("/status")
def status():
    with lock:
        fill = len(window)

    return {
        "state": "WARMUP" if fill < MIN_SAMPLES else detector.state,
        "window_fill": fill,
        "last_psi": state["last_psi"],
        "drifted_features": state["drifted_features"],
        "cooldown_remaining_sec": round(detector.cooldown_remaining(time.time()), 1),
        "baseline_run_id": state["baseline_run_id"],
    }