import uuid
from datetime import datetime, timezone

from evaluator.drift import DriftReport

SUBJECT = "mlops.drift.detected"
EVENT_TYPE = "com.selfheal.drift.detected.v1"
SOURCE = "/mlops/drift-evaluator"

def build_event(report: DriftReport,
    current_run_id: str,
    dataset_uri: str,
    threshold: float,
    window_size: int,
    model_name: str = "churn-clf",
    ) -> dict:
    event_id = "evt-" + uuid.uuid4().hex[:8]

    event_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    data = {
        "model_name": model_name,
        "current_run_id": current_run_id,
        "metric": "psi",
        "threshold": threshold,
        "drift_score": round(report.drift_score, 4),
        "drifted_features": report.drifted_features,
        "window_size": window_size,
        "dataset_uri": dataset_uri,
    }


    return {
        "specversion": "1.0",
        "id": event_id,
        "source": SOURCE,
        "type": EVENT_TYPE,
        "time": event_time,
        "datacontenttype": "application/json",
        "data": data,
    }

if __name__ == "__main__":
    import json

    report = DriftReport(
        psi_by_feature={"f3": 1.3, "f7": 1.2, "f9": 1.4},
        drifted_features=["f3", "f7", "f9"],
        drift_score=1.312345,
        is_drift=True,
    )

    event = build_event(report, "run-0001", "s3://datasets/incoming/latest.parquet", 0.2, 500)
    print(json.dumps(event, indent=2))