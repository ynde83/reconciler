import time
from app.logging_json import log

ERROR_LOG_INTERVAL = 60.0
_last_error_log = float("-inf")


def should_log(now: float, last: float, interval: float) -> bool:
    return now - last >= interval


async def forward_to_evaluator(client, url: str, features: list[float]):
    global _last_error_log
    if not url:
        return

    try:
        r = await client.post(url, json={"features": features})
        if r.status_code != 202:
            raise RuntimeError(f"unexpected status: {r.status_code}")

    except Exception as e:
        now = time.monotonic()
        if should_log(now, _last_error_log, ERROR_LOG_INTERVAL):
            log("WARNING", "evaluator_forward_failed", url=url, error=repr(e))
            _last_error_log = now
