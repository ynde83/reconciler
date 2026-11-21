import json
from evaluator.events import SUBJECT
from evaluator.logging_json import log

async def publish_event(js, event: dict) -> bool:
    payload = json.dumps(event).encode()

    try:
        ack = await js.publish(
            SUBJECT,
            payload,
            headers={"Nats-Msg-Id": event["id"]},
            timeout=5,
        )

    except Exception as e:
        log("ERROR", "drift_event_publish_failed", event_id=event["id"], error=str(e))
        return False

    log("INFO", "drift_event_published", event_id=event["id"],
        stream=ack.stream, seq=ack.seq, duplicate=ack.duplicate)
    return True

if __name__ == "__main__":
    import asyncio
    import os

    import nats

    async def demo():
        from evaluator.logging_json import setup_logging

        setup_logging()
        nc = await nats.connect(os.environ.get("NATS_URL", "nats://localhost:4222"))
        js = nc.jetstream()
        event = {"id": "evt-demo0001", "type": "test"}

        await publish_event(js, event)
        await publish_event(js, event)
        await nc.close()

    asyncio.run(demo())