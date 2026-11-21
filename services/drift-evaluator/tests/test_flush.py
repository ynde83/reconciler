import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[3] / "scripts"))

from evaluator import main

class FakeJS:
    def __init__(self, fail=False):
        self.fail = fail
        self.sent = []

    async def publish(self, subject, payload, headers=None, timeout=None):
        if self.fail:
            raise TimeoutError("nats unavailable")
        self.sent.append(headers["Nats-Msg-Id"])
        return SimpleNamespace(stream="MLOPS", seq=len(self.sent), duplicate=None)

EVENT = {"id": "evt-0000abcd", "type": "test"}


def test_pending_cleared_after_success():
    js = FakeJS()
    main.state["pending_event"] = EVENT
    asyncio.run(main.flush_pending(js))
    assert main.state["pending_event"] is None
    assert js.sent == ["evt-0000abcd"]

def test_pending_kept_after_failure():
    js = FakeJS(fail=True)
    main.state["pending_event"] = EVENT
    asyncio.run(main.flush_pending(js))
    assert main.state["pending_event"] == EVENT       
    assert js.sent == []

def test_retry_uses_same_id():
    main.state["pending_event"] = EVENT
    asyncio.run(main.flush_pending(FakeJS(fail=True)))   
    js = FakeJS()
    asyncio.run(main.flush_pending(js))                 
    assert js.sent == ["evt-0000abcd"]       