import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import forward
from app.forward import forward_to_evaluator, should_log

URL = "http://evaluator/ingest"
ROW = [0.0] * 10


class FakeClient:
    def __init__(self, fail=False, status=202):
        self.fail = fail
        self.status = status
        self.calls = []

    async def post(self, url, json=None):
        if self.fail:
            raise httpx.ConnectError("evaluator down")
        self.calls.append(json)
        return SimpleNamespace(status_code=self.status)


def test_should_log_first_time():
    assert should_log(100, float("-inf"), 60) is True


def test_should_log_too_early():
    assert should_log(130, 100, 60) is False


def test_should_log_exactly_interval():
    assert should_log(160, 100, 60) is True


def test_forward_sends_features():
    client = FakeClient()
    asyncio.run(forward_to_evaluator(client, URL, ROW))
    assert client.calls == [{"features": ROW}]


def test_forward_disabled_without_url():
    client = FakeClient()
    asyncio.run(forward_to_evaluator(client, "", ROW))
    assert client.calls == []


def test_forward_survives_evaluator_down():
    asyncio.run(forward_to_evaluator(FakeClient(fail=True), URL, ROW))


def test_forward_survives_bad_status():
    asyncio.run(forward_to_evaluator(FakeClient(status=500), URL, ROW))


def test_error_logged_once_per_interval(monkeypatch):
    calls = []
    monkeypatch.setattr(forward, "log", lambda *args, **kwargs: calls.append(args[1]))
    monkeypatch.setattr(forward, "_last_error_log", float("-inf"))
    for _ in range(3):
        asyncio.run(forward_to_evaluator(FakeClient(fail=True), URL, ROW))
    assert calls == ["evaluator_forward_failed"]
