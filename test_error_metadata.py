"""Structured error diagnostics for issue #20."""

from __future__ import annotations

import json
from types import SimpleNamespace

import requests

from agentknit._core import _run_turn


class _FailingCompletions:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def create(self, **kwargs):
        raise self.exc


class _FailingClient:
    def __init__(self, exc: Exception) -> None:
        self.base_url = SimpleNamespace(host="api.example.test")
        self.chat = SimpleNamespace(completions=_FailingCompletions(exc))


def _session(tmp_path, endpoint: str = "https://api.example.test/v1") -> tuple[dict, list]:
    events = []
    return {
        "messages": [{"role": "system", "content": "system"}],
        "tools": [],
        "structured": True,
        "tool_dispatch": {},
        "session_id": "error-test",
        "cache_key": "error-test",
        "endpoint": endpoint,
        "options": [],
        "streaming": False,
        "provider": None,
        "max_output_tokens": None,
        "usage_totals": {"prompt": 0, "completion": 0, "total": 0,
                         "cached": 0, "cache_write": 0},
        "strict_cache_proof": False,
        "llm_call_count": 0,
        "non_interactive": True,
        "log_path": tmp_path / "session.jsonl",
        "on_event": lambda event_type, data: events.append((event_type, data)),
        "_event_handlers": {},
    }, events


def test_http_error_event_and_log_include_structured_fields(tmp_path) -> None:
    response = requests.Response()
    response.status_code = 500
    exc = requests.HTTPError("server error", response=response)
    session, events = _session(tmp_path)

    _run_turn(_FailingClient(exc), "model", session, "hello")

    event = next(data for event_type, data in events if event_type == "error")
    assert event["error_class"] == "HTTPError"
    assert event["http_status"] == 500
    assert event["adapter"] == "http"
    assert isinstance(event["elapsed_s"], float)
    assert event["elapsed_s"] >= 0

    record = json.loads(session["log_path"].read_text().splitlines()[-1])
    assert {key: record[key] for key in ("error_class", "http_status", "adapter")} == {
        "error_class": "HTTPError", "http_status": 500, "adapter": "http",
    }
    assert record["elapsed_s"] == event["elapsed_s"]


def test_run_endpoint_errors_are_classified_as_subprocess(tmp_path) -> None:
    session, events = _session(tmp_path, endpoint="run:///tmp/completions")

    _run_turn(_FailingClient(RuntimeError("binary failed")), "run:///tmp/completions",
              session, "hello")

    event = next(data for event_type, data in events if event_type == "error")
    assert event["error_class"] == "RuntimeError"
    assert event["http_status"] is None
    assert event["adapter"] == "subprocess"
    assert isinstance(event["elapsed_s"], float)
