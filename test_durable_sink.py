"""Public session-directory and synchronous durable-sink coverage."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from agentknit import init_session, run_task
from agentknit._core import _handle_tool_call


class Recorder:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def append(self, record: dict) -> None:
        self.records.append(record)


def _schema(tool_dispatch: dict | None = None) -> dict:
    return {
        "model": "test/model",
        "endpoint": "https://api.example.test/v1",
        "inferred_tool_schema": [{"type": "function", "function": {
            "name": "noop", "description": "No operation",
            "parameters": {"type": "object", "properties": {}},
        }}],
        "tool_dispatch": tool_dispatch or {},
    }


def test_session_dir_persists_before_tool_execution_and_event_handlers(tmp_path: Path) -> None:
    sink = Recorder()
    order: list[str] = []

    def tool(**kwargs):
        assert any(r["type"] == "tool_start" for r in sink.records)
        order.append("tool")
        return "done", {}

    session = init_session(
        _schema({"write": {"python_function": tool}}),
        session_dir=tmp_path / "one-session",
        durable_sink=sink,
        on_event=lambda _kind, _data: order.append("event"),
    )
    try:
        _handle_tool_call("write", {"path": "x"}, session)
        assert order == ["event", "tool", "event"]
        assert [r["type"] for r in sink.records[:2]] == ["message", "tool_start"]
        journal = (tmp_path / "one-session" / "journal.jsonl")
        assert journal.exists()
        records = [json.loads(line) for line in journal.read_text().splitlines()]
        assert any(r["type"] == "tool_end" and r["result"] == "done" for r in records)
    finally:
        session["_journal"].close()


def test_request_is_committed_before_submit(tmp_path: Path) -> None:
    sink = Recorder()
    submitted: list[dict] = []

    class Completions:
        def create(self, **kwargs):
            assert any(r["type"] == "model_request" for r in sink.records)
            submitted.append({"last_content": kwargs["messages"][-1]["content"]})
            msg = SimpleNamespace(content="ok", tool_calls=None)
            usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1,
                                    total_tokens=2, cached_tokens=0,
                                    cache_creation_tokens=0,
                                    has_cache_proof=True)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage,
                                   provider=None, reasoning=None)

    client = SimpleNamespace(
        base_url=SimpleNamespace(host=""),
        chat=SimpleNamespace(completions=Completions()),
    )
    result = run_task(_schema(), "hello", client=client,
                      session_dir=tmp_path / "request-session", durable_sink=sink)
    assert result.final_reply == "ok"
    assert submitted[0]["last_content"] == "hello"
    assert (tmp_path / "request-session" / "messages.json").exists()
