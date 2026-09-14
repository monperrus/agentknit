"""A turn interrupted between tool_calls and their results must not poison the session.

Ctrl-C (or a crash) during a tool call leaves an assistant message whose
tool_calls were never answered.  Providers reject that transcript outright
("No tool output found for function call …"), so without an in-process repair
the live session is dead: every later turn fails the same way, and the user's
next message is lost with it.  Resume already repaired this when loading from
disk; ``_run_turn`` now does it for the session in memory too.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from agentknit._core import _run_turn


class _RecordingCompletions:
    def __init__(self) -> None:
        self.sent: list[list[dict]] = []

    def create(self, **kwargs):
        self.sent.append(kwargs["messages"])
        raise RuntimeError("stop here — we only care about what was sent")


class _RecordingClient:
    def __init__(self) -> None:
        self.base_url = SimpleNamespace(host="api.example.test")
        self.completions = _RecordingCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


def _session(tmp_path, messages: list[dict]) -> dict:
    return {
        "messages": messages,
        "tools": [],
        "structured": True,
        "tool_dispatch": {},
        "session_id": "interrupt-test",
        "cache_key": "interrupt-test",
        "endpoint": "https://api.example.test/v1",
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
        "on_event": lambda event_type, data: None,
        "_event_handlers": {},
    }


def _interrupted_history() -> list[dict]:
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "do the thing"},
        {"role": "assistant", "tool_calls": [{
            "id": "call_abc", "type": "function",
            "function": {"name": "long_task", "arguments": "{}"},
        }]},
        # ← Ctrl-C landed here: no tool message for call_abc
    ]


def test_dangling_tool_call_is_answered_before_the_request(tmp_path) -> None:
    session = _session(tmp_path, _interrupted_history())
    client = _RecordingClient()

    _run_turn(client, "model", session, "what happened?")

    sent = client.completions.sent[0]
    results = [m for m in sent if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in results] == ["call_abc"]
    assert "interrupted" in results[0]["content"]
    # the placeholder precedes the new user message, keeping the block contiguous
    assert sent.index(results[0]) < sent.index(
        next(m for m in sent if m.get("content") == "what happened?"))


def test_repair_is_applied_to_the_live_session(tmp_path) -> None:
    messages = _interrupted_history()
    session = _session(tmp_path, messages)

    _run_turn(_RecordingClient(), "model", session, "what happened?")

    # repaired in place: the caller's list object is the session's list
    assert session["messages"] is messages
    assert any(m.get("tool_call_id") == "call_abc" for m in messages)


def test_repair_is_logged_once_and_not_repeated(tmp_path) -> None:
    session = _session(tmp_path, _interrupted_history())
    client = _RecordingClient()

    _run_turn(client, "model", session, "first")
    _run_turn(client, "model", session, "second")

    entries = [json.loads(line) for line in
               (tmp_path / "session.jsonl").read_text().splitlines() if line.strip()]
    repairs = [e for e in entries if e.get("type") == "repair_tool_call_pairing"]
    assert len(repairs) == 1 and repairs[0]["delta"] == 1


def test_healthy_history_is_left_alone(tmp_path) -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "do the thing"},
        {"role": "assistant", "tool_calls": [{
            "id": "call_abc", "type": "function",
            "function": {"name": "long_task", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "call_abc", "content": "done"},
    ]
    session = _session(tmp_path, messages)
    before = json.dumps(messages)

    _run_turn(_RecordingClient(), "model", session, "and then?")

    assert json.dumps(messages[:4]) == before
