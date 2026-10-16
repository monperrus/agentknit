"""A provider that reuses a tool-call id must not make the history invalid.

Kimi names calls ``<tool>:<n>`` and reused ``exec_shell:7`` turns later in a live
session; every following request then failed at every backend ("tool call id
exec_shell:7 is duplicated", "`tool_use` ids must be unique"). The framework renames
a reused id before the call enters the history, and pairs the result with the new id.
"""

from __future__ import annotations

from agentknit import Tool
from agentknit._core import _unique_tool_call_ids
from test_parallel_tool_dispatch import (
    _call, _done_msg, _run_scripted, _schema, _tool_msgs, _tools_msg,
)


def _ids(session: dict) -> list[str]:
    return [tc["id"] for m in session["messages"] if m.get("role") == "assistant"
            for tc in (m.get("tool_calls") or [])]


def test_id_reused_on_a_later_step_is_renamed(monkeypatch) -> None:
    schema = _schema([Tool("echo", "E", lambda: ("ok", {"result": "ok"}))])
    session = _run_scripted(monkeypatch, schema, [
        _tools_msg(_call("exec_shell:7", "echo")),
        _tools_msg(_call("exec_shell:7", "echo")),
        _done_msg(),
    ])
    assert _ids(session) == ["exec_shell:7", "exec_shell:7-2"]
    assert [m["tool_call_id"] for m in _tool_msgs(session)] == _ids(session)


def test_unique_ids_are_left_alone(monkeypatch) -> None:
    schema = _schema([Tool("echo", "E", lambda: ("ok", {"result": "ok"}))])
    session = _run_scripted(monkeypatch, schema, [
        _tools_msg(_call("a", "echo"), _call("b", "echo")), _done_msg(),
    ])
    assert _ids(session) == ["a", "b"]


def test_duplicate_within_one_batch_and_empty_ids() -> None:
    history = [{"role": "assistant", "tool_calls": [{"id": "x"}, {"id": "x-2"}]}]
    calls = [_call("x", "t"), _call("x", "t"), _call("", "t"), _call("", "t")]
    _unique_tool_call_ids(history, calls)
    assert [c.id for c in calls] == ["x-3", "x-4", "call", "call-2"]
