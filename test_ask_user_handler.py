"""set_ask_user_handler routes ask_user / ask_user_question away from stdin,
and tool events carry the call_id that pairs a tool_call with its tool_result."""

from __future__ import annotations

import json

import pytest

from agentknit import set_ask_user_handler
from agentknit.tool_library import t_ask_user, t_ask_user_question


@pytest.fixture(autouse=True)
def _reset_handler():
    yield
    set_ask_user_handler(None)


def _no_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a, **k):
        raise AssertionError("stdin must not be read")
    monkeypatch.setattr("builtins.input", boom)


def test_ask_user_uses_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_stdin(monkeypatch)
    seen: list[tuple[str, list[str]]] = []
    set_ask_user_handler(lambda q, opts: seen.append((q, opts)) or "blue")
    text, _ = t_ask_user("colour?")
    assert json.loads(text) == {"answer": "blue"}
    assert seen == [("colour?", [])]


def test_ask_user_question_passes_options(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_stdin(monkeypatch)
    set_ask_user_handler(lambda q, opts: "2")
    text, meta = t_ask_user_question("pick", '["a", "b"]')
    assert json.loads(text) == {"answer": "b"}
    assert meta["options"] == ["a", "b"]


def test_handler_failure_is_a_tool_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_stdin(monkeypatch)

    def gone(q: str, opts: list[str]) -> str:
        raise TimeoutError("nobody there")

    set_ask_user_handler(gone)
    text, meta = t_ask_user("still there?")
    assert text.startswith("ERROR:") and "nobody there" in text
    assert meta.get("ok") is False


def test_tool_events_carry_call_id() -> None:
    from agentknit import execute_tool_call, init_session, load_specification, subscribe

    session = init_session(load_specification("test/model", "https://api.test/v1"),
                           durable=False)
    events: list[tuple[str, dict]] = []
    for et in ("tool_call", "tool_result"):
        subscribe(session, et, lambda t, d: events.append((t, d)))
    execute_tool_call(session, "list_dir", {"path": "."}, call_id="call_42")
    assert [(t, d["call_id"]) for t, d in events] == [("tool_call", "call_42"),
                                                      ("tool_result", "call_42")]
