"""Tests for injectable clients (issue #23).

``run_task``, ``run_repl`` and ``run_async_repl`` accept an optional
``client=`` so callers can pass a custom/wrapped client instead of the one
``create_client`` builds from the schema.
"""

from __future__ import annotations

import contextlib
import io

import agentknit._core as _core
from agentknit import load_specification, run_task, run_repl
from agentknit.openai_compat import (
    SubprocessOpenAI, _Message, _Choice, _Usage, _Response, _SubprocessChat,
)

MODEL = "test/model"
ENDPOINT = "https://api.test/v1"


class RecordingOpenAI(SubprocessOpenAI):
    """Stub client that records requests and returns a canned reply."""

    def __init__(self, reply: str = "ok") -> None:
        # Deliberately skip SubprocessOpenAI.__init__: no binary involved.
        from agentknit.openai_compat import _BaseURL
        self._binary_path = "stub"
        self.base_url = _BaseURL("")
        self.requests: list[dict] = []
        self._reply = reply
        self.chat = _SubprocessChat(self)

    def _complete(self, model: str, messages: list[dict],
                  **kwargs: dict) -> _Response:
        self.requests.append({"model": model, "messages": list(messages),
                              **kwargs})
        msg = _Message(role="assistant", content=self._reply, tool_calls=None)
        usage = _Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        return _Response(choices=[_Choice(msg)], usage=usage)


def _schema() -> dict:
    return load_specification(MODEL, ENDPOINT)


def _stub_completions(monkeypatch) -> None:
    """Route _SubprocessCompletions.create to the client's _complete."""
    from agentknit.openai_compat import _SubprocessCompletions

    def _create(self, *, model, messages, **kwargs):
        return self._client._complete(model, messages, **kwargs)

    monkeypatch.setattr(_SubprocessCompletions, "create", _create)


def test_run_task_accepts_injected_client(monkeypatch) -> None:
    """run_task uses the passed client and never calls create_client."""
    called = []
    monkeypatch.setattr(_core, "create_client",
                        lambda schema: called.append(schema))
    _stub_completions(monkeypatch)
    client = RecordingOpenAI(reply="injected")
    result = run_task(_schema(), "hello", client=client)
    assert not called
    assert client.requests, "injected client must receive the request"
    assert result.final_reply == "injected"


def test_run_task_creates_client_when_none(monkeypatch) -> None:
    """Without client=, run_task still builds one via create_client."""
    _stub_completions(monkeypatch)
    client = RecordingOpenAI()
    monkeypatch.setattr(_core, "create_client", lambda schema: client)
    result = run_task(_schema(), "hello")
    assert client.requests
    assert result.usage["total"] == 2


def test_repl_setup_accepts_injected_client(monkeypatch) -> None:
    """_repl_setup (used by run_repl/run_async_repl) honours client=."""
    called = []
    monkeypatch.setattr(_core, "create_client",
                        lambda schema: called.append(schema))
    client = RecordingOpenAI()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        got_client, session, model, _hist = _core._repl_setup(
            _schema(), client=client)
    assert not called
    assert got_client is client
    assert model == MODEL


def test_run_repl_uses_injected_client(monkeypatch) -> None:
    """End-to-end: run_repl drives the injected client for each turn."""
    _stub_completions(monkeypatch)
    monkeypatch.setattr(_core.select, "select", lambda *_: ([], [], []))
    client = RecordingOpenAI(reply="repl-reply")
    buf = io.StringIO()
    monkeypatch.setattr(_core.sys, "stdin", io.StringIO("hi\nexit\n"))
    with contextlib.redirect_stdout(buf):
        run_repl(_schema(), client=client)
    assert len(client.requests) == 1
    assert client.requests[0]["messages"][-1]["content"] == "hi"


def test_repl_continue_retries_without_adding_user_message(monkeypatch) -> None:
    """/c replays an interrupted transcript instead of appending "go"."""
    _stub_completions(monkeypatch)
    monkeypatch.setattr(_core.select, "select", lambda *_: ([], [], []))
    client = RecordingOpenAI(reply="repl-reply")
    buf = io.StringIO()
    monkeypatch.setattr(_core.sys, "stdin", io.StringIO("work\n/c\nexit\n"))
    with contextlib.redirect_stdout(buf):
        run_repl(_schema(), client=client)

    assert len(client.requests) == 2
    assert [m["content"] for m in client.requests[1]["messages"]
            if m["role"] == "user"] == ["work"]
