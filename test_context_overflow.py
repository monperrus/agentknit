"""Tests for context-window-overflow recovery in agentknit._core.

When the provider rejects a request with HTTP 400/413 "token limit
exceeded" — which happens when the compaction trigger is misconfigured
above the true context window, or when tool results outgrow the window
between two usage measurements — the agent loop must compact the history
and retry instead of dying with a raw API error.
"""

from __future__ import annotations

import contextlib
import io

import agentknit._core as _core
from agentknit import ContextWindowExceededError, load_specification
from agentknit.openai_compat import (
    SubprocessOpenAI, _Message, _Choice, _Usage, _Response, _SubprocessChat,
)
from agentknit.openai_compat import _SubprocessCompletions

MODEL = "test/model"
ENDPOINT = "https://api.test/v1"


class _ToolFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, call_id: str, name: str, arguments: str) -> None:
        self.id = call_id
        self.type = "function"
        self.custom_input = None
        self.function = _ToolFunction(name, arguments)


class _OverflowThenOK(SubprocessOpenAI):
    """Stub client: rejects with context-window errors a fixed number of
    times, then returns a plain final answer."""

    def __init__(self, rejections: list[BaseException]) -> None:
        from agentknit.openai_compat import _BaseURL
        self._binary_path = "stub"
        self.base_url = _BaseURL("")
        self.requests: list[dict] = []
        self._rejections = list(rejections)
        self.chat = _SubprocessChat(self)

    def _complete(self, model: str, messages: list[dict],
                  **kwargs: dict) -> _Response:
        self.requests.append({"model": model, "messages": list(messages),
                              **kwargs})
        # Only reject genuine agent-loop requests; compaction pre/summary
        # calls are always served (their exceptions are swallowed internally,
        # which would silently consume queued rejections).
        last = (messages[-1].get("content") or "") if messages else ""
        is_compaction_call = ("Summarize the conversation above" in last
                              or "compaction is about to run" in last)
        if self._rejections and not is_compaction_call:
            raise self._rejections.pop(0)
        # Compaction calls: return a usable summary instead of a final answer.
        if (messages and messages[-1].get("role") == "user"
                and "Summarize the conversation above"
                in (messages[-1].get("content") or "")):
            usage = _Usage(prompt_tokens=50, completion_tokens=5,
                           total_tokens=55, has_cache_proof=True)
            return _Response(choices=[
                _Choice(_Message("assistant", "summary", None))], usage=usage)
        usage = _Usage(prompt_tokens=100, completion_tokens=5,
                       total_tokens=105, has_cache_proof=True)
        return _Response(choices=[_Choice(_Message("assistant", "done", None))],
                         usage=usage)


def _patch_create(monkeypatch) -> None:
    def _create(self, *, model, messages, **kwargs):
        return self._client._complete(model, messages, **kwargs)

    monkeypatch.setattr(_SubprocessCompletions, "create", _create)


def _quiet(monkeypatch) -> None:
    monkeypatch.setattr(_core, "_default_event_handler", lambda *_a: None)


def _seed_session(schema, **init_kwargs):
    session = _core.init_session(schema, non_interactive=True, durable=False,
                                 **init_kwargs)
    # Simulate an overgrown history that no longer fits the window.
    session["messages"].extend(
        {"role": "user" if i % 2 == 0 else "assistant",
         "content": f"turn {i} " + "x" * 200}
        for i in range(20)
    )
    return session


def _run(monkeypatch, client, session):
    _patch_create(monkeypatch)
    _quiet(monkeypatch)
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        result = _core.run_turn(client, MODEL, session, "continue the task")
    return result


def _kimi_style_overflow() -> RuntimeError:
    """The exact failure from the incident: plain RuntimeError, status only
    inside the message text."""
    return RuntimeError(
        '[HTTP 400] {"error":{"message":"Invalid request: Your request '
        'exceeded model token limit: 1048576 (requested: 1680796)",'
        '"type":"invalid_request_error"}}'
    )


def test_context_overflow_compacts_and_retries(monkeypatch) -> None:
    """A token-limit 400 mid-turn triggers compaction + immediate retry, and
    the turn completes instead of aborting."""
    client = _OverflowThenOK([_kimi_style_overflow()])
    session = _seed_session(load_specification(MODEL, ENDPOINT))
    n_before = len(session["messages"])
    _run(monkeypatch, client, session)
    # The retried call saw a compacted (smaller) history.
    assert len(client.requests) >= 2
    assert len(client.requests[1]["messages"]) < n_before
    # The loop reached a final answer after the retry.
    assert session["messages"][-1]["role"] == "assistant"
    assert session["messages"][-1]["content"] == "done"


def test_context_overflow_with_typed_exception(monkeypatch) -> None:
    """ContextWindowExceededError instances are handled the same way."""
    client = _OverflowThenOK([
        ContextWindowExceededError(
            "request exceeded model token limit", status_code=400)
    ])
    session = _seed_session(load_specification(MODEL, ENDPOINT))
    _run(monkeypatch, client, session)
    assert session["messages"][-1]["content"] == "done"


def test_context_overflow_aborts_after_repeated_rejections(monkeypatch) -> None:
    """Two consecutive rejections (history still too big after compaction)
    abort the turn rather than looping forever."""
    client = _OverflowThenOK([_kimi_style_overflow(), _kimi_style_overflow(),
                              _kimi_style_overflow(), _kimi_style_overflow()])
    session = _seed_session(load_specification(MODEL, ENDPOINT))
    # Run 1: two consecutive rejections exhaust the retry budget → abort.
    # (Compaction requests in between are served normally, so each
    # compaction+retry pair consumes exactly one rejection.)
    _run(monkeypatch, client, session)
    assert session["messages"][-1]["content"] != "done"


def test_context_overflow_resets_hysteresis(monkeypatch) -> None:
    """The overflow path resets compaction hysteresis so the threshold check
    can fire again immediately after the emergency compaction."""
    client = _OverflowThenOK([_kimi_style_overflow()])
    session = _seed_session(load_specification(MODEL, ENDPOINT))
    session["compaction_last_prompt_tokens"] = 999_999
    _run(monkeypatch, client, session)
    assert session.get("compaction_last_prompt_tokens", 0) == 0
    assert session["messages"][-1]["content"] == "done"


def test_non_context_400_still_aborts(monkeypatch) -> None:
    """A 400 that is not about token limits must not trigger compaction."""
    client = _OverflowThenOK([RuntimeError('[HTTP 400] {"error":{"message":'
                                           '"invalid api key"}}')])
    session = _seed_session(load_specification(MODEL, ENDPOINT))
    n_before = len(session["messages"]) + 1  # run_turn appends the task
    _run(monkeypatch, client, session)
    # No compaction happened: history untouched, turn aborted.
    assert len(session["messages"]) == n_before
