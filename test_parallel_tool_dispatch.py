"""Opt-in parallel dispatch of several tool calls in one assistant message.

With ``session["parallel_tool_dispatch"]`` (wired from the schema) and more
than one call in the assistant message, ``_run_turn`` dispatches the calls in
a :class:`~concurrent.futures.ThreadPoolExecutor` but still appends the tool
messages from the main thread, in ``tool_calls`` order.  Default (flag off)
keeps the sequential loop.
"""

from __future__ import annotations

import contextlib
import io
import json
import threading

import agentknit._core as _core
from agentknit import Tool, build_tool_spec
from agentknit._journal import SessionJournal
from agentknit.openai_compat import (
    SubprocessOpenAI, _Message, _Choice, _Usage, _Response, _SubprocessChat,
)
from agentknit.openai_compat import _SubprocessCompletions

MODEL = "test/model"
ENDPOINT = "https://api.test/v1"


class _FakeToolFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: str) -> None:
        self.id = call_id
        self.type = "function"
        self.custom_input = None
        self.function = _FakeToolFunction(name, arguments)


class ScriptedOpenAI(SubprocessOpenAI):
    """Stub client returning a scripted sequence of assistant messages."""

    def __init__(self, script: list[_Message]) -> None:
        from agentknit.openai_compat import _BaseURL
        self._binary_path = "stub"
        self.base_url = _BaseURL("")
        self.requests: list[dict] = []
        self._script = list(script)
        self.chat = _SubprocessChat(self)

    def _complete(self, model: str, messages: list[dict],
                  **kwargs: dict) -> _Response:
        self.requests.append({"model": model, "messages": list(messages),
                              **kwargs})
        msg = self._script.pop(0)
        usage = _Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15,
                       has_cache_proof=True)
        return _Response(choices=[_Choice(msg)], usage=usage)


def _patch_create(monkeypatch) -> None:
    def _create(self, *, model, messages, **kwargs):
        return self._client._complete(model, messages, **kwargs)

    monkeypatch.setattr(_SubprocessCompletions, "create", _create)


def _quiet(monkeypatch) -> None:
    monkeypatch.setattr(_core, "_default_event_handler", lambda *_a: None)


def _schema(tools: list[Tool], **flags) -> dict:
    specs, dispatch = build_tool_spec(tools)
    # build_tool_spec records the callable by name; bind the callable itself
    # so the test tools need no TOOL_LIBRARY registration.
    for tool in tools:
        dispatch[tool.name]["python_function"] = tool.fn
    return {
        "model": MODEL,
        "endpoint": ENDPOINT,
        "tool_specs": specs,
        "tool_dispatch": dispatch,
        "behaviour": {"call_delivery_mode": "structured_tool_calls"},
        **flags,
    }


def _run_scripted(monkeypatch, schema: dict, script: list[_Message]) -> dict:
    _patch_create(monkeypatch)
    _quiet(monkeypatch)
    client = ScriptedOpenAI(script)
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        session = _core.init_session(schema, non_interactive=True, durable=False)
        _core.run_turn(client, MODEL, session, "run the tools")
    return session


def _tool_msgs(session: dict) -> list[dict]:
    return [m for m in session["messages"] if m.get("role") == "tool"]


def _call(call_id: str, name: str, arguments: str = "{}") -> _FakeToolCall:
    return _FakeToolCall(call_id, name, arguments)


def _tools_msg(*calls: _FakeToolCall) -> _Message:
    return _Message(role="assistant", content=None, tool_calls=list(calls))


def _done_msg() -> _Message:
    return _Message(role="assistant", content="done", tool_calls=None)


def test_parallel_calls_execute_concurrently(monkeypatch) -> None:
    """Two tools rendezvous on a Barrier(2): only concurrent dispatch passes."""
    barrier = threading.Barrier(2, timeout=10)

    def tool_a():
        barrier.wait()
        return "a", {"result": "a"}

    def tool_b():
        barrier.wait()
        return "b", {"result": "b"}

    schema = _schema([Tool("alpha", "A", tool_a), Tool("beta", "B", tool_b)],
                     parallel_tool_dispatch=True)
    session = _run_scripted(
        monkeypatch, schema,
        [_tools_msg(_call("c1", "alpha"), _call("c2", "beta")), _done_msg()])

    msgs = _tool_msgs(session)
    assert [m["tool_call_id"] for m in msgs] == ["c1", "c2"]
    assert msgs[0]["content"].startswith("a")
    assert msgs[1]["content"].startswith("b")
    assert session["messages"][-1]["content"] == "done"


def test_history_order_preserved(monkeypatch) -> None:
    """Results are appended in the assistant's tool_calls order, not by race."""
    def tool_index(index: int):
        return str(index), {"result": str(index)}

    schema = _schema([Tool("index_probe", "I", tool_index)],
                     parallel_tool_dispatch=True)
    calls = [_call(f"c{i}", "index_probe", json.dumps({"index": i}))
             for i in range(3)]
    session = _run_scripted(monkeypatch, schema, [_tools_msg(*calls), _done_msg()])

    assistant = [m for m in session["messages"]
                 if m.get("role") == "assistant" and m.get("tool_calls")]
    assert len(assistant) == 1
    assert [tc["id"] for tc in assistant[0]["tool_calls"]] == ["c0", "c1", "c2"]

    msgs = _tool_msgs(session)
    assert [m["tool_call_id"] for m in msgs] == ["c0", "c1", "c2"]
    assert [m["content"][0] for m in msgs] == ["0", "1", "2"]


def test_sequential_by_default(monkeypatch) -> None:
    """Without the flag, both calls run on the single calling thread."""
    seen: list[int] = []

    def tool_a():
        seen.append(threading.get_ident())
        return "a", {"result": "a"}

    def tool_b():
        seen.append(threading.get_ident())
        return "b", {"result": "b"}

    schema = _schema([Tool("alpha", "A", tool_a), Tool("beta", "B", tool_b)])
    session = _run_scripted(
        monkeypatch, schema,
        [_tools_msg(_call("c1", "alpha"), _call("c2", "beta")), _done_msg()])

    assert session["parallel_tool_dispatch"] is False
    assert len(seen) == 2
    # Sequential path: no executor, both calls on the calling thread.
    assert seen[0] == seen[1] == threading.get_ident()


def test_journal_concurrent_appends(tmp_path) -> None:
    """8 threads x 50 records: seqs strictly increasing, no lost lines."""
    journal = SessionJournal(tmp_path / "s_journal.jsonl")
    n_threads, per_thread = 8, 50

    def worker(n: int) -> None:
        for i in range(per_thread):
            journal.append({"type": "probe", "worker": n, "i": i})

    threads = [threading.Thread(target=worker, args=(n,))
               for n in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = (tmp_path / "s_journal.jsonl").read_text().splitlines()
    recs = [json.loads(ln) for ln in lines]
    assert len(recs) == n_threads * per_thread
    assert [r["seq"] for r in recs] == list(range(1, n_threads * per_thread + 1))
    assert sorted((r["worker"], r["i"]) for r in recs) == sorted(
        (n, i) for n in range(n_threads) for i in range(per_thread))
