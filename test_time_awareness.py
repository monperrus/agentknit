"""Tests for model-facing time awareness in agentknit._core.

Time awareness injects *measured* wall-clock readings into the model's own
context: a timing line opening every turn (session elapsed · last tool ·
wall since the model's previous message) and an ISO-8601 start/end/duration
stamp on every tool result.  Every number comes from a real clock reading —
an estimated or fabricated duration is a bug.
"""

from __future__ import annotations

import contextlib
import datetime
import io
import json

import agentknit._core as _core
from agentknit import load_specification
from agentknit._core import fmt_duration
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
        usage = _Usage(prompt_tokens=1000, completion_tokens=10,
                       total_tokens=1010, has_cache_proof=True)
        return _Response(choices=[_Choice(self._script.pop(0))], usage=usage)


class FakeClock:
    """Monotone fake wall clock; every reading advances it by *tick*."""

    def __init__(self, start: float = 1_700_000_000.0, tick: float = 1.0) -> None:
        self.now = start
        self.tick = tick

    def __call__(self) -> float:
        value = self.now
        self.now += self.tick
        return value


def _patch_create(monkeypatch) -> None:
    def _create(self, *, model, messages, **kwargs):
        return self._client._complete(model, messages, **kwargs)

    monkeypatch.setattr(_SubprocessCompletions, "create", _create)


def _quiet(monkeypatch) -> None:
    monkeypatch.setattr(_core, "_default_event_handler", lambda *_a: None)


def _tool_call(call_id: str = "c1") -> _Message:
    return _Message(role="assistant", content=None,
                    tool_calls=[_FakeToolCall(call_id, "read_file",
                                              '{"path": "x"}')])


def _done(text: str = "done") -> _Message:
    return _Message(role="assistant", content=text, tool_calls=None)


def _run_scripted(monkeypatch, script, tasks=("do the thing",),
                  started_at: float | None = None, **init_kwargs):
    _patch_create(monkeypatch)
    _quiet(monkeypatch)
    schema = load_specification(MODEL, ENDPOINT)
    client = ScriptedOpenAI(script)
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        session = _core.init_session(schema, non_interactive=True, durable=False,
                                     **init_kwargs)
        if started_at is not None:
            # Anchor the session clock on the fake clock used by the test.
            session["time_awareness_started_at"] = started_at
        for task in tasks:
            _core.run_turn(client, MODEL, session, task)
    return session, client


def _all_content(messages: list[dict]) -> str:
    """Conversation content without the system prompt (which *describes*
    the markers and would match a search for them)."""
    return "\n".join(str(m.get("content") or "") for m in messages[1:])


def _user_contents(messages: list[dict]) -> list[str]:
    return [str(m["content"]) for m in messages if m.get("role") == "user"]


def test_fmt_duration_scales() -> None:
    """Sub-second readings keep milliseconds; longer ones stay readable."""
    assert fmt_duration(0.34) == "340ms"
    assert fmt_duration(0) == "0ms"
    assert fmt_duration(9.42) == "9.4s"
    assert fmt_duration(42.4) == "42s"
    assert fmt_duration(91) == "1m31s"
    assert fmt_duration(862) == "14m22s"
    assert fmt_duration(7500) == "2h05m"


def test_system_prompt_declares_time_awareness_by_default(monkeypatch) -> None:
    """Default (enabled): the system prompt explains the two injections and
    frames elapsed time as information, not pressure."""
    _quiet(monkeypatch)
    schema = load_specification(MODEL, ENDPOINT)
    session = _core.init_session(schema, durable=False)
    sys_msg = session["messages"][0]["content"]
    assert "<session_time>" in sys_msg and "<tool_time>" in sys_msg
    assert "information, not pressure" in sys_msg
    assert session["time_awareness_enabled"] is True
    assert session["time_awareness_tool_timestamps"] is True
    assert session["time_awareness_last_tool_ms"] is None
    assert session["time_awareness_last_reply_at"] is None


def test_disabled_removes_everything(monkeypatch) -> None:
    """Disabled: nothing in the system prompt, the prompt or tool results."""
    session, _ = _run_scripted(monkeypatch, [_tool_call(), _done()],
                               time_awareness_enabled=False)
    assert "<session_time>" not in session["messages"][0]["content"]
    assert "<session_time>" not in _all_content(session["messages"])
    assert "<tool_time" not in _all_content(session["messages"])
    assert _user_contents(session["messages"]) == ["do the thing"]


def test_turn_opens_with_the_three_numbers(monkeypatch) -> None:
    """The submitted prompt carries session elapsed, last tool duration and
    the wall time since the model's previous message.  On the first turn the
    last two have no reading yet and say so rather than inventing one."""
    clock = FakeClock(tick=0.0)
    monkeypatch.setattr(_core.time, "time", clock)
    session, client = _run_scripted(monkeypatch, [_done()],
                                    started_at=clock.now)
    prompt = _user_contents(session["messages"])[0]
    assert prompt.startswith("do the thing\n\n<session_time>")
    assert ("<session_time>session elapsed: 0ms · last tool: n/a · "
            "wall since your previous message: n/a</session_time>") in prompt
    # What the model actually received, not just what the history says.
    assert prompt == client.requests[0]["messages"][-1]["content"]


def test_second_turn_reports_real_tool_and_idle_times(monkeypatch) -> None:
    """Turn two reports the measured duration of the last tool call and how
    long the human took to answer — both from the clock, never estimated."""
    clock = FakeClock(tick=0.0)
    monkeypatch.setattr(_core.time, "time", clock)
    _patch_create(monkeypatch)
    _quiet(monkeypatch)
    schema = load_specification(MODEL, ENDPOINT)
    client = ScriptedOpenAI([_tool_call(), _done(), _done("again")])

    def read_file(path: str) -> str:  # noqa: ARG001 - stub tool
        clock.now += 0.34          # the tool takes 340ms
        return "contents"

    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        session = _core.init_session(schema, non_interactive=True, durable=False)
        session["time_awareness_started_at"] = clock.now
        session["tool_dispatch"]["read_file"] = {"python_function": read_file,
                                                 "param_map": {}}
        _core.run_turn(client, MODEL, session, "first")
        clock.now += 91           # the human is away for 91 seconds
        _core.run_turn(client, MODEL, session, "second")

    second = _user_contents(session["messages"])[1]
    assert ("session elapsed: 1m31s · last tool: 340ms · "
            "wall since your previous message: 1m31s") in second


def test_tool_results_carry_iso_timestamps(monkeypatch) -> None:
    """Every tool result is stamped with real ISO-8601 start/end/duration;
    the final answer is left alone."""
    session, _ = _run_scripted(monkeypatch, [_tool_call(), _done()])
    tool_results = [m for m in session["messages"] if m.get("role") == "tool"]
    assert len(tool_results) == 1
    content = tool_results[0]["content"]
    assert "<tool_time start=" in content and 'duration="' in content
    start = content.split('start="')[1].split('"')[0]
    end = content.split('end="')[1].split('"')[0]
    # Parseable ISO-8601, and time moves forward.
    assert (datetime.datetime.fromisoformat(end)
            >= datetime.datetime.fromisoformat(start))
    assert session["messages"][-1]["content"] == "done"
    assert session["time_awareness_last_tool_ms"] is not None


def test_tool_timestamps_can_be_disabled_alone(monkeypatch) -> None:
    """tool_timestamps=False drops the per-result stamp but keeps the
    per-turn timing line (and the measured last-tool duration feeding it)."""
    session, _ = _run_scripted(monkeypatch, [_tool_call(), _done()],
                               time_awareness_tool_timestamps=False)
    assert "<tool_time" not in _all_content(session["messages"])
    assert "<session_time>" in _user_contents(session["messages"])[0]
    assert session["time_awareness_last_tool_ms"] is not None


def test_malformed_tool_call_result_is_not_stamped(monkeypatch) -> None:
    """No tool ran, so there is no span to report — nothing is invented."""
    bad = _Message(role="assistant", content=None,
                   tool_calls=[_FakeToolCall("c1", "read_file", "not json")])
    session, _ = _run_scripted(monkeypatch, [bad, _done()])
    tool_results = [m for m in session["messages"] if m.get("role") == "tool"]
    assert "<tool_time" not in tool_results[0]["content"]


def test_tool_result_log_record_carries_duration(monkeypatch) -> None:
    """Journals keep the durations, so a session's lived distribution of
    "how long does this take here" can be replayed afterwards."""
    session, _ = _run_scripted(monkeypatch, [_tool_call(), _done()])
    records = [json.loads(line)
               for line in session["log_path"].read_text().splitlines()]
    tool_records = [r for r in records if r["type"] == "tool_result"]
    assert tool_records and isinstance(tool_records[0]["duration_ms"], int)
    assert datetime.datetime.fromisoformat(tool_records[0]["started_at"])
    assert datetime.datetime.fromisoformat(tool_records[0]["ended_at"])


def test_snapshot_metadata_records_knobs_and_clock(monkeypatch, tmp_path) -> None:
    """Snapshot metadata keeps the knobs and the session clock's anchor."""
    _quiet(monkeypatch)
    schema = load_specification(MODEL, ENDPOINT)
    session = _core.init_session(schema, durable=False, session_dir=tmp_path,
                                 time_awareness_tool_timestamps=False)
    session["messages"].append({"role": "user", "content": "hi"})
    _core._save_messages_snapshot(session)
    payload = json.loads((tmp_path / "messages.json").read_text())
    ta = payload["metadata"]["time_awareness"]
    assert ta["enabled"] is True
    assert ta["tool_timestamps"] is False
    assert datetime.datetime.fromisoformat(ta["started_at"])


def test_old_session_dict_backfilled_on_restore(monkeypatch) -> None:
    """Sessions snapshotted before time awareness get defaults on restore,
    with the clock anchored on the original session start — a session
    resumed later really is that old."""
    _quiet(monkeypatch)
    schema = load_specification(MODEL, ENDPOINT)
    session = _core.init_session(schema, durable=False)
    for k in [k for k in session if k.startswith("time_awareness")]:
        del session[k]
    session["session_start_ts"] = (
        datetime.datetime.now() - datetime.timedelta(hours=2)
    ).isoformat(timespec="seconds")
    restored = _core.init_session(schema, session=session, durable=False)
    assert restored["time_awareness_enabled"] is True
    assert restored["time_awareness_tool_timestamps"] is True
    elapsed = _core.time.time() - restored["time_awareness_started_at"]
    assert 7000 < elapsed < 7400            # ~2h, not 0
    line = _core._time_awareness_preamble(restored)
    assert line is not None and "session elapsed: 2h00m" in line
