"""Tests for the six situational-awareness senses in agentknit._core.

Every sense (user, system, git, time, token, change) must be on by default,
reported on the console at session start (``👤 user awareness: ✅``) and
switchable off individually — schema key, ``init_session`` kwarg or CLI flag.
The change sense must also inject an honest-provenance note when a watched
file moves outside the session's tool calls, and stay silent otherwise.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import time

import pytest

import agentknit._core as _core
from agentknit import load_specification
from agentknit._core import AWARENESS_TYPES, parse_args
from agentknit.openai_compat import (
    SubprocessOpenAI, _Message, _Choice, _Usage, _Response, _SubprocessChat,
)
from agentknit.openai_compat import _SubprocessCompletions

MODEL = "test/model"
ENDPOINT = "https://api.test/v1"

_ANSI = __import__("re").compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


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


def _patch_create(monkeypatch) -> None:
    def _create(self, *, model, messages, **kwargs):
        return self._client._complete(model, messages, **kwargs)

    monkeypatch.setattr(_SubprocessCompletions, "create", _create)


def _quiet(monkeypatch) -> None:
    monkeypatch.setattr(_core, "_default_event_handler", lambda *_a: None)


def _capture_events(monkeypatch) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(_core, "_default_event_handler",
                        lambda et, data: events.append((et, dict(data))))
    return events


def _tool_call(call_id: str = "c1", args: str = '{"path": "x"}') -> _Message:
    return _Message(role="assistant", content=None,
                    tool_calls=[_FakeToolCall(call_id, "read_file", args)])


def _done(text: str = "done") -> _Message:
    return _Message(role="assistant", content=text, tool_calls=None)


def _user_contents(messages: list[dict]) -> list[str]:
    return [str(m["content"]) for m in messages if m.get("role") == "user"]


def _run_scripted(monkeypatch, script, tasks=("do the thing",), **init_kwargs):
    _patch_create(monkeypatch)
    _quiet(monkeypatch)
    schema = load_specification(MODEL, ENDPOINT)
    client = ScriptedOpenAI(script)
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        session = _core.init_session(schema, non_interactive=True, durable=False,
                                     **init_kwargs)
        for task in tasks:
            _core.run_turn(client, MODEL, session, task)
    return session, client


# ── startup checklist ──────────────────────────────────────────────────────

def test_checklist_reports_all_six_senses(monkeypatch) -> None:
    """Defaults: one console line per sense, each saying `✅`."""
    events = _capture_events(monkeypatch)
    schema = load_specification(MODEL, ENDPOINT)
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        _core.init_session(schema, durable=False)
    checks = [e for e in events if e[0] == "awareness_checklist"]
    assert len(checks) == 1
    fmt = checks[0][1]["fmt"]
    lines = [_strip_ansi(line) for line in fmt.splitlines()]
    assert lines == [
        f"{_core.AWARENESS_EMOJI[t]} {t} awareness: ✅"
        for t in AWARENESS_TYPES]


def test_checklist_says_off_for_disabled_sense(monkeypatch) -> None:
    """A disabled sense reports `❌` on its line, the others stay `✅`."""
    events = _capture_events(monkeypatch)
    schema = load_specification(MODEL, ENDPOINT)
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        _core.init_session(schema, durable=False, change_awareness_enabled=False)
    fmt = dict([e for e in events if e[0] == "awareness_checklist"][0][1])["fmt"]
    lines = [_strip_ansi(line) for line in fmt.splitlines()]
    assert lines[AWARENESS_TYPES.index("change")] == "🔄 change awareness: ❌"
    assert sum(line.endswith(": ❌") for line in lines) == 1


# ── per-sense gating ─────────────────────────────────────────────────────────

def test_user_awareness_gates_claude_md(monkeypatch) -> None:
    """User sense off: ~/.claude/CLAUDE.md is not in the system prompt."""
    _quiet(monkeypatch)
    home = os.environ["HOME"]
    claude = __import__("pathlib").Path(home) / ".claude" / "CLAUDE.md"
    claude.parent.mkdir(parents=True, exist_ok=True)
    claude.write_text("HOSTILE USER DOSSIER")
    schema = load_specification(MODEL, ENDPOINT)
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        on = _core.init_session(schema, durable=False)
        off = _core.init_session(schema, durable=False,
                                 user_awareness_enabled=False)
    assert "HOSTILE USER DOSSIER" in on["messages"][0]["content"]
    assert "HOSTILE USER DOSSIER" not in off["messages"][0]["content"]


def test_system_awareness_gates_environment_lines(monkeypatch) -> None:
    _quiet(monkeypatch)
    schema = load_specification(MODEL, ENDPOINT)
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        off = _core.init_session(schema, durable=False,
                                 system_awareness_enabled=False)
    sys_msg = off["messages"][0]["content"]
    assert "## Environment" in sys_msg          # header survives
    assert "Working directory:" not in sys_msg
    assert "OS:" not in sys_msg
    assert "Scratchpad" not in sys_msg
    # Attribution stays regardless of the switch.
    assert "Attribution" in sys_msg


def test_git_awareness_gates_git_block(monkeypatch) -> None:
    _quiet(monkeypatch)
    schema = load_specification(MODEL, ENDPOINT)
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        on = _core.init_session(schema, durable=False)
        off = _core.init_session(schema, durable=False,
                                 git_awareness_enabled=False)
    on_msg, off_msg = on["messages"][0]["content"], off["messages"][0]["content"]
    if "Git:" in on_msg:                        # inside a repo (the repo under test is)
        assert "Git:" not in off_msg
    assert "Working directory:" in off_msg       # the rest of the block stays


def test_schema_key_disables_a_sense(monkeypatch) -> None:
    """The schema carries the same switches as the kwargs."""
    _quiet(monkeypatch)
    schema = load_specification(MODEL, ENDPOINT)
    schema["change_awareness_enabled"] = False
    schema["user_awareness_enabled"] = False
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        session = _core.init_session(schema, durable=False)
    assert session["change_awareness_enabled"] is False
    assert session["user_awareness_enabled"] is False


def test_change_awareness_system_prompt_declares_the_note(monkeypatch) -> None:
    _quiet(monkeypatch)
    schema = load_specification(MODEL, ENDPOINT)
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        session = _core.init_session(schema, durable=False)
    assert "<ground_moved>" in session["messages"][0]["content"]
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        off = _core.init_session(schema, durable=False,
                                 change_awareness_enabled=False)
    assert "<ground_moved>" not in off["messages"][0]["content"]


# ── change awareness ─────────────────────────────────────────────────────────

@pytest.fixture
def outside_repo(monkeypatch, tmp_path):
    """Run from a directory outside any git repository."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_external_change_is_reported_with_honest_provenance(
        monkeypatch, outside_repo) -> None:
    """A file the session read, then modified externally, is reported on the
    next turn — without inventing who wrote it."""
    target = outside_repo / "watched.txt"
    target.write_text("v1")

    def read_file(path: str) -> str:  # noqa: ARG001 - stub tool
        return "contents"

    _patch_create(monkeypatch)
    _quiet(monkeypatch)
    schema = load_specification(MODEL, ENDPOINT)
    client = ScriptedOpenAI([
        _tool_call(args=json.dumps({"path": str(target)})), _done(), _done("again")])
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        session = _core.init_session(schema, non_interactive=True, durable=False)
        session["tool_dispatch"]["read_file"] = {"python_function": read_file,
                                                 "param_map": {}}
        _core.run_turn(client, MODEL, session, "first")
        # The watched file changes outside any tool call of this session.
        target.write_text("v2")
        os.utime(target, ns=(time.time_ns() + 10_000_000_000, 0))
        _core.run_turn(client, MODEL, session, "second")

    second = _user_contents(session["messages"])[1]
    assert "<ground_moved>" in second
    assert "watched.txt" in second
    assert "cannot be determined" in second     # honest provenance


def test_own_tool_writes_are_not_reported(monkeypatch, outside_repo) -> None:
    """What the session's tools wrote is the session's own footprint: the
    next turn must not flag it as external change."""
    target = outside_repo / "mine.txt"

    def write_file(path: str, content: str) -> str:  # noqa: ARG001 - stub tool
        target.write_text(content)
        return "wrote"

    _patch_create(monkeypatch)
    _quiet(monkeypatch)
    schema = load_specification(MODEL, ENDPOINT)
    client = ScriptedOpenAI([
        _Message(role="assistant", content=None,
                 tool_calls=[_FakeToolCall("c1", "write_file",
                                           json.dumps({"path": str(target),
                                                       "content": "hello"}))]),
        _done(), _done("again")])
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        session = _core.init_session(schema, non_interactive=True, durable=False)
        session["tool_dispatch"]["write_file"] = {"python_function": write_file,
                                                  "param_map": {}}
        _core.run_turn(client, MODEL, session, "first")
        _core.run_turn(client, MODEL, session, "second")

    second = _user_contents(session["messages"])[1]
    assert "<ground_moved>" not in second


def test_change_awareness_off_means_no_note(monkeypatch, outside_repo) -> None:
    target = outside_repo / "silent.txt"
    target.write_text("v1")

    def read_file(path: str) -> str:  # noqa: ARG001 - stub tool
        return "contents"

    _patch_create(monkeypatch)
    _quiet(monkeypatch)
    schema = load_specification(MODEL, ENDPOINT)
    client = ScriptedOpenAI([
        _tool_call(args=json.dumps({"path": str(target)})), _done(), _done("again")])
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        session = _core.init_session(schema, non_interactive=True, durable=False,
                                     change_awareness_enabled=False)
        session["tool_dispatch"]["read_file"] = {"python_function": read_file,
                                                 "param_map": {}}
        _core.run_turn(client, MODEL, session, "first")
        target.write_text("v2")
        os.utime(target, ns=(time.time_ns() + 10_000_000_000, 0))
        _core.run_turn(client, MODEL, session, "second")

    assert "<ground_moved>" not in _user_contents(session["messages"])[1]


def test_untouched_ground_means_no_note(monkeypatch, outside_repo) -> None:
    """Nothing moved between turns → no note, the prompt stays clean."""
    session, _ = _run_scripted(monkeypatch, [_done(), _done("again")],
                               tasks=("first", "second"))
    contents = _user_contents(session["messages"])
    assert contents[0] == "first" or contents[0].startswith("first")
    assert "<ground_moved>" not in contents[1]


# ── CLI wiring ───────────────────────────────────────────────────────────────

def test_cli_flags_disable_each_sense() -> None:
    args = parse_args([MODEL, "task",
                       "--no-user-awareness", "--no-system-awareness",
                       "--no-git-awareness", "--no-time-awareness",
                       "--no-token-awareness", "--no-change-awareness"])
    for t in AWARENESS_TYPES:
        assert getattr(args, f"{t}_awareness_enabled") is False


def test_cli_flags_default_to_none() -> None:
    args = parse_args([MODEL, "task"])
    for t in AWARENESS_TYPES:
        assert getattr(args, f"{t}_awareness_enabled") is None


# ── snapshot + restore ───────────────────────────────────────────────────────

def test_snapshot_metadata_records_the_switches(monkeypatch, tmp_path) -> None:
    _quiet(monkeypatch)
    schema = load_specification(MODEL, ENDPOINT)
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        session = _core.init_session(schema, durable=False,
                                     session_dir=tmp_path, change_awareness_enabled=False)
        session["messages"].append({"role": "user", "content": "hi"})
        _core._save_messages_snapshot(session)
    meta = json.loads((tmp_path / "messages.json").read_text())["metadata"]
    assert meta["awareness"] == {t: t != "change" for t in AWARENESS_TYPES}


def test_restore_backfills_old_snapshots(monkeypatch) -> None:
    """A session dict from before the switches restores with them all on."""
    _quiet(monkeypatch)
    schema = load_specification(MODEL, ENDPOINT)
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        session = _core.init_session(schema, durable=False)
    legacy = {k: v for k, v in session.items()
              if not k.endswith("_awareness_enabled")
              and k not in ("_awareness_file_watch", "_awareness_git_digest")}
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        restored = _core.init_session(schema, session=legacy)  # type: ignore[arg-type]
    for t in AWARENESS_TYPES:
        assert restored[f"{t}_awareness_enabled"] is True
