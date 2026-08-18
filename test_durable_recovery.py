"""Integration tests for durable recovery: journaling inside the agent loop.

Covers the failure model the journal exists for: a crash mid-turn loses the
snapshot's view of the turn, but the journal replays it — completed tool
results are re-injected and in-flight tool calls are flagged as unknown.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import agentknit._core as core
from agentknit._core import (
    _handle_tool_call,
    _journal_path,
    init_session,
)
from agentknit._journal import SessionJournal, replay_journal


# ── fixtures ──────────────────────────────────────────────────────────────────

def _schema(tmp_path: Path, **extra) -> dict:
    return {
        "model": "test/model",
        "endpoint": "https://api.example.com/v1",
        "inferred_tool_schema": [],
        "tool_dispatch": {
            "read_file": {"python_function": "t_read"},
        },
        **extra,
    }


def _session(tmp_path: Path, monkeypatch, **schema_extra) -> dict:
    monkeypatch.setattr(core, "LOG_BASE", tmp_path)
    schema = _schema(tmp_path, **schema_extra)
    session = init_session(schema)
    return session


class _Recorder:
    """Tool that records invocations and can simulate a crash."""

    def __init__(self, result: str = "ok", crash: bool = False) -> None:
        self.calls: list[dict] = []
        self.result = result
        self.crash = crash
        self.__name__ = "t_read_file_recorder"

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.crash:
            raise KeyboardInterrupt  # simulates kill -9 mid-tool
        return self.result, {"result": "ok"}


# ── _handle_tool_call journals both sides of the call ─────────────────────────

def test_handle_tool_call_writes_tool_start_and_end(tmp_path, monkeypatch):
    session = _session(tmp_path, monkeypatch)
    session["tool_dispatch"]["read_file"] = {
        "python_function": _Recorder("file body")}
    monkeypatch.setattr(core._tool_module, "_tool_context",
                        MagicMock(session_id="s", tool_dispatch={}))

    _handle_tool_call("read_file", {"path": "a.txt"}, session)

    state = replay_journal(_journal_path("test/model", session["session_id"]))
    assert state.pending_tool_calls == []
    # tool_start + tool_end both present → replay knows the outcome.
    lines = _journal_path("test/model", session["session_id"]).read_text().splitlines()
    types = [json.loads(ln)["type"] for ln in lines]
    assert types == ["tool_start", "tool_end"]


def test_handle_tool_call_crash_mid_tool_leaves_pending(tmp_path, monkeypatch):
    session = _session(tmp_path, monkeypatch)
    session["tool_dispatch"]["read_file"] = {
        "python_function": _Recorder(crash=True)}
    monkeypatch.setattr(core._tool_module, "_tool_context",
                        MagicMock(session_id="s", tool_dispatch={}))

    try:
        _handle_tool_call("read_file", {"path": "a.txt"}, session)
    except KeyboardInterrupt:
        pass

    state = replay_journal(_journal_path("test/model", session["session_id"]))
    # tool_start persisted, no tool_end → recovery must treat side effects
    # as unknown rather than assuming the tool never ran.
    assert len(state.pending_tool_calls) == 1
    assert state.pending_tool_calls[0].name == "read_file"
    assert state.pending_tool_calls[0].args == {"path": "a.txt"}


# ── durability flag plumbing ──────────────────────────────────────────────────

def test_durable_default_on(tmp_path, monkeypatch):
    session = _session(tmp_path, monkeypatch)
    assert isinstance(session["_journal"], SessionJournal)
    assert session["durable"] is True


def test_durable_schema_opt_out(tmp_path, monkeypatch):
    session = _session(tmp_path, monkeypatch, durable=False)
    assert session["_journal"] is None
    assert session["durable"] is False


def test_durable_param_overrides_schema(tmp_path, monkeypatch):
    session = _session(tmp_path, monkeypatch, durable=True)
    assert isinstance(session["_journal"], SessionJournal)


# ── recovery on resume ────────────────────────────────────────────────────────

def test_resume_recovers_journal_state_over_stale_snapshot(tmp_path, monkeypatch):
    """Snapshot is a turn behind the journal → journal wins."""
    monkeypatch.setattr(core, "LOG_BASE", tmp_path)
    schema = _schema(tmp_path)

    # First process: run one turn, then crash *before* the turn-boundary
    # snapshot is written (the exact window durable recovery targets).
    session = init_session(schema)
    sid = session["session_id"]
    jpath = _journal_path("test/model", sid)
    j = SessionJournal(jpath)
    j.turn_start("deploy")
    j.message({"role": "user", "content": "deploy"})
    j.message({"role": "assistant", "tool_calls": [
        {"id": "c1", "type": "function",
         "function": {"name": "t_run", "arguments": '{"command": "deploy.sh"}'}}]})
    j.tool_start("c1", "t_run", {"command": "deploy.sh"})
    j.tool_end("c1", "deploy ok", name="t_run")  # side effects happened

    # No snapshot exists at all (crash before turn end).

    events: list[tuple[str, dict]] = []
    session2 = init_session(schema, resumed_from=sid,
                            on_event=lambda et, d: events.append((et, d)))

    kinds = [e for e, _ in events]
    assert "journal_recovered" in kinds
    contents = [m.get("content") for m in session2["messages"]]
    assert "deploy" in contents
    # The tool ran and finished — its result is re-injected, never re-run.
    notes = [c for c in contents if c and "RECOVERY NOTE" in c]
    assert any("deploy ok" in n and "NOT re-run" in n for n in notes)


def test_resume_flags_pending_tool_calls(tmp_path, monkeypatch):
    """Crash mid-tool → resume injects a verify-state note for the model."""
    monkeypatch.setattr(core, "LOG_BASE", tmp_path)
    schema = _schema(tmp_path)

    session = init_session(schema)
    sid = session["session_id"]
    j = SessionJournal(_journal_path("test/model", sid))
    j.turn_start("deploy")
    j.message({"role": "user", "content": "deploy"})
    j.tool_start("c1", "t_run", {"command": "deploy.sh"})
    # crash: no tool_end

    session2 = init_session(schema, resumed_from=sid)
    last = session2["messages"][-1]
    assert last["role"] == "user"
    assert "UNKNOWN" in last["content"]
    assert "t_run" in last["content"]
    assert "verify" in last["content"].lower()


def test_resume_keeps_structured_tool_calls_by_default(tmp_path, monkeypatch):
    """Default resume must not flatten tool calls into imitable prose (issue #25).

    Feeding the model a transcript where every prior tool call/result is
    plain assistant text — while tools are still declared for the request —
    teaches it to write tool calls as text and fabricate the results itself.
    """
    monkeypatch.setattr(core, "LOG_BASE", tmp_path)
    schema = _schema(tmp_path)

    session = init_session(schema)
    sid = session["session_id"]
    j = SessionJournal(_journal_path("test/model", sid))
    j.turn_start("deploy")
    j.message({"role": "user", "content": "deploy"})
    j.message({"role": "assistant", "tool_calls": [
        {"id": "c1", "type": "function",
         "function": {"name": "t_run", "arguments": '{"command": "deploy.sh"}'}}]})
    j.tool_start("c1", "t_run", {"command": "deploy.sh"})
    j.tool_end("c1", "deploy ok", name="t_run")

    session2 = init_session(schema, resumed_from=sid)

    assert any("tool_calls" in m for m in session2["messages"])
    for m in session2["messages"]:
        content = m.get("content") or ""
        assert "[Tool call:" not in content
        assert "[Tool result:" not in content
        assert "prior tool use:" not in content


def test_resume_flattens_tool_calls_when_provider_opts_in(tmp_path, monkeypatch):
    """behaviour.resume_rejects_stale_tool_call_ids=True still flattens, safely."""
    monkeypatch.setattr(core, "LOG_BASE", tmp_path)
    schema = _schema(
        tmp_path,
        behaviour={"resume_rejects_stale_tool_call_ids": True},
    )

    session = init_session(schema)
    sid = session["session_id"]
    j = SessionJournal(_journal_path("test/model", sid))
    j.turn_start("deploy")
    j.message({"role": "user", "content": "deploy"})
    j.message({"role": "assistant", "tool_calls": [
        {"id": "c1", "type": "function",
         "function": {"name": "t_run", "arguments": '{"command": "deploy.sh"}'}}]})
    j.tool_start("c1", "t_run", {"command": "deploy.sh"})
    j.tool_end("c1", "deploy ok", name="t_run")

    session2 = init_session(schema, resumed_from=sid)

    assert not any("tool_calls" in m for m in session2["messages"])
    assistant_contents = [m.get("content") or "" for m in session2["messages"]
                          if m.get("role") == "assistant"]
    assert any("prior tool use: t_run" in c for c in assistant_contents)
    assert not any("[Tool call:" in c or "[Tool result:" in c
                  for c in assistant_contents)
    # The raw tool result must never appear inside assistant-authored text —
    # that's exactly what teaches the model to fabricate results (issue #25).
    assert not any("deploy ok" in c for c in assistant_contents)


def test_resume_without_journal_falls_back_to_snapshot(tmp_path, monkeypatch):
    """No journal (old session or durable=False) → snapshot resume, unchanged."""
    monkeypatch.setattr(core, "LOG_BASE", tmp_path)
    schema = _schema(tmp_path)

    session = init_session(schema, durable=False)
    sid = session["session_id"]
    session["messages"].append({"role": "user", "content": "from snapshot"})
    core._save_messages_snapshot(session)

    session2 = init_session(schema, resumed_from=sid)
    contents = [m.get("content") for m in session2["messages"]
                if m.get("role") != "system"]
    assert any("from snapshot" in c for c in contents if c)
