"""Tests for bare mode (``init_session(bare=True)`` / ``--bare``).

Bare mode sends no agentknit prompting and no awareness at all: the model
sees only what the caller wrote — the system prompt supplement and the task —
plus raw tool results.
"""

from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path

import agentknit._core as _core
from agentknit import load_specification
from agentknit._core import AWARENESS_TYPES, parse_args
from test_situational_awareness import (
    ENDPOINT, MODEL, _capture_events, _done, _run_scripted, _tool_call,
)


def _plant_hostile_context(tmp_path: Path, monkeypatch) -> None:
    claude = Path(os.environ["HOME"]) / ".claude" / "CLAUDE.md"
    claude.parent.mkdir(parents=True, exist_ok=True)
    claude.write_text("HOSTILE USER DOSSIER")
    (tmp_path / "AGENTS.md").write_text("HOSTILE AGENTS NOTES")
    (tmp_path / "x").write_text("file body")
    monkeypatch.chdir(tmp_path)


def test_bare_sends_only_the_callers_words(monkeypatch, tmp_path) -> None:
    """Every message on the wire is caller text or a raw tool result."""
    _plant_hostile_context(tmp_path, monkeypatch)
    _, client = _run_scripted(
        monkeypatch, [_tool_call(), _done()], tasks=("do the thing",),
        bare=True, system_prompt_supplement="SEED RULES")
    last = client.requests[-1]["messages"]
    assert (last[0]["role"], last[0]["content"]) == ("system", "SEED RULES")
    assert [str(m["content"]) for m in last if m["role"] == "user"] == ["do the thing"]
    tool_msgs = [m for m in last if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert "file body" in str(tool_msgs[0]["content"])
    wire = repr(client.requests)
    for leak in ("HOSTILE", "## Environment", "Token usage", "system_warning",
                 "ground_moved", "elapsed", "helpful coding agent"):
        assert leak not in wire, leak


def test_bare_without_supplement_has_empty_system_prompt(monkeypatch, tmp_path) -> None:
    _plant_hostile_context(tmp_path, monkeypatch)
    session, _ = _run_scripted(monkeypatch, [_done()], bare=True)
    assert session["messages"][0]["content"] == ""


def test_bare_overrides_awareness_kwargs(monkeypatch) -> None:
    """bare wins over an explicit per-sense kwarg and prints no checklist."""
    events = _capture_events(monkeypatch)
    schema = load_specification(MODEL, ENDPOINT)
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        session = _core.init_session(schema, durable=False, bare=True,
                                     time_awareness_enabled=True)
        _core.init_session(schema, durable=False, bare=True, session=session)
    for t in AWARENESS_TYPES:
        assert session.get(f"{t}_awareness_enabled") is False, t
    assert not [e for e in events if e[0] == "awareness_checklist"]


def test_default_is_not_bare(monkeypatch, tmp_path) -> None:
    _plant_hostile_context(tmp_path, monkeypatch)
    session, _ = _run_scripted(monkeypatch, [_done()])
    sys_msg = session["messages"][0]["content"]
    assert "helpful coding agent" in sys_msg
    assert "HOSTILE AGENTS NOTES" in sys_msg


def test_cli_bare_flag() -> None:
    assert parse_args([MODEL]).bare is False
    assert parse_args([MODEL, "--bare"]).bare is True
