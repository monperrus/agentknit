"""Regression tests for interactive REPL input."""

from __future__ import annotations

import builtins
import io

from agentknit import _core


def _fake_stdio(monkeypatch, data: str) -> list[str]:
    """Point stdin at *data*, make select() report pending input, spy history."""
    monkeypatch.setattr(_core.sys, "stdin", io.StringIO(data))
    monkeypatch.setattr(_core.select, "select", lambda *_: ([_core.sys.stdin], [], []))
    history: list[str] = []
    monkeypatch.setattr(_core.readline, "add_history", history.append)
    monkeypatch.setattr(
        builtins, "input", lambda _: (_ for _ in ()).throw(AssertionError("must not use input"))
    )
    return history


def test_read_repl_input_keeps_all_lines_of_a_paste(monkeypatch):
    """Read directly from stdin so readline cannot hide pasted lines."""
    history = _fake_stdio(monkeypatch, "first\nsecond\nthird\n")

    assert _core.read_repl_input("prompt> ") == "first\nsecond\nthird"
    assert history == ["first\nsecond\nthird"]


def test_read_repl_input_backslash_continuation(monkeypatch):
    """A trailing backslash keeps the prompt open — the hand-typed newline."""
    history = _fake_stdio(monkeypatch, "select * from\\\n  users\n")

    assert _core.read_repl_input("prompt> ") == "select * from\n  users"
    assert history == ["select * from\n  users"]


def test_read_repl_input_continuation_then_paste(monkeypatch):
    """Continuation lines and a following paste coalesce into one turn."""
    _fake_stdio(monkeypatch, "one\\\ntwo\nthree\n")

    assert _core.read_repl_input("prompt> ") == "one\ntwo\nthree"
