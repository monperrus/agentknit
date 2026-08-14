"""Regression tests for interactive REPL input."""

from __future__ import annotations

import builtins
import io

from agentknit import _core


def test_read_repl_input_keeps_all_lines_of_a_paste(monkeypatch):
    """Read directly from stdin so readline cannot hide pasted lines."""
    monkeypatch.setattr(_core.sys, "stdin", io.StringIO("first\nsecond\nthird\n"))
    monkeypatch.setattr(_core.select, "select", lambda *_: ([_core.sys.stdin], [], []))
    history = []
    monkeypatch.setattr(_core.readline, "add_history", history.append)
    monkeypatch.setattr(
        builtins, "input", lambda _: (_ for _ in ()).throw(AssertionError("must not use input"))
    )

    assert _core.read_repl_input("prompt> ") == "first\nsecond\nthird"
    assert history == ["first\nsecond\nthird"]
