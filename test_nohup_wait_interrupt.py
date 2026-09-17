"""nohup_wait must not hold the conversation hostage.

Two halves of the same guarantee:

* a wait is cut short as soon as the user types (``wait_interrupt_hook``), so
  the turn ends and the queued message runs;
* the execution that kept running is reported to the model afterwards
  (``drain_completions`` / ``completion_notice``), so nothing is lost.
"""

from __future__ import annotations

import json
import time

import pytest

import agentknit.async_toolkit as at
from agentknit._core import BackgroundWake, _InputCollector, read_repl_input


@pytest.fixture(autouse=True)
def _clean_hook():
    at.wait_interrupt_hook = None
    at.drain_completions()
    yield
    at.wait_interrupt_hook = None
    at.drain_completions()


def test_wait_returns_early_when_user_types() -> None:
    """The hook firing ends the wait long before the budget expires."""
    exec_id = json.loads(t_start("sleep 30"))["tool_exec_id"]
    typed = {"yes": False}
    at.wait_interrupt_hook = lambda: typed["yes"]

    # Not typing yet: a short budget is honoured in full.
    started = time.monotonic()
    d = json.loads(at.t_nohup_wait(exec_id, howmuch=1, unit="s")[0])
    assert d["completed"] is False
    assert "interrupted_by" not in d
    assert time.monotonic() - started >= 0.9

    # Typing: the wait returns right away, with the reason and the advice.
    typed["yes"] = True
    started = time.monotonic()
    d = json.loads(at.t_nohup_wait(exec_id, howmuch=30, unit="m")[0])
    assert time.monotonic() - started < 1.0
    assert d["completed"] is False
    assert d["interrupted_by"] == "user_input"
    assert "end your turn" in d["hint"]
    assert d["waited_seconds"] < 1.0

    at._async_executions[exec_id]["proc"].kill()  # type: ignore[union-attr]


def test_wait_hook_is_off_by_default() -> None:
    """Without a REPL setting the hook, waits behave exactly as before."""
    assert at.wait_interrupt_hook is None
    exec_id = json.loads(t_start("sleep 30"))["tool_exec_id"]
    d = json.loads(at.t_nohup_wait(exec_id, howmuch=1, unit="s")[0])
    assert d["completed"] is False
    assert "interrupted_by" not in d
    at._async_executions[exec_id]["proc"].kill()  # type: ignore[union-attr]


def test_completion_is_drained_into_a_notice() -> None:
    """An execution finishing with nobody waiting becomes a model-facing ping."""
    exec_id = json.loads(t_start("echo hi"))["tool_exec_id"]
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not at.async_completion_queue.qsize():
        time.sleep(0.02)

    completions = at.drain_completions()
    assert [c["tool_exec_id"] for c in completions] == [exec_id]
    assert at.drain_completions() == []          # drained exactly once

    notice = at.completion_notice(completions)
    assert exec_id in notice
    assert "returncode=0" in notice
    assert "echo hi" in notice


def test_input_collector_has_pending() -> None:
    """The async REPL's queue predicate is what the hook is wired to."""
    c = _InputCollector()
    assert c.has_pending() is False
    c._q.put("hello")
    assert c.has_pending() is True
    assert c.drain() == ["hello"]
    assert c.has_pending() is False


def test_read_repl_input_wakes_on_background_event(monkeypatch) -> None:
    """An idle prompt gives way to a background event instead of blocking."""
    monkeypatch.setattr("agentknit._core.select.select",
                        lambda *a, **k: ([], [], []))   # stdin never ready
    with pytest.raises(BackgroundWake):
        read_repl_input("> ", wake=lambda: True)


def t_start(command: str) -> str:
    """Start a background command, returning t_nohup's raw JSON result."""
    return at.t_nohup(command, timeout=1)[0]
