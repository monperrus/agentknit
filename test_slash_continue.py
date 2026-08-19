"""`/c` must retry the interrupted turn, not just flag it and stop.

`SlashCommandRegistry.dispatch` resolves the ``_continue_requested`` flag
itself via an optional ``on_continue`` callback, so callers (the REPL, or
any other front end) don't need to know the flag exists.
"""

from __future__ import annotations

from agentknit import _core
from agentknit.slash_commands import REGISTRY


def test_dispatch_without_on_continue_leaves_flag_for_caller():
    session = {"messages": []}
    handled = REGISTRY.dispatch("/c", session, client=None, model="m")

    assert handled is True
    assert session["_continue_requested"] is True  # untouched, back-compat


def test_dispatch_with_on_continue_invokes_it_and_clears_flag():
    session = {"messages": []}
    calls = []
    handled = REGISTRY.dispatch(
        "/c", session, client=None, model="m", on_continue=lambda: calls.append(1)
    )

    assert handled is True
    assert calls == [1]
    assert "_continue_requested" not in session


def test_dispatch_with_on_continue_is_noop_for_other_commands():
    session = {"messages": [{"role": "system", "content": "sys"}]}
    calls = []
    handled = REGISTRY.dispatch(
        "/usage", session, client=None, model="m", on_continue=lambda: calls.append(1)
    )

    assert handled is True
    assert calls == []


def test_repl_loop_body_retries_turn_on_slash_c(monkeypatch):
    session = {"messages": [{"role": "system", "content": "sys"}], "model": "m",
               "session_id": "sid"}
    calls = []
    monkeypatch.setattr(_core, "_sync_repl_turn",
                        lambda t, client, session, model: calls.append(t))
    monkeypatch.setattr(_core, "_save_messages_snapshot", lambda session: None)

    _core._repl_loop_body("/c", client=None, session=session, model="m")

    assert calls == [None]
