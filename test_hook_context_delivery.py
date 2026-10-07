"""Hook context reaches the model exactly once, even in a turn without tool calls.

Before: UserPromptSubmit's additionalContext and a resumed session's
SessionStart output only rode on the next tool result (lost when the reply had
no tool calls), and a fresh session's SessionStart output was seen twice.
"""

from __future__ import annotations

import sys

import agentknit._core as core
from agentknit import hooks as H
from agentknit._core import init_session
from test_hooks_integration import _ScriptedClient, _session, _text_response, _tool_call_response

NOTES = {"hooks": {"SessionStart": [{"hooks": [{"type": "command",
                                                "command": f"{sys.executable} -c 'print(\"INTENT: ship weekly\")'"}]}]}}


def _sent(client: _ScriptedClient, role: str) -> list[str]:
    return [m.get("content") or "" for r in client.requests for m in r["messages"] if m["role"] == role]


def test_userpromptsubmit_context_rides_on_the_prompt(tmp_path, monkeypatch) -> None:
    session = _session(tmp_path, monkeypatch)
    H.register_hook(session, "UserPromptSubmit", lambda p: {"hookSpecificOutput": {
        "hookEventName": "UserPromptSubmit", "additionalContext": "owner prefers terse"}})
    client = _ScriptedClient([_text_response("ok")])  # no tool call in this turn
    core.run_turn(client, "test/model", session, "status?")
    assert "owner prefers terse" in _sent(client, "user")[-1]


def test_fresh_sessionstart_context_is_seen_once(tmp_path, monkeypatch) -> None:
    session = _session(tmp_path, monkeypatch, behaviour=NOTES)
    assert "INTENT: ship weekly" in session["messages"][0]["content"]
    client = _ScriptedClient([_tool_call_response("c1", "read_file", {"path": "x"}), _text_response("ok")])
    core.run_turn(client, "test/model", session, "go")
    later = [m.get("content") or "" for m in client.requests[-1]["messages"] if m["role"] != "system"]
    assert not any("INTENT: ship weekly" in c for c in later)


def test_resumed_sessionstart_context_reaches_a_toolless_turn(tmp_path, monkeypatch) -> None:
    first = _session(tmp_path, monkeypatch, behaviour=NOTES)
    core.run_turn(_ScriptedClient([_text_response("done")]), "test/model", first, "first turn")
    schema = {"model": "test/model", "endpoint": "https://api.example.com/v1", "inferred_tool_schema": [],
              "tool_dispatch": {"read_file": {"python_function": "t_read", "param_map": {}}}, "behaviour": NOTES}
    resumed = init_session(schema, strict_cache_proof=False, resumed_from=first["session_id"])
    client = _ScriptedClient([_text_response("ok")])
    core.run_turn(client, "test/model", resumed, "quote the intent")
    assert "INTENT: ship weekly" in _sent(client, "user")[-1]
