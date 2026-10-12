"""run_turn(sender=): who wrote a turn reaches the model, the hooks, the journal and the log."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import agentknit._core as core
from agentknit import hooks as H
from test_hooks_integration import _ScriptedClient, _session, _text_response

SUPERVISOR = {"kind": "supervisor", "name": "kimi-k3"}


def _turn(tmp_path: Path, monkeypatch, sender: dict[str, str] | None) -> tuple[dict, list[dict], str]:
    """Run one turn; return the session, the UserPromptSubmit payloads and the user message sent."""
    session = _session(tmp_path, monkeypatch)
    payloads: list[dict[str, Any]] = []
    H.register_hook(session, "UserPromptSubmit", lambda p: payloads.append(p))
    client = _ScriptedClient([_text_response("done")])
    kw = {"sender": sender} if sender is not None else {}
    core.run_turn(client, "test/model", session, "Go on with step 3.", **kw)
    sent = [m for m in client.requests[0]["messages"] if m["role"] == "user"][-1]["content"]
    return session, payloads, sent


def _log_users(session: dict) -> list[dict]:
    lines = Path(session["log_path"]).read_text().splitlines()
    return [r for r in map(json.loads, lines) if r.get("type") == "user"]


def test_supervisor_turn_is_headed_and_recorded(tmp_path, monkeypatch) -> None:
    session, payloads, sent = _turn(tmp_path, monkeypatch, SUPERVISOR)
    assert sent.startswith("[This turn is from the supervisor kimi-k3, not typed by the operator.]\n\n"
                           "Go on with step 3.")
    assert payloads[0]["sender"] == SUPERVISOR
    assert payloads[0]["prompt"] == "Go on with step 3."  # the hook sees the text as written
    assert _log_users(session)[-1]["sender"] == SUPERVISOR
    assert _log_users(session)[-1]["content"] == "Go on with step 3."


def test_hook_sender_names_an_automated_message(tmp_path, monkeypatch) -> None:
    _, _, sent = _turn(tmp_path, monkeypatch, {"kind": "hook", "name": "wake w-1 via inbox c-04"})
    assert sent.startswith("[This turn is from an automated message from wake w-1 via inbox c-04, "
                           "not typed by the operator.]")


def test_human_sender_has_no_header_but_is_recorded(tmp_path, monkeypatch) -> None:
    session, payloads, sent = _turn(tmp_path, monkeypatch, {"kind": "human", "name": "phone"})
    assert sent.startswith("Go on with step 3.")
    assert payloads[0]["sender"] == {"kind": "human", "name": "phone"}


def test_no_sender_is_the_operator_as_before(tmp_path, monkeypatch) -> None:
    session, payloads, sent = _turn(tmp_path, monkeypatch, None)
    assert sent.startswith("Go on with step 3.")
    assert "sender" not in payloads[0]
    assert "sender" not in _log_users(session)[-1]


def test_unknown_sender_kind_is_rejected(tmp_path, monkeypatch) -> None:
    session = _session(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="sender kind"):
        core.run_turn(_ScriptedClient([]), "test/model", session, "x",
                      sender={"kind": "operator", "name": "me"})


def test_journal_turn_start_carries_the_sender(tmp_path, monkeypatch) -> None:
    records: list[dict] = []
    monkeypatch.setattr(core, "_write_journal_record", lambda s, r: records.append(r))
    session = _session(tmp_path, monkeypatch)
    session["_journal"] = object()  # any journal: records go through _write_journal_record
    client = _ScriptedClient([_text_response("done")])
    core.run_turn(client, "test/model", session, "Go on.", sender=SUPERVISOR)
    starts = [r for r in records if r["type"] == "turn_start"]
    assert starts == [{"type": "turn_start", "task": "Go on.", "sender": SUPERVISOR}]
