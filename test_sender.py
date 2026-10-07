"""Tests for the sender of a user message: who wrote it (human, supervisor, launcher, peer, hook)."""

from __future__ import annotations

import json

import pytest

import agentknit._core as core
from agentknit import hooks as H
from agentknit._core import _journal_path, _normalise_for_resume, sender_header
from test_hooks_integration import _ScriptedClient, _session, _text_response

KIMI = {"kind": "supervisor", "name": "kimi-k3"}


def _user_requests(client: _ScriptedClient) -> list[dict]:
    return [m for m in client.requests[-1]["messages"] if m["role"] == "user"]


def test_a_supervisor_message_is_recorded_and_headed(tmp_path, monkeypatch) -> None:
    session = _session(tmp_path, monkeypatch)
    seen: list[dict] = []
    H.register_hook(session, "UserPromptSubmit", lambda p: seen.append(p) or None)
    client = _ScriptedClient([_text_response("ok")])
    core.run_turn(client, "test/model", session, "Go on with step 3.", sender=KIMI)

    assert seen[0]["sender"] == KIMI and seen[0]["prompt"] == "Go on with step 3."
    sent = _user_requests(client)[-1]
    assert sent["content"].startswith("[supervisor kimi-k3, answering in the operator's place; "
                                      "not typed by the operator]\nGo on with step 3.")
    assert "sender" not in sent  # the provider never sees the private key
    assert [m for m in session["messages"] if m["role"] == "user"][-1]["sender"] == KIMI
    journal = [json.loads(x) for x in _journal_path("test/model", session["session_id"]).read_text().splitlines()]
    assert [r["sender"] for r in journal if r["type"] == "turn_start"] == [KIMI]


def test_the_human_is_unchanged(tmp_path, monkeypatch) -> None:
    session = _session(tmp_path, monkeypatch)
    seen: list[dict] = []
    H.register_hook(session, "UserPromptSubmit", lambda p: seen.append(p) or None)
    client = _ScriptedClient([_text_response("ok")])
    core.run_turn(client, "test/model", session, "status?")

    assert seen[0]["sender"] == {"kind": "human"}
    assert _user_requests(client)[-1]["content"].startswith("status?")
    assert "sender" not in [m for m in session["messages"] if m["role"] == "user"][-1]
    journal = [json.loads(x) for x in _journal_path("test/model", session["session_id"]).read_text().splitlines()]
    assert all("sender" not in r for r in journal if r["type"] == "turn_start")
    phone = {"kind": "human", "name": "phone"}  # a named human is recorded, never headed
    assert sender_header(phone) == ""


def test_boundary_merges_keep_every_sender_and_bad_kinds_fail(tmp_path, monkeypatch) -> None:
    msgs = [{"role": "system", "content": "s"},
            {"role": "user", "content": "[supervisor kimi-k3, …]\nGo on", "sender": KIMI},
            {"role": "user", "content": "no, stop"}]  # the owner, after an unanswered supervisor message
    merged = [m for m in _normalise_for_resume(msgs) if m["role"] == "user"]
    assert len(merged) == 1 and merged[0]["sender"] == [KIMI, {"kind": "human"}]
    same = [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
    assert "sender" not in _normalise_for_resume(same)[0]  # two human messages stay unmarked

    session = _session(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="unknown sender kind"):
        core.run_turn(_ScriptedClient([]), "test/model", session, "x", sender={"kind": "owner"})
