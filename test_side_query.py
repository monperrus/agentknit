"""``side_query``: answer a question mid-turn without touching the session."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import agentknit
import agentknit._core as core


def _session(tmp_path, messages, structured=True) -> dict:
    return {
        "messages": messages,
        "tools": [{"type": "function", "function": {"name": "subagent"}}],
        "structured": structured,
        "session_id": "side-test",
        "cache_key": "side-test",
        "options": [],
        "max_output_tokens": None,
        "log_path": tmp_path / "session.jsonl",
    }


class _Client:
    def __init__(self, answer: str = "it is exploring the repo") -> None:
        self.calls: list[dict] = []
        self.answer = answer
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=f"  {self.answer}\n", tool_calls=None))])


def _midturn_messages() -> list[dict]:
    """A turn blocked in a tool call: the assistant's call has no result yet."""
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "fix the bug", "ts": "t0"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "subagent", "arguments": "{}"}}]},
    ]


def test_inflight_tool_call_gets_still_running_result(tmp_path) -> None:
    session = _session(tmp_path, _midturn_messages())
    client = _Client()
    answer = agentknit.side_query(client, "m", session, "what are you doing?")

    assert answer == "it is exploring the repo"
    (call,) = client.calls
    msgs = call["messages"]
    assert msgs[3] == {"role": "tool", "tool_call_id": "c1",
                       "content": core.SIDE_QUERY_PENDING_TOOL}
    assert msgs[4]["role"] == "user"
    assert msgs[4]["content"].endswith("what are you doing?")
    assert call["tool_choice"] == "none"
    assert call["tools"] == session["tools"]
    assert call["extra_body"] == {"prompt_cache_key": "side-test"}


def test_session_is_untouched(tmp_path) -> None:
    session = _session(tmp_path, _midturn_messages())
    before = copy.deepcopy(session["messages"])
    agentknit.side_query(_Client(), "m", session, "status?")
    assert session["messages"] == before
    (record,) = [json.loads(line) for line in
                 session["log_path"].read_text().splitlines()]
    assert record["type"] == "side_query"
    assert record["question"] == "status?"


def test_answered_calls_keep_their_results(tmp_path) -> None:
    msgs = _midturn_messages() + [
        {"role": "tool", "tool_call_id": "c1", "content": "done"}]
    out = core._side_query_messages(msgs, "q", structured=True)
    assert [m.get("content") for m in out if m.get("role") == "tool"] == ["done"]


def test_non_structured_merges_into_trailing_user(tmp_path) -> None:
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "Tool results:\nx"}]
    session = _session(tmp_path, msgs, structured=False)
    client = _Client()
    agentknit.side_query(client, "m", session, "q?")
    (call,) = client.calls
    assert "tools" not in call and "tool_choice" not in call
    assert len(call["messages"]) == 2
    assert call["messages"][1]["content"].startswith("Tool results:\nx")
    assert call["messages"][1]["content"].endswith("q?")
