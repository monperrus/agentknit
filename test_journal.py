"""Tests for the durable recovery write-ahead journal (agentknit._journal)."""

from __future__ import annotations

import json

from agentknit._journal import (
    JournalState,
    KnownToolResult,
    PendingToolCall,
    SessionJournal,
    new_call_id,
    replay_journal,
)


# ── SessionJournal.append ─────────────────────────────────────────────────────

def test_append_assigns_monotonic_seq_and_persists(tmp_path):
    j = SessionJournal(tmp_path / "s_journal.jsonl")
    j.turn_start("do work")
    j.message({"role": "user", "content": "hello"})
    j.turn_end()

    lines = (tmp_path / "s_journal.jsonl").read_text().splitlines()
    assert len(lines) == 3
    recs = [json.loads(ln) for ln in lines]
    assert [r["seq"] for r in recs] == [1, 2, 3]
    assert all("ts" in r for r in recs)


def test_append_resumes_seq_across_reopen(tmp_path):
    path = tmp_path / "s_journal.jsonl"
    j = SessionJournal(path)
    j.message({"role": "user", "content": "one"})
    # Simulate a restart: a new journal object over the same file.
    j2 = SessionJournal(path)
    j2.message({"role": "user", "content": "two"})
    recs = [json.loads(ln) for ln in path.read_text().splitlines()]
    assert [r["seq"] for r in recs] == [1, 2]


# ── replay_journal ────────────────────────────────────────────────────────────

def test_replay_empty_or_missing_file(tmp_path):
    state = replay_journal(tmp_path / "nope.jsonl")
    assert isinstance(state, JournalState)
    assert state.messages == []
    assert state.pending_tool_calls == []
    assert state.entries_replayed == 0
    assert state.mid_turn is False


def test_replay_reconstructs_messages(tmp_path):
    j = SessionJournal(tmp_path / "s_journal.jsonl")
    j.turn_start("task")
    j.message({"role": "user", "content": "hi"})
    j.message({"role": "assistant", "content": "hello"})
    j.turn_end()

    state = replay_journal(tmp_path / "s_journal.jsonl")
    assert [m["content"] for m in state.messages] == ["hi", "hello"]
    assert state.mid_turn is False
    assert state.completed_turns == 1


def test_replay_completed_tool_call_is_not_pending(tmp_path):
    j = SessionJournal(tmp_path / "s_journal.jsonl")
    j.tool_start("call-1", "t_run", {"command": "deploy.sh"})
    j.tool_end("call-1", "deploy ok")

    state = replay_journal(tmp_path / "s_journal.jsonl")
    assert state.pending_tool_calls == []


def test_replay_unfinished_tool_call_is_pending(tmp_path):
    """Crash between tool_start and tool_end → side effects unknown."""
    j = SessionJournal(tmp_path / "s_journal.jsonl")
    j.turn_start("task")
    j.tool_start("call-1", "t_run", {"command": "deploy.sh"})
    # No tool_end: process died mid-execution.

    state = replay_journal(tmp_path / "s_journal.jsonl")
    assert state.pending_tool_calls == [
        PendingToolCall(call_id="call-1", name="t_run", args={"command": "deploy.sh"})
    ]
    assert state.mid_turn is True


def test_replay_tool_end_without_result_message(tmp_path):
    """Tool finished but the model never saw the result → result is known."""
    j = SessionJournal(tmp_path / "s_journal.jsonl")
    j.message({"role": "user", "content": "go"})
    j.tool_start("c1", "t_write", {"path": "a.txt", "content": "x"})
    j.tool_end("c1", "wrote 1 bytes", name="t_write")

    state = replay_journal(tmp_path / "s_journal.jsonl")
    assert state.pending_tool_calls == []
    assert len(state.messages) == 1  # the tool result lives in the journal
    assert state.unreceived_results == [
        KnownToolResult(call_id="c1", name="", result="wrote 1 bytes")
    ]


def test_replay_tool_end_followed_by_tool_message_is_received(tmp_path):
    """Result already in the conversation → not unreceived."""
    j = SessionJournal(tmp_path / "s_journal.jsonl")
    j.message({"role": "user", "content": "go"})
    j.tool_start("c1", "t_read", {"path": "a.txt"})
    j.tool_end("c1", "body", name="t_read")
    j.message({"role": "tool", "tool_call_id": "c1", "content": "body"})

    state = replay_journal(tmp_path / "s_journal.jsonl")
    assert state.unreceived_results == []


def test_replay_reset_messages_replaces_history(tmp_path):
    j = SessionJournal(tmp_path / "s_journal.jsonl")
    j.message({"role": "user", "content": "old1"})
    j.message({"role": "user", "content": "old2"})
    j.reset_messages([{"role": "system", "content": "sys"},
                      {"role": "user", "content": "kept"}], reason="compaction")
    j.message({"role": "assistant", "content": "after"})

    state = replay_journal(tmp_path / "s_journal.jsonl")
    assert [m["content"] for m in state.messages] == ["sys", "kept", "after"]


def test_replay_ignores_torn_tail(tmp_path):
    path = tmp_path / "s_journal.jsonl"
    j = SessionJournal(path)
    j.message({"role": "user", "content": "ok"})
    # Simulate a crash mid-write: half a JSON line.
    with path.open("a") as f:
        f.write('{"type": "tool_start", "call_i')

    state = replay_journal(path)
    assert state.entries_replayed == 1
    assert [m["content"] for m in state.messages] == ["ok"]
    assert state.pending_tool_calls == []


def test_replay_ignores_non_dict_lines(tmp_path):
    path = tmp_path / "s_journal.jsonl"
    path.write_text('"just a string"\n42\n{"type": "turn_end"}\n')
    state = replay_journal(path)
    assert state.completed_turns == 1


def test_replay_unbalanced_turn_start_marks_mid_turn(tmp_path):
    j = SessionJournal(tmp_path / "s_journal.jsonl")
    j.turn_start("t1")
    j.turn_end()
    j.turn_start("t2")  # crash before turn_end
    state = replay_journal(j.path)
    assert state.mid_turn is True
    assert state.completed_turns == 1


def test_new_call_id_unique():
    ids = {new_call_id() for _ in range(100)}
    assert len(ids) == 100
