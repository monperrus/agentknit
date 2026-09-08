"""Durable write-ahead journal for agent sessions.

The message snapshot (``<session_id>_messages.json``) is only rewritten at
turn boundaries, so a crash mid-turn loses the whole turn from the
transcript — *including tool calls whose side effects already happened*
(file writes, shell commands, deployments).  Resuming from the snapshot
alone makes the model re-derive the turn and blindly re-run those tools.

The journal closes that gap: every state transition inside a turn is
appended and fsync'd *as it happens*, so a resumed session can be
reconstructed to the exact point of failure:

* ``message`` — one record per message appended to ``session["messages"]``
* ``tool_start`` — written **before** a tool executes (call_id, name, args)
* ``tool_end`` — written **after** it returns (call_id, result)
* ``turn_start`` / ``turn_end`` — turn lifecycle markers
* ``reset_messages`` — the history was replaced (compaction, ``/clear``)

Replay semantics (:func:`replay_journal`):

* messages rebuild the conversation exactly as it was;
* a ``tool_start`` with no matching ``tool_end`` means the process died
  while (or before) the tool ran — its side effects are *unknown*, and the
  recovery path must tell the model to verify state instead of re-running;
* a ``tool_end`` with no later assistant message means the tool ran and its
  result is known — the result is re-injected so the tool is not re-run.

A torn tail line (crash mid-write) is ignored: it cannot be part of a
completed record because every record is a single fsync'd line.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime as _dt
from pathlib import Path
from typing import Any, Protocol


# A conversation message / journal record (JSON-shaped dict).
Msg = dict[str, Any]
Args = dict[str, Any]


class DurableSink(Protocol):
    """Synchronous destination for the ordered session event stream.

    ``append`` must not return until *record* is durable.  AgentKnit calls it
    on the producer side of every boundary, before a handler, model request,
    or tool can consume that record.
    """

    def append(self, record: dict[str, Any]) -> None: ...


def new_call_id() -> str:
    """Return a fresh tool-call id for journaling inline tool calls."""
    return uuid.uuid4().hex[:12]


@dataclass(frozen=True)
class PendingToolCall:
    """A tool call that started but whose completion is unknown."""

    call_id: str
    name: str
    args: Args


@dataclass(frozen=True)
class KnownToolResult:
    """A tool result the process recorded but the model never saw."""

    call_id: str
    name: str
    result: str


@dataclass
class JournalState:
    """Result of replaying a session journal."""

    messages: list[Msg] = field(default_factory=list)
    pending_tool_calls: list[PendingToolCall] = field(default_factory=list)
    unreceived_results: list[KnownToolResult] = field(default_factory=list)
    entries_replayed: int = 0
    completed_turns: int = 0
    mid_turn: bool = False


class SessionJournal:
    """Append-only, fsync-per-record write-ahead journal for one session."""

    def __init__(self, path: str | Path, *, exclusive: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fsync_directory()
        self._lock_fd: int | None = None
        if exclusive:
            # A session directory has one ordered writer.  A non-blocking
            # lock turns accidental concurrent ownership into a clear error
            # instead of interleaved JSON records.
            import fcntl
            self._lock_fd = os.open(self.path.with_suffix(self.path.suffix + ".lock"),
                                    os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(self._lock_fd)
                self._lock_fd = None
                raise RuntimeError(f"session journal is already open: {self.path}")
        self._repair_torn_tail()
        self._seq = self._count_records()

    def close(self) -> None:
        """Release the optional one-writer lock."""
        if self._lock_fd is None:
            return
        import fcntl
        fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
        os.close(self._lock_fd)
        self._lock_fd = None

    def _fsync_directory(self) -> None:
        """Make a newly-created journal directory durable on POSIX filesystems."""
        try:
            fd = os.open(self.path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _repair_torn_tail(self) -> None:
        """Drop only an incomplete final record; reject corruption before it."""
        if not self.path.exists():
            return
        raw = self.path.read_bytes()
        if not raw:
            return
        lines = raw.splitlines(keepends=True)
        repaired = False
        for index, line in enumerate(lines):
            complete = line.endswith((b"\n", b"\r"))
            try:
                json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if index == len(lines) - 1 and not complete:
                    raw = b"".join(lines[:index])
                    with self.path.open("wb") as f:
                        f.write(raw)
                        f.flush()
                        os.fsync(f.fileno())
                    self._fsync_directory()
                    repaired = True
                    break
                raise ValueError(f"corrupt session journal at record {index + 1}") from exc
        if repaired:
            return

    def _count_records(self) -> int:
        if not self.path.exists():
            return 0
        n = 0
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
        return n

    def append(self, record: dict[str, Any]) -> None:
        """Append one record with a sequence number and timestamp, fsync'd."""
        self._seq += 1
        rec = {
            "seq": self._seq,
            "ts": _dt.now().isoformat(timespec="milliseconds"),
            **record,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    # ── record builders ───────────────────────────────────────────────────

    def turn_start(self, task: str | None) -> None:
        self.append({"type": "turn_start", "task": task})

    def turn_end(self) -> None:
        self.append({"type": "turn_end"})

    def message(self, msg: Msg) -> None:
        """Journal one message exactly as appended to the session history."""
        self.append({"type": "message", "msg": msg})

    def tool_start(self, call_id: str, name: str, args: Args) -> None:
        """Record that *name(args)* is about to execute (side effects unknown)."""
        self.append({"type": "tool_start", "call_id": call_id,
                     "name": name, "args": args})

    def tool_end(self, call_id: str, result: str, name: str = "") -> None:
        """Record that the tool call finished and its observable result."""
        self.append({"type": "tool_end", "call_id": call_id, "result": result})

    def reset_messages(self, messages: list[Msg], reason: str = "") -> None:
        """Record that the history was replaced (compaction, ``/clear``)."""
        self.append({"type": "reset_messages", "reason": reason,
                     "messages": list(messages)})


def replay_journal(path: str | Path) -> JournalState:
    """Reconstruct session state from a journal file.

    A torn tail is ignored because it was never acknowledged.  Corruption in
    the middle is rejected: silently skipping it would make recovery lie.
    """
    state = JournalState()
    journal = Path(path)
    if not journal.exists():
        return state

    pending: dict[str, PendingToolCall] = {}
    finished: dict[str, KnownToolResult] = {}
    open_turns = 0

    raw_lines = journal.read_bytes().splitlines(keepends=True)
    for index, raw_line in enumerate(raw_lines):
            complete = raw_line.endswith((b"\n", b"\r"))
            try:
                line = raw_line.decode("utf-8").strip()
            except UnicodeDecodeError as exc:
                if index == len(raw_lines) - 1 and not complete:
                    break
                raise ValueError(f"corrupt session journal at record {index + 1}") from exc
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                if index == len(raw_lines) - 1 and not complete:
                    break
                raise ValueError(f"corrupt session journal at record {index + 1}") from exc
            if not isinstance(rec, dict):
                continue
            rtype = rec.get("type")
            state.entries_replayed += 1
            if rtype == "message":
                msg = rec.get("msg")
                if isinstance(msg, dict):
                    state.messages.append(dict(msg))
            elif rtype == "reset_messages":
                msgs = rec.get("messages")
                if isinstance(msgs, list):
                    state.messages = [dict(m) for m in msgs if isinstance(m, dict)]
                else:
                    state.messages = []
            elif rtype == "tool_start":
                call_id = str(rec.get("call_id") or new_call_id())
                name = str(rec.get("name") or "?")
                args = rec.get("args")
                pending[call_id] = PendingToolCall(
                    call_id=call_id, name=name,
                    args=args if isinstance(args, dict) else {},
                )
            elif rtype == "tool_end":
                call_id = str(rec.get("call_id") or "")
                pending.pop(call_id, None)
                # Track the result: a crash after tool_end but before the
                # corresponding assistant/tool message means the model never
                # saw it — recovery re-injects it instead of re-running.
                finished[call_id] = KnownToolResult(
                    call_id=call_id,
                    name=str(rec.get("name") or ""),
                    result=str(rec.get("result") or ""),
                )
            elif rtype == "turn_start":
                open_turns += 1
            elif rtype == "turn_end":
                if open_turns > 0:
                    open_turns -= 1
                state.completed_turns += 1

    state.pending_tool_calls = list(pending.values())
    # A finished call whose result already reached the conversation (via the
    # journalled tool message) is no longer unreceived.
    seen_ids = set()
    for msg in state.messages:
        tc_id = msg.get("tool_call_id")
        if msg.get("role") == "tool" and isinstance(tc_id, str):
            seen_ids.add(tc_id)
    state.unreceived_results = [
        r for cid, r in sorted(finished.items()) if cid not in seen_ids
    ]
    state.mid_turn = open_turns > 0
    return state
