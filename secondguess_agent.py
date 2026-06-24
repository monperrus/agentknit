#!/usr/bin/env python3
"""
Second-guess coding agent: standard read/write/edit/exec tools, but every
execute_shell_command call waits 2 seconds before starting, giving the
operator (human Ctrl-C or supervisor LLM cancellation) time to abort.

Exposed tools:
  execute_shell_command  — shell exec with 2 s grace period
  query_tool_exec        — poll running commands
  read_file              — read file contents
  write_file             — write file contents
  str_replace_edit       — substring replacement edit
"""
from __future__ import annotations

import argparse
import json
import queue
import select
import sys
import threading
import time

from agentknit import Tool, build_tool_spec, register_tools_in_library
from agentknit import (
    t_execute_async, t_query_exec,
    ASYNC_FAST_THRESHOLD_S, ASYNC_INLINE_MAX_BYTES,
    async_completion_queue,
    run_task,
)
from agentknit.tool_library import t_read, t_write, t_update, _async_try_inline, _async_last_lines
from agentknit._core import (
    create_client, init_session, run_turn, _save_messages_snapshot,
    DEFAULT_ENDPOINT,
    BOLD, DIM, RESET, RL_BOLD, RL_RESET,
    print_session_history, _build_resume_cmd,
)

# ── grace-period cancellation mechanism ───────────────────────────────────────
# Set this Event to True to cancel the next pending exec command.
_grace_cancel: threading.Event = threading.Event()

GRACE_PERIOD_SECONDS = 2.0
GRACE_POLL_INTERVAL   = 0.1


def cancel_next_exec() -> None:
    """Request cancellation of the next pending exec command.

    Call from another thread (e.g. a supervisor LLM via a tool or signal
    handler) to abort the command during its 2-second grace period.
    """
    _grace_cancel.set()


def t_secondguess_exec(command: str, when: int = 0) -> tuple[str, dict]:
    """Execute a shell command, but wait {GRACE_PERIOD_SECONDS}s before
    starting so the operator can Ctrl-C to abort.

    During the grace period the operator (human or supervisor LLM) can:
    - Press Ctrl-C (KeyboardInterrupt) to cancel.
    - Call cancel_next_exec() to programmatically cancel.

    If cancelled, returns a cancellation notice instead of running the command.
    """
    # Apply the `when` delay first (if any) — same as t_execute_async does.
    if when:
        time.sleep(when * 60)

    # ── grace period ─────────────────────────────────────────────────────
    # Check first — a supervisor may have called cancel_next_exec() while
    # the agent was *generating* this tool call (before we started).
    cancelled = False
    if _grace_cancel.is_set():
        cancelled = True
    else:
        steps = int(GRACE_PERIOD_SECONDS / GRACE_POLL_INTERVAL)
        for _ in range(steps):
            if _grace_cancel.is_set():
                cancelled = True
                break
            try:
                time.sleep(GRACE_POLL_INTERVAL)
            except KeyboardInterrupt:
                cancelled = True
                break

    # Reset for the next potential call.  Only clear after the grace period
    # fully completes (or the command runs) so a pre-set flag is not lost.
    _grace_cancel.clear()

    if cancelled:
        r = json.dumps({
            "cancelled": True,
            "command": command,
            "message": (
                f"Command was cancelled during the {GRACE_PERIOD_SECONDS}s "
                f"grace period."
            ),
        })
        return r, {"result": r}

    # ── proceed with actual execution ────────────────────────────────────
    # Pass when=0 because we already applied the delay above.
    return t_execute_async(command, when=0)


# ── tool definitions ─────────────────────────────────────────────────────────
_TOOLS = [
    Tool(
        "execute_shell_command",
        (
            f"Start a shell command asynchronously with a {GRACE_PERIOD_SECONDS}s "
            f"grace period. Returns tool_exec_id and local file paths for stdin "
            f"(FIFO), stdout, and stderr. Write to stdin_localfile to send input "
            f"to the running process. Optional `when` (integer minutes, default 0) "
            f"delays the start. If the command finishes within "
            f"{int(ASYNC_FAST_THRESHOLD_S * 1000)} ms and both outputs are under "
            f"{ASYNC_INLINE_MAX_BYTES} bytes, stdout/stderr are inlined immediately. "
            f"**IMPORTANT**: The command does NOT start immediately. There is a "
            f"{GRACE_PERIOD_SECONDS}s grace period during which you can cancel by "
            f"pressing Ctrl-C (or having a supervisor call cancel_next_exec()). "
            f"Use query_tool_exec to poll status. When a background command "
            f"finishes the agent is notified automatically."
        ),
        t_secondguess_exec,
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run."},
                "when": {
                    "type": "integer",
                    "description": "Minutes to wait before starting the command (default 0).",
                },
            },
            "required": ["command"],
        },
    ),
    Tool(
        "query_tool_exec",
        f"Poll a command started with execute_shell_command. When completed, "
        f"includes returncode and inlines stdout/stderr if both are under "
        f"{ASYNC_INLINE_MAX_BYTES} bytes; otherwise reports file sizes.",
        t_query_exec,
        parameters={
            "type": "object",
            "properties": {
                "tool_exec_id": {
                    "type": "string",
                    "description": "The ID returned by execute_shell_command.",
                },
            },
            "required": ["tool_exec_id"],
        },
    ),
    Tool(
        "read_file",
        "Read the contents of a local file.",
        t_read,
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file."},
            },
            "required": ["path"],
        },
    ),
    Tool(
        "write_file",
        "Write content to a local file, creating parent directories as needed.",
        t_write,
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file."},
                "content": {"type": "string", "description": "Content to write."},
            },
            "required": ["path", "content"],
        },
    ),
    Tool(
        "str_replace_edit",
        "Edit an existing file by replacing a specific substring. Use this for "
        "precise edits — supply enough context in old_str to uniquely identify "
        "the target location.",
        t_update,
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file."},
                "old_str": {
                    "type": "string",
                    "description": "The exact text to replace (must exist in the file).",
                },
                "new_str": {
                    "type": "string",
                    "description": "The replacement text.",
                },
            },
            "required": ["path", "old_str", "new_str"],
        },
    ),
]

_TOOL_SCHEMA, _TOOL_DISPATCH = build_tool_spec(_TOOLS)
register_tools_in_library(_TOOLS)

_SYSTEM_SUPPLEMENT = (
    "You are a coding agent. Start shell commands with execute_shell_command — "
    "they run in the background. Use query_tool_exec to poll status. Pass `when` "
    "to execute_shell_command to delay a command by N minutes. When a background "
    "command finishes you will be notified automatically with its output.\n\n"
    f"**Grace period**: every execute_shell_command call pauses {GRACE_PERIOD_SECONDS}s "
    "before actually starting the command. During this window the operator can "
    "Ctrl-C to abort. The command only begins after the grace period expires."
)


def _build_schema(model: str, endpoint: str) -> dict:
    return {
        "model": model,
        "endpoint": endpoint,
        "status": "default",
        "inferred_tool_schema": _TOOL_SCHEMA,
        "behaviour": {"call_delivery_mode": "structured_tool_calls"},
        "tool_dispatch": _TOOL_DISPATCH,
    }


def _completion_message(event: dict) -> str:
    """Format a completion event into a natural-language notification for the LLM."""
    exec_id = event["tool_exec_id"]
    rc = event["returncode"]
    duration = event["duration"]
    stdout_path = event["stdout_file"]
    stderr_path = event["stderr_file"]
    stdout = _async_try_inline(stdout_path)
    if stdout is None:
        tail = _async_last_lines(stdout_path, 3)
        stdout_text = f"(see {stdout_path})" + (f"\nlast 3 lines:\n{tail}" if tail else "")
    else:
        stdout_text = stdout.rstrip() or "(empty)"
    stderr = _async_try_inline(stderr_path)
    if stderr is None:
        tail = _async_last_lines(stderr_path, 3)
        stderr_text = f"(see {stderr_path})" + (f"\nlast 3 lines:\n{tail}" if tail else "")
    else:
        stderr_text = stderr.rstrip()
    parts = [f"Background command {exec_id} finished after {duration:.1f}s "
             f"(returncode={rc})."]
    parts.append(f"stdout: {stdout_text}")
    if stderr_text.strip():
        parts.append(f"stderr: {stderr_text}")
    return "\n".join(parts)


def _drain_completion_queue() -> list[str]:
    """Return formatted messages for every completed async command."""
    msgs: list[str] = []
    while True:
        try:
            msgs.append(_completion_message(async_completion_queue.get_nowait()))
        except queue.Empty:
            break
    return msgs


def _repl(schema: dict, session_id: str | None, system_prompt_supplement: str) -> None:
    """Custom REPL that auto-triggers LLM turns when background commands finish."""
    import readline  # noqa: F401 — enables arrow keys in input()
    import hashlib

    client = create_client(schema)
    session = init_session(
        schema,
        resumed_from=session_id,
        system_prompt_supplement=system_prompt_supplement,
    )
    model = schema["model"]
    resume_cmd = _build_resume_cmd(model, session["session_id"])

    if session_id:
        print_session_history(session)

    # Per-directory readline history
    from pathlib import Path
    _hist_dir = Path.home() / ".local/share/agent_probe/repl_history"
    _hist_dir.mkdir(parents=True, exist_ok=True)
    _hist_file = _hist_dir / f"{hashlib.md5(str(Path.cwd()).encode()).hexdigest()[:12]}.hist"
    try:
        readline.read_history_file(_hist_file)
    except FileNotFoundError:
        pass
    readline.set_history_length(500)

    print(f"{BOLD}secondguess-agent {model}{RESET}  (type 'exit' to quit)\n")

    _POLL_INTERVAL = 0.25  # seconds between completion-queue checks at the prompt

    pending: list[str] = []

    def _run(task: str) -> None:
        try:
            run_turn(client, model, session, task)
        except KeyboardInterrupt:
            print(f"\n{DIM}[interrupted]{RESET}")
        finally:
            _save_messages_snapshot(session)

    try:
        while True:
            # Drain any completions that arrived during the last turn.
            for msg in _drain_completion_queue():
                pending.append(msg)

            if pending:
                task = pending.pop(0)
                print(f"{DIM}[completion] {task.splitlines()[0]}{RESET}")
                _run(task)
                continue

            # Wait for user input, polling the completion queue periodically.
            sys.stdout.write(f"{RL_BOLD}>{RL_RESET} ")
            sys.stdout.flush()
            user_task: str | None = None
            try:
                while user_task is None:
                    ready, _, _ = select.select([sys.stdin], [], [], _POLL_INTERVAL)
                    if ready:
                        line = sys.stdin.readline()
                        if not line:  # EOF
                            raise EOFError
                        user_task = line.rstrip("\n")
                    else:
                        for msg in _drain_completion_queue():
                            pending.append(msg)
                        if pending and not readline.get_line_buffer():
                            sys.stdout.write("\n")
                            sys.stdout.flush()
                            break  # leave user_task = None, loop will handle pending
            except KeyboardInterrupt:
                print()
                continue
            except EOFError:
                print()
                break

            if user_task is None:
                continue  # completion arrived while waiting; loop handles it

            cmd = user_task.strip()
            if not cmd:
                continue
            if cmd.lower() in ("exit", "quit", "q"):
                break

            _run(user_task)

    finally:
        try:
            readline.write_history_file(_hist_file)
        except Exception:
            pass
        _save_messages_snapshot(session)
        print(f"\n{DIM}Resume: {resume_cmd}{RESET}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Second-guess coding agent: standard tools with a 2-second "
                    "grace period before every shell exec so the operator can Ctrl-C."
    )
    p.add_argument("model", help="Model ID or run:// URI")
    p.add_argument("task", nargs="*", help="One-shot task (omit for REPL)")
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--session", metavar="SESSION_ID")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    schema = _build_schema(args.model, args.endpoint)
    opts = dict(session_id=args.session, system_prompt_supplement=_SYSTEM_SUPPLEMENT)
    if args.task:
        run_task(schema, " ".join(args.task), **opts)
    else:
        _repl(schema, args.session, _SYSTEM_SUPPLEMENT)


if __name__ == "__main__":
    main()
