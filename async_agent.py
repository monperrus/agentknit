#!/usr/bin/env python3
"""
Async coding agent with non-blocking shell execution.

Tools exposed to the LLM:
  execute_shell_command  — t_execute_async from agentknit.tool_library
  query_tool_exec        — t_query_exec    from agentknit.tool_library
  read_file              — t_read          from agentknit.tool_library
  write_file             — t_write         from agentknit.tool_library
"""
from __future__ import annotations

import argparse

from agentknit import Tool, build_tool_spec, register_tools_in_library
from agentknit import (
    t_execute_async, t_query_exec,
    ASYNC_FAST_THRESHOLD_S, ASYNC_INLINE_MAX_BYTES,
    run_task, run_repl,
)
from agentknit.tool_library import t_read, t_write
from agentknit._core import DEFAULT_ENDPOINT

_TOOLS = [
    Tool(
        "execute_shell_command",
        f"Start a shell command asynchronously. Returns tool_exec_id and local "
        f"file paths for stdin (FIFO), stdout, and stderr. Write to stdin_localfile "
        f"to send input to the running process. Optional `when` (integer minutes, "
        f"default 0) delays the start. If the command finishes within "
        f"{int(ASYNC_FAST_THRESHOLD_S * 1000)} ms and both outputs are under "
        f"{ASYNC_INLINE_MAX_BYTES} bytes, stdout/stderr are inlined immediately.",
        t_execute_async,
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
]

_TOOL_SCHEMA, _TOOL_DISPATCH = build_tool_spec(_TOOLS)
register_tools_in_library(_TOOLS)

_SYSTEM_SUPPLEMENT = (
    "You are a coding agent. Start shell commands with execute_shell_command — "
    "they run in the background. Use query_tool_exec to poll status. Pass `when` "
    "to execute_shell_command to delay a command by N minutes instead of polling."
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Async coding agent with non-blocking shell tools.")
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
        run_repl(schema, **opts)


if __name__ == "__main__":
    main()
