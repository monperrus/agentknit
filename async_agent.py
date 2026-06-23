#!/usr/bin/env python3
"""
Async coding agent with non-blocking shell execution.

Tools exposed to the LLM:
  execute_shell_command  — t_execute_async from agentknit.tool_library
  query_tool_exec        — t_query_exec    from agentknit.tool_library
  plan_shell_command     — t_plan_delay    from agentknit.tool_library
  read_file              — t_read          from agentknit.tool_library
  write_file             — t_write         from agentknit.tool_library
"""
from __future__ import annotations

import argparse

from agentknit import Tool, build_tool_spec, register_tools_in_library
from agentknit import (
    t_execute_async, t_query_exec, t_plan_delay,
    ASYNC_FAST_THRESHOLD_S, ASYNC_INLINE_MAX_BYTES,
)
from agentknit.tool_library import t_read, t_write
from agentknit._core import (
    init_session, run_turn, run_repl,
    _save_messages_snapshot,
    create_client,
    DEFAULT_ENDPOINT,
    DIM, RESET,
)

_TOOLS = [
    Tool(
        "execute_shell_command",
        f"Start a shell command asynchronously. If it finishes within "
        f"{int(ASYNC_FAST_THRESHOLD_S * 1000)} ms and both stdout and stderr "
        f"are under {ASYNC_INLINE_MAX_BYTES} bytes, the output is inlined "
        f"(fields: stdout, stderr, completed, returncode, duration_time). "
        f"Otherwise returns tool_exec_id and local file paths only.",
        t_execute_async,
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run."},
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
        "plan_shell_command",
        "Wait `when` minutes, then return control to the LLM so it can re-plan "
        "and check on running commands. Use this to avoid busy-polling.",
        t_plan_delay,
        parameters={
            "type": "object",
            "properties": {
                "when": {
                    "type": "integer",
                    "description": "Number of minutes to wait before re-planning.",
                },
            },
            "required": ["when"],
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
    "You are an async coding agent. Start shell commands with "
    "execute_shell_command — they run in the background. Use query_tool_exec "
    "to poll status. Use plan_shell_command(when) to wait N minutes before "
    "re-checking long-running commands."
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

    client = create_client(schema)
    session = init_session(
        schema,
        system_prompt_supplement=_SYSTEM_SUPPLEMENT,
        resumed_from=args.session,
    )

    tool_names = [t["function"]["name"] for t in _TOOL_SCHEMA]
    print(f"{DIM}Model: {args.model}  |  tools: {', '.join(tool_names)}{RESET}\n")
    print(f"{DIM}Session: {session['session_id']}{RESET}\n")

    if args.task:
        try:
            run_turn(client, args.model, session, " ".join(args.task))
        finally:
            _save_messages_snapshot(session)
        return

    run_repl(schema, session_id=args.session, system_prompt_supplement=_SYSTEM_SUPPLEMENT)


if __name__ == "__main__":
    main()
