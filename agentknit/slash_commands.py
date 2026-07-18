"""
Slash-command dispatch for the agentknit REPL.

Provides a registry and built-in commands:

* ``/clear``   – reset session context (keep system prompt)
* ``/compact`` – summarize older history into a compact continuation summary
* ``/model``   – list / switch models (queries the endpoint's ``/models`` endpoint)
* ``/usage``   – show token usage for the current session

Commands are intercepted in the REPL loop before the input is sent to the model.

Also provides :func:`t_slash_command`, a tool function that exposes all slash
commands to the LLM as a single structured tool call, and
:data:`SLASH_COMMAND_TOOL`, a ready-made :class:`~agentknit.tool.Tool` object
that agents can include in their tool list.
"""

from __future__ import annotations

import contextlib
import io
import json
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from .openai_compat import OpenAI, SubprocessOpenAI

# ── colour helpers (same palette as _core.py) ─────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YEL = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"
MAG = "\033[35m"


# ── command registration ──────────────────────────────────────────────────────

@dataclass
class SlashCommand:
    """A slash command that can be registered and dispatched in the REPL.

    Attributes
    ----------
    name
        Command name without the leading slash, e.g. ``"clear"``.
    description
        One-line help text shown in ``/help``.
    handler
        Coroutine that will be called when the command is invoked:
        ``handler(session, client, model, args)``.
        It should print its output directly (it runs in the REPL thread).
    """

    name: str
    description: str
    handler: Callable[[dict, Any, str, str], None]


class SlashCommandRegistry:
    """Manages registration and dispatch of slash commands."""

    def __init__(self) -> None:
        self._commands: dict[str, SlashCommand] = {}

    def register(self, cmd: SlashCommand) -> None:
        """Register a new command."""
        self._commands[cmd.name] = cmd

    def unregister(self, name: str) -> None:
        """Remove a previously registered command."""
        self._commands.pop(name, None)

    def dispatch(self, line: str,
                 session: dict,
                 client: OpenAI | SubprocessOpenAI,
                 model: str) -> bool:
        """Parse *line* for a slash command and run it if found.

        Returns ``True`` if a command was handled (caller should skip model
        invocation), ``False`` if *line* is not a slash command.
        """
        stripped = line.strip()
        if not stripped.startswith("/"):
            return False

        parts = stripped[1:].split(None, 1)  # split off command name
        cmd_name = parts[0].lower()
        cmd_args = parts[1] if len(parts) > 1 else ""

        cmd = self._commands.get(cmd_name)
        if cmd is None:
            print(f"{RED}Unknown command: /{cmd_name}. "
                  f"Type /help for available commands.{RESET}")
            return True

        try:
            cmd.handler(session, client, model, cmd_args)
        except Exception as exc:
            print(f"{RED}Error running /{cmd_name}: {exc}{RESET}")
        return True

    def help_text(self) -> str:
        """Return a formatted list of available commands."""
        lines = [f"{BOLD}Available slash commands:{RESET}"]
        for name in sorted(self._commands):
            cmd = self._commands[name]
            lines.append(f"  /{name:<12}  {cmd.description}")
        return "\n".join(lines)


# ── built-in command handlers ─────────────────────────────────────────────────

def _handle_clear(session: dict, client: Any, model: str, args: str) -> None:
    """Reset the session message history, keeping only the system prompt."""
    # Keep the first message (the system prompt).
    system_msgs = [m for m in session["messages"] if m.get("role") == "system"]
    if not system_msgs:
        # Fallback: keep at least the first message.
        system_msgs = [session["messages"][0]] if session["messages"] else []
    session["messages"] = system_msgs
    # Reset usage totals.
    session["usage_totals"] = {"prompt": 0, "completion": 0, "total": 0,
                               "cached": 0, "cache_write": 0}
    # Reset compaction watermark so the next growth cycle can trigger again.
    session["compaction_last_prompt_tokens"] = 0
    print(f"{GREEN}Context cleared. Session history has been reset.{RESET}")


def _handle_compact(session: dict, client: Any, model: str, args: str) -> None:
    """Compact the session history into a continuation summary now."""
    from ._core import compact_session

    before = len(session["messages"])
    if compact_session(client, session.get("model", model), session):
        after = len(session["messages"])
        print(f"{GREEN}Context compacted: {before} → {after} messages.{RESET}")
    else:
        print(f"{YEL}Nothing to compact.{RESET}")


def _fetch_models_from_endpoint(endpoint: str, api_key: str) -> list[dict]:
    """Query the ``/models`` endpoint and return the list of model objects.

    Works for any OpenAI-compatible API that exposes GET ``/models``
    (e.g. OpenRouter, local LLM servers).
    """
    base = endpoint.rstrip("/")
    # If the endpoint already ends with a path like /v1, replace /v1 with
    # nothing before appending /models; otherwise just append.
    models_url = base
    if models_url.endswith("/v1"):
        models_url = models_url[:-3]
    if not models_url.endswith("/models"):
        models_url = models_url.rstrip("/") + "/models"

    req = urllib.request.Request(models_url)
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("User-Agent", "agentknit/1.0")

    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode())

    # OpenAI-compatible /models returns {"data": [{"id": "...", ...}]}.
    # Some providers return a flat list instead.
    raw = data if isinstance(data, list) else data.get("data", [])
    return raw


def _get_endpoint_and_key(client: Any, session: dict) -> tuple[str | None, str | None]:
    """Extract the endpoint URL and API key from the client or session."""
    # Prefer the endpoint stored in the session.
    endpoint = session.get("endpoint")
    if not endpoint:
        # Try the client's internal base URL.
        if hasattr(client, '_base_url'):
            endpoint = client._base_url
        elif hasattr(client, '_client') and hasattr(client._client, '_base_url'):
            endpoint = client._client._base_url

    # Get API key.
    api_key = getattr(client, '_api_key', None)
    if api_key is None and hasattr(client, '_client'):
        api_key = getattr(client._client, '_api_key', None)

    return endpoint or None, api_key


def _handle_model(session: dict, client: Any, model: str, args: str) -> None:
    """List available models or switch to a different model."""
    from ._core import _parse_run_uri

    args = args.strip()

    # If no arguments, list models from the endpoint.
    if not args:
        endpoint, api_key = _get_endpoint_and_key(client, session)
        if not endpoint:
            print(f"{RED}Cannot determine endpoint URL to query /models.{RESET}")
            return

        # Don't query /models for subprocess backends.
        if _parse_run_uri(endpoint):
            print(f"{YEL}/models is not available for subprocess backends.{RESET}")
            return

        try:
            models_raw = _fetch_models_from_endpoint(endpoint, api_key or "")
        except Exception as exc:
            print(f"{RED}Failed to fetch models from endpoint: {exc}{RESET}")
            print(f"{DIM}The endpoint may not support GET /models.{RESET}")
            return

        if not models_raw:
            print(f"{YEL}No models returned by the endpoint.{RESET}")
            return

        # Display models.
        current = session.get("model", model)
        print(f"{BOLD}Available models ({len(models_raw)}):{RESET}")
        for m in models_raw:
            mid = m.get("id") or m.get("name") or str(m)
            prefix = f"{GREEN}*{RESET} " if mid == current else "  "
            print(f"  {prefix}{mid}")

        print(f"\n{DIM}To switch: /model <model-id>{RESET}")
        return

    # An argument was provided — switch to that model.
    new_model = args
    old_model = session.get("model", model)
    session["model"] = new_model
    print(f"{GREEN}Model switched from {old_model} → {new_model}{RESET}")
    print(f"{DIM}The next turn will use the new model.{RESET}")


def _handle_usage(session: dict, client: Any, model: str, args: str) -> None:
    """Display token usage for the current session."""
    t = session.get("usage_totals", {})
    prompt = t.get("prompt", 0)
    completion = t.get("completion", 0)
    total = t.get("total", 0)
    cached = t.get("cached", 0)
    cache_write = t.get("cache_write", 0)

    session_id = session.get("session_id", "unknown")
    parts = [
        f"{BOLD}Session token usage:{RESET}",
        f"  trajectory: {session_id}",
        f"  prompt:     {prompt:>10,} tokens",
    ]
    if cached:
        pct = (cached / prompt * 100) if prompt else 0
        parts.append(f"    └─ cached: {cached:>9,} ({pct:.0f}%)")
    if cache_write:
        parts.append(f"    └─ cache-write: {cache_write:>6,}")
    parts.append(f"  completion: {completion:>10,} tokens")
    parts.append(f"  {CYAN}total:      {total:>10,} tokens{RESET}")

    # Also show message count.
    msg_count = len([m for m in session.get("messages", [])
                     if m.get("role") != "system"])
    parts.append(f"  messages:   {msg_count:>10,} (excl. system)")

    print("\n".join(parts))


def _handle_help(session: dict, client: Any, model: str, args: str) -> None:
    """Show available slash commands."""
    print(REGISTRY.help_text())


# ── global registry ───────────────────────────────────────────────────────────

REGISTRY = SlashCommandRegistry()

# Register built-in commands.
REGISTRY.register(SlashCommand(
    name="clear",
    description="Reset the session message history (keep system prompt).",
    handler=_handle_clear,
))
REGISTRY.register(SlashCommand(
    name="compact",
    description="Summarize older history into a compact continuation summary.",
    handler=_handle_compact,
))
REGISTRY.register(SlashCommand(
    name="model",
    description="List available models or switch: /model <model-id>.",
    handler=_handle_model,
))
REGISTRY.register(SlashCommand(
    name="usage",
    description="Show token usage for the current session.",
    handler=_handle_usage,
))
REGISTRY.register(SlashCommand(
    name="help",
    description="Show this help message.",
    handler=_handle_help,
))

# ── LLM-callable tool ─────────────────────────────────────────────────────────

# Shared context populated by the agent at startup so t_slash_command can
# forward calls to handlers that need session + client.
slash_tool_ctx: dict = {"session": None, "client": None, "model": None}

_HANDLERS: dict[str, Callable] = {
    "clear":   _handle_clear,
    "compact": _handle_compact,
    "model":   _handle_model,
    "usage":   _handle_usage,
    "help":    _handle_help,
}


def t_slash_command(command: str, args: str = "") -> tuple[str, dict]:
    """Run a slash command and return its output as a tool result.

    command must be one of: clear, compact, model, usage, help.
    For 'model', pass a model-id in args to switch; omit to list.

    Populate :data:`slash_tool_ctx` with the live session, client, and model
    name before registering this tool in an agent.
    """
    handler = _HANDLERS.get(command)
    if handler is None:
        r = f"ERROR: unknown command '{command}'. Valid: {', '.join(_HANDLERS)}"
        return r, {"result": r}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        handler(slash_tool_ctx["session"], slash_tool_ctx["client"],
                slash_tool_ctx["model"], args)
    out = buf.getvalue().strip()
    return out, {"result": out}


# Register in TOOL_LIBRARY so the dispatch mechanism can find it by name.
from .tool_library import TOOL_LIBRARY as _TOOL_LIBRARY  # noqa: E402
_TOOL_LIBRARY["t_slash_command"] = t_slash_command

# Ready-made Tool object: import and add to your _TOOLS list.
from .tool import Tool as _Tool  # noqa: E402

SLASH_COMMAND_TOOL = _Tool(
    "slash_command",
    "Run a slash command. command: one of clear, compact, model, usage, help. "
    "For 'model', pass a model-id in args to switch; omit args to list.",
    t_slash_command,
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "enum": ["clear", "compact", "model", "usage", "help"],
                "description": "Slash command to run.",
            },
            "args": {
                "type": "string",
                "description": "Optional argument (e.g. model-id for 'model').",
            },
        },
        "required": ["command"],
    },
)
