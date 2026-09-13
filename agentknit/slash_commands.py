"""
Slash-command dispatch for the agentknit REPL.

Provides a registry and built-in commands:

* ``/clear``   – reset session context (keep system prompt)
* ``/compact`` – summarize older history into a compact continuation summary
* ``/model``   – list / switch models (queries the endpoint's ``/models`` endpoint)
* ``/usage``   – show token usage for the current session
* ``/tool``    – list / activate / remove tools at runtime
* ``/c``       – retry an interrupted turn without adding a user message

Commands are intercepted in the REPL loop before the input is sent to the model.

Also provides :func:`t_slash_command`, a tool function that exposes all slash
commands to the LLM as a single structured tool call, and
:data:`SLASH_COMMAND_TOOL`, a ready-made :class:`~agentknit.tool.Tool` object
that agents can include in their tool list.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import json
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from .openai_compat import OpenAI, SubprocessOpenAI

if TYPE_CHECKING:
    from ._core import Session

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
    handler: Callable[["Session", Any, str, str], None]


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
                 session: Session,
                 client: OpenAI | SubprocessOpenAI,
                 model: str,
                 *,
                 on_continue: Callable[[], None] | None = None) -> bool:
        """Parse *line* for a slash command and run it if found.

        Returns ``True`` if a command was handled (caller should skip model
        invocation), ``False`` if *line* is not a slash command.

        Some commands (``/c``) print nothing; they ask for the interrupted
        turn to be retried by setting ``session["_continue_requested"]``.
        Pass *on_continue* to have that request resolved right here: it is
        invoked, and the flag cleared, whenever a handler sets it — so
        callers don't need to know the flag exists. Without *on_continue*
        the flag is left untouched for the caller to check itself.
        """
        stripped = line.strip()
        if not stripped.startswith("/"):
            return False

        parts = stripped[1:].split(None, 1)  # split off command name
        if not parts:  # bare "/" — a common typo for /help, not a crash
            print(f"{RED}Empty command. Type /help for available commands.{RESET}")
            return True
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

        if on_continue is not None and session.pop("_continue_requested", False):
            on_continue()
        return True

    def help_text(self) -> str:
        """Return a formatted list of available commands."""
        lines = [f"{BOLD}Available slash commands:{RESET}"]
        for name in sorted(self._commands):
            cmd = self._commands[name]
            lines.append(f"  /{name:<12}  {cmd.description}")
        return "\n".join(lines)


# ── built-in command handlers ─────────────────────────────────────────────────

def _handle_hooks(session: Session, client: Any, model: str, args: str) -> None:
    """List configured hooks: event, matcher, handler, source."""
    from .hooks import HOOKS_NEVER_FIRE

    entries = session.get("hooks") or []
    if not entries:
        print("No hooks configured. Add ~/.agentknit/hooks.json, "
              "<git-root>/.agentknit/hooks.json, spec behaviour.hooks, "
              "or register_hook().")
        return
    if not session.get("hooks_enabled", True):
        print(f"{YEL}hooks are disabled for this session (hooks_enabled=false){RESET}")
    by_event: dict[str, list[Any]] = {}
    for e in entries:
        by_event.setdefault(e.event, []).append(e)
    for event in sorted(by_event):
        marker = f" {DIM}(never fires: no such lifecycle point yet){RESET}" \
            if event in HOOKS_NEVER_FIRE else ""
        print(f"{BOLD}{event}{RESET}{marker}")
        for e in by_event[event]:
            print(f"  {e.matcher or '*':<24} {e.handler.describe()}"
                  f"  {DIM}[{e.source}]{RESET}")
    state = session.get("_hook_state") or {}
    pending = len(state.get("pending_context") or [])
    queued = len(state.get("async_results") or [])
    if pending or queued:
        print(f"{DIM}pending context: {pending}, queued async results: {queued}{RESET}")


def _handle_clear(session: Session, client: Any, model: str, args: str) -> None:
    """Reset the session message history, keeping only the system prompt."""
    # Keep the first message (the system prompt).
    system_msgs = [m for m in session["messages"] if m.get("role") == "system"]
    if not system_msgs:
        # Fallback: keep at least the first message.
        system_msgs = [session["messages"][0]] if session["messages"] else []
    session["messages"] = system_msgs
    journal = session.get("_journal")
    if journal is not None:
        record = {"type": "reset_messages", "reason": "clear",
                  "messages": list(session["messages"])}
        journal.append(record)
        sink = session.get("durable_sink")
        if sink is not None and sink is not journal:
            sink.append(dict(record))
    # Reset usage totals.
    session["usage_totals"] = {"prompt": 0, "completion": 0, "total": 0,
                               "cached": 0, "cache_write": 0}
    # Reset compaction watermark so the next growth cycle can trigger again.
    session["compaction_last_prompt_tokens"] = 0
    # SessionEnd hooks fire for /clear with reason="clear" (advisory).
    from ._core import _fire_session_end
    _fire_session_end(session, "clear")
    print(f"{GREEN}Context cleared. Session history has been reset.{RESET}")


def _handle_compact(session: Session, client: Any, model: str, args: str) -> None:
    """Compact the session history into a continuation summary now."""
    from ._core import compact_session

    before = len(session["messages"])
    if compact_session(client, session.get("model", model), session):
        after = len(session["messages"])
        print(f"{GREEN}Context compacted: {before} → {after} messages.{RESET}")
    else:
        print(f"{YEL}Nothing to compact.{RESET}")


def _fetch_models_from_endpoint(endpoint: str, api_key: str) -> list[dict[str, object]]:
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


def _get_endpoint_and_key(client: Any, session: Session) -> tuple[str | None, str | None]:
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


def _handle_model(session: Session, client: Any, model: str, args: str) -> None:
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


def _tool_name(tool_spec: dict[str, Any]) -> str:
    """Model-facing name of a tool spec (function or custom shape)."""
    if tool_spec.get("type") == "custom":
        return str(tool_spec.get("name", ""))
    fn = tool_spec.get("function") or tool_spec
    return str(fn.get("name", ""))


def _tool_description(tool_spec: dict[str, Any]) -> str:
    """One-line description of a tool spec (function or custom shape)."""
    if tool_spec.get("type") == "custom":
        return str(tool_spec.get("description", ""))
    fn = tool_spec.get("function") or tool_spec
    return str(fn.get("description", ""))


def _library_tool_specs() -> dict[str, dict[str, Any]]:
    """Model-facing name → parsed ``Tool spec:`` docstring, for TOOL_LIBRARY."""
    from . import tool_library as _tool_library_module
    from ._tool_spec import extract_tool_specs_from_module
    specs: dict[str, dict[str, Any]] = {}
    for fn_name, s in extract_tool_specs_from_module(_tool_library_module).items():
        if isinstance(s.get("name"), str) and s["name"]:
            s["_function_name"] = fn_name
            specs[s["name"]] = s
    return specs


def _spec_from_docstring(doc_spec: dict[str, Any],
                         session: Session) -> dict[str, Any] | None:
    """Build an OpenAI function spec from a parsed ``Tool spec:`` docstring.

    The dispatch entry (``python_function`` + identity ``param_map``) is
    written into ``session["tool_dispatch"]`` as a side effect.
    """
    name = str(doc_spec.get("name", ""))
    fn_name = str(doc_spec.get("_function_name", ""))
    if not name or not fn_name:
        return None

    props: dict[str, Any] = {}
    required: list[str] = []
    for pname, pdef in (doc_spec.get("parameters") or {}).items():
        prop: dict[str, Any] = {"type": pdef.get("type", "string")}
        if pdef.get("description"):
            prop["description"] = pdef["description"]
        props[pname] = prop
        required.append(pname)
    session["tool_dispatch"][name] = {"python_function": fn_name,
                                      "param_map": {}}
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": doc_spec.get("description", name),
            "parameters": {"type": "object", "properties": props,
                           "required": required},
        },
    }


def _handle_tool(session: Session, client: Any, model: str, args: str) -> None:
    """List, activate or remove the session's tools at runtime.

    * ``/tool`` or ``/tool list`` — show active tools (✓), the inactive ones
      parked by a previous ``/tool remove`` (✗), and other TOOL_LIBRARY
      functions that could be activated.
    * ``/tool activate <tool_name>`` — (re)add a tool to the session:
      an inactive tool is restored verbatim; a TOOL_LIBRARY function
      (docstring spec, or plain signature) gets a spec built on the fly;
      a legacy alias (e.g. ``execute_shell_command``) is expanded.
    * ``/tool remove <tool_name>`` — drop a tool from the session (parked in
      ``session["_removed_tools"]`` so ``/tool activate`` can restore it).
    """
    from ._core import _LEGACY_TOOL_ALIASES, _log
    from .tool_library import TOOL_LIBRARY

    parts = args.split(None, 1)
    sub = (parts[0].lower() if parts else "list")
    name = parts[1].strip() if len(parts) > 1 else ""

    tools: list[dict[str, Any]] = session["tools"]
    # Inactive tools: specs (and dispatch entries) parked by a previous
    # /tool remove, so they can be re-activated later verbatim.
    parked_tools: dict[str, dict[str, Any]] = dict(session.get("_removed_tools") or {})
    parked_dispatch: dict[str, dict[str, Any]] = dict(
        session.get("_removed_dispatch") or {})
    inactive: dict[str, dict[str, Any]] = {}
    for parked_spec in parked_tools.values():
        n = _tool_name(parked_spec)
        if n:
            inactive[n] = parked_spec

    def _active_names() -> list[str]:
        return [_tool_name(t) for t in tools]

    if sub in ("", "list", "ls"):
        active = _active_names()
        print(f"{BOLD}Active tools ({len(active)}):{RESET}")
        for t in tools:
            desc = _tool_description(t).split("\n")[0]
            suffix = f"  {DIM}{desc[:70]}{RESET}" if desc else ""
            print(f"  {GREEN}✓{RESET} {_tool_name(t)}{suffix}")
        if inactive:
            print(f"{BOLD}Inactive (removed, re-activatable):{RESET}")
            for n in sorted(inactive):
                print(f"  {RED}✗{RESET} {n}   {DIM}/tool activate {n}{RESET}")
        # TOOL_LIBRARY functions not yet advertised: candidates for activate.
        from .tool_library import _ASK_USER_FNS
        lib = _library_tool_specs()
        by_fn = {str(s.get("_function_name")): s for s in lib.values()}
        known = set(active) | set(inactive)
        candidates: list[str] = []
        for fn_name, fn in TOOL_LIBRARY.items():
            if session.get("non_interactive") and fn_name in _ASK_USER_FNS:
                continue
            model_name = str((by_fn.get(fn_name) or {}).get("name")
                             or (fn_name[2:] if fn_name.startswith("t_") else fn_name))
            if model_name not in known and model_name not in candidates:
                candidates.append(model_name)
        if candidates:
            print(f"{BOLD}Available in TOOL_LIBRARY (not active):{RESET}")
            for n in sorted(candidates):
                print(f"  {DIM}-{RESET} {n}   {DIM}/tool activate {n}{RESET}")
        print(f"\n{DIM}Usage: /tool list | /tool activate <name> | /tool remove <name>{RESET}")
        return

    if not name:
        print(f"{RED}Usage: /tool {sub} <tool_name>{RESET}")
        return

    if sub == "activate":
        # Legacy aliases (e.g. execute_shell_command) resolve to their
        # canonical name; the alias itself keeps working through the
        # dispatch-only entry installed at session start.
        name = _LEGACY_TOOL_ALIASES.get(name, name)
        if name in _active_names():
            pass  # already active — idempotent no-op
        else:
            spec: dict[str, Any] | None = None
            # 1. A parked spec from a previous /tool remove — restore both
            #    the schema and the parked dispatch entry verbatim.
            if name in inactive:
                spec = inactive[name]
                for dispatch_name, entry in parked_dispatch.items():
                    if dispatch_name == name or _LEGACY_TOOL_ALIASES.get(dispatch_name) == name:
                        session["tool_dispatch"][dispatch_name] = entry
            # 2. A TOOL_LIBRARY function advertised by its docstring spec.
            if spec is None:
                doc_spec = _library_tool_specs().get(name)
                if doc_spec is not None:
                    spec = _spec_from_docstring(doc_spec, session)
            # 3. Any other TOOL_LIBRARY function — schema inferred from its
            #    signature via Tool/build_tool_spec.
            if spec is None and name in TOOL_LIBRARY:
                from .tool import Tool as _T, build_tool_spec as _bts
                fn = TOOL_LIBRARY[name]
                model_name = name[2:] if name.startswith("t_") else name
                doc = (inspect.getdoc(fn) or model_name).split("\n\nTool spec:")[0]
                desc = next((ln.strip() for ln in doc.splitlines() if ln.strip()),
                             model_name)
                schema_list, disp = _bts([_T(model_name, desc, fn)])
                session["tool_dispatch"].update(disp)
                spec = schema_list[0]
            if spec is None:
                known_names: set[str] = set(TOOL_LIBRARY) | set(_library_tool_specs())
                known_names |= set(_active_names()) | set(inactive)
                known_list = sorted(known_names)
                print(f"{RED}Unknown tool '{name}'. Known: "
                      f"{', '.join(known_list) or '(none)'}{RESET}")
                return
            tools.append(spec)
        # Drop the parked copies of the same name (it is active again now).
        session["_removed_tools"] = {
            k: v for k, v in parked_tools.items() if _tool_name(v) != name
        }
        session["_removed_dispatch"] = {
            k: v for k, v in parked_dispatch.items()
            if k != name and _LEGACY_TOOL_ALIASES.get(k) != name
        }
        _log(session, {"type": "tool_activated", "tool": name})
        print(f"{GREEN}Tool activated: {name}{RESET}")
        print(f"{DIM}The next turn will offer it to the model.{RESET}")
        return

    if sub == "remove":
        active_names = _active_names()
        if name not in active_names:
            print(f"{RED}Tool '{name}' is not active. Active: "
                  f"{', '.join(active_names) or '(none)'}{RESET}")
            return
        removed_spec = tools.pop(active_names.index(name))
        session.setdefault("_removed_tools", {})[name] = removed_spec
        # Also retire the dispatch entry (and any alias pointing at it), so
        # the model cannot call a tool it can no longer see; parked entries
        # are restored verbatim on the next activate.
        _aliases = [a for a, c in _LEGACY_TOOL_ALIASES.items() if c == name]
        for dispatch_name in [name, *_aliases]:
            parked_entry = session["tool_dispatch"].pop(dispatch_name, None)
            if parked_entry is not None:
                session.setdefault("_removed_dispatch", {})[dispatch_name] = parked_entry
        _log(session, {"type": "tool_removed", "tool": name})
        print(f"{GREEN}Tool removed: {name}{RESET}")
        print(f"{DIM}Re-activate later with /tool activate {name}.{RESET}")
        return

    print(f"{RED}Unknown /tool sub-command '{sub}'. "
          f"Usage: /tool list | /tool activate <name> | /tool remove <name>{RESET}")


def _handle_usage(session: Session, client: Any, model: str, args: str) -> None:
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


def _handle_help(session: Session, client: Any, model: str, args: str) -> None:
    """Show available slash commands."""
    print(REGISTRY.help_text())


def _handle_continue(session: Session, client: Any, model: str, args: str) -> None:
    """Request a retry of the current turn without changing its transcript."""
    session["_continue_requested"] = True


# ── global registry ───────────────────────────────────────────────────────────

REGISTRY = SlashCommandRegistry()

# Register built-in commands.
REGISTRY.register(SlashCommand(
    name="c",
    description="Retry an interrupted turn without adding a user message.",
    handler=_handle_continue,
))
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
    name="tool",
    description="List / activate / remove tools at runtime: "
                "/tool list | /tool activate <name> | /tool remove <name>.",
    handler=_handle_tool,
))
REGISTRY.register(SlashCommand(
    name="hooks",
    description="List configured lifecycle hooks and their sources.",
    handler=_handle_hooks,
))
REGISTRY.register(SlashCommand(
    name="help",
    description="Show this help message.",
    handler=_handle_help,
))

# ── LLM-callable tool ─────────────────────────────────────────────────────────

# Shared context populated by the agent at startup so t_slash_command can
# forward calls to handlers that need session + client.
slash_tool_ctx: dict[str, object] = {"session": None, "client": None, "model": None}

_HANDLERS: dict[str, Callable[..., object]] = {
    "clear":   _handle_clear,
    "compact": _handle_compact,
    "model":   _handle_model,
    "usage":   _handle_usage,
    "hooks":   _handle_hooks,
    "tool":    _handle_tool,
    "help":    _handle_help,
}


def t_slash_command(command: str, args: str = "") -> tuple[str, dict[str, object]]:
    """Run a slash command and return its output as a tool result.

    command must be one of: clear, compact, model, usage, hooks, tool, help.
    For 'model', pass a model-id in args to switch; omit to list.
    For 'tool', args is one of: 'list', 'activate <tool_name>',
    'remove <tool_name>'.

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
    "Run a slash command. command: one of clear, compact, model, usage, hooks, tool, help. "
    "For 'model', pass a model-id in args to switch; omit args to list. "
    "For 'tool', args is one of: 'list', 'activate <tool_name>', 'remove <tool_name>'.",
    t_slash_command,
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "enum": ["clear", "compact", "model", "usage", "hooks", "tool", "help"],
                "description": "Slash command to run.",
            },
            "args": {
                "type": "string",
                "description": ("Optional argument (e.g. model-id for 'model', "
                                "'activate read_file' for 'tool')."),
            },
        },
        "required": ["command"],
    },
)
