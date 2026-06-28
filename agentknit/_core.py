#!/usr/bin/env python3
"""
Generic CLI coding agent for any Completions API endpoint.

Run with --help for usage information.

Aliases
───────
The probe JSON may contain an optional top-level `aliases` table:

  "aliases": {
    "execute": "execute_command",
    "run":     "execute_command"
  }

Each entry maps an alias name to an existing tool name in `tool_dispatch`.
At session start both the tool schema (for structured mode) and the dispatch
table are expanded so the alias behaves identically to the canonical tool.
An alias that already has its own `tool_dispatch` entry is left untouched.

Options
───────
The probe JSON may contain an optional top-level `options` array of strings
that modify agent behaviour. Currently supported options:

  "exclude-prompt_cache_key"
    Skip sending the `prompt_cache_key` field in the extra_body of API
    requests. Use this for providers (e.g. NVIDIA NIM) that reject unknown
    extra_body fields. The `user` field (which also carries the cache key)
    is still sent as usual.

No hardcoded provider URLs should be added to the framework code; use the
options mechanism in the agent spec JSON instead.

Events
──────
The framework emits typed events that consumers can subscribe to via the
:func:`subscribe` function (or its alias :func:`on`).  The full list of event
types is documented in the README and in :func:`subscribe`'s docstring.

Event types
~~~~~~~~~~~

``tool_call``
    Before dispatching a tool.  Data: ``name``, ``args``, ``fmt``.
``tool_result``
    After receiving tool result.  Data: ``name``, ``result``, ``streamed``, ``fmt``.
``content_delta``
    Streaming text chunk from the model.  Data: ``text``, ``first``, ``no_newline``, ``fmt``.
``reasoning_delta``
    Streaming reasoning trace from the model.  Data: ``text``, ``first``, ``no_newline``, ``fmt``.
``content_stream_end``
    End of a streaming content sequence.  Data: ``no_newline``, ``fmt``.
``reasoning_stream_end``
    End of a streaming reasoning sequence.  Data: ``no_newline``, ``fmt``.
``usage``
    Per-turn token usage report.  Data: ``prompt``, ``completion``, ``total``,
    ``cached``, ``cache_write``, ``fmt``.
``session_usage``
    Cumulative session usage emitted alongside the final answer.  Data:
    ``prompt``, ``completion``, ``total``, ``cached``, ``cache_write``, ``fmt``.
``error``
    API or dispatch error.  Data: ``text``, ``fmt``.
``final_answer``
    The agent produced its final reply.  Data: ``text``, ``fmt``.
``token_limit``
    The token budget was exceeded.  Data: ``used``, ``limit``, ``fmt``.
``session_resumed``
    Session history was loaded from disk (or not found).  Data: ``session_id``,
    ``messages_loaded``, ``source_model`` (optional), ``fmt``.
``provider_pinned``
    OpenRouter provider was locked for the remainder of the session.  Data:
    ``provider``, ``fmt``.

Every data dict contains a ``\"fmt\"`` key with a pre-formatted ANSI string.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import datetime
import json
import os
import queue
import re
import readline  # noqa: F401 — enables arrow keys / history in input()
import select
import signal
import sys
import threading
import urllib.request
import uuid
from pathlib import Path
from typing import Callable

from . import openai_compat as openai
from .openai_compat import SubprocessOpenAI

from . import tool_library as _tool_module
from .tool_library import TOOL_LIBRARY, _ASK_USER_FNS
from .exceptions import (
    AgentSpecDisabledError, AgentSpecInvalidError,
    PricingLimitExceededError, AuthenticationError, CacheProofError,
)
from .slash_commands import REGISTRY as _slash_registry


DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1"
DEFAULT_MAX_TOKENS = 3_000_000
LOG_BASE = Path.home() / ".local" / "share" / "agent_probe"

DEFAULT_COMPACTION_TRIGGER_TOKENS = 100_000
DEFAULT_COMPACTION_TARGET_TOKENS = 20_000
DEFAULT_COMPACTION_KEEP_LAST_TURNS = 2

_COMPACTION_PROMPT = (
    "Summarize the conversation above into a dense, structured summary "
    "optimized for continuing a coding task. Preserve all state needed to "
    "keep working without re-reading files.\n\n"
    "Preserve:\n"
    "- The current objective and any user constraints\n"
    "- Files that have been touched and what changes were made\n"
    "- Commands or tools used and their key outcomes\n"
    "- Errors, failing tests, or build failures\n"
    "- Failed hypotheses or dead ends already explored\n"
    "- Unresolved issues or blockers\n"
    "- The immediate next step if one was identified\n\n"
    "Avoid:\n"
    "- Conversational filler or chatter\n"
    "- Repeated log output (summarize outcomes, don't quote logs verbatim)\n"
    "- Redundant observations\n"
    "- Rewriting uncertainty as certainty\n\n"
    "Format the summary as plain text with clear sections. Be concise but complete."
)

BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YEL = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"
MAG = "\033[35m"

# readline-safe versions: \x01/\x02 mark zero-width sequences so readline
# computes line length correctly and doesn't corrupt long input lines.
RL_BOLD  = "\x01\033[1m\x02"
RL_RESET = "\x01\033[0m\x02"
PASTE_IDLE_TIMEOUT_S = 0.25

# ── Ctrl-C handling ───────────────────────────────────────────────────────────
# True while run_turn() is executing; False at the REPL prompt.
_in_turn: bool = False


def _sigint_handler(sig: int, frame: object) -> None:
    """SIGINT handler: kill the active subprocess and abort the turn.

    When the agent is executing a tool (run_turn is active), immediately
    SIGKILL the current subprocess (if any) then raise KeyboardInterrupt so
    run_turn unwinds back to the REPL.  When idle at the prompt, do nothing.
    """
    if not _in_turn:
        return
    proc = _tool_module._active_proc
    if proc is not None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    raise KeyboardInterrupt()


signal.signal(signal.SIGINT, _sigint_handler)


# ── event system ──────────────────────────────────────────────────────────────

# EventCallback(event_type, data) — the data dict always contains a "fmt" key
# with a pre-formatted ANSI string so simple handlers can just print it.
EventCallback = Callable[[str, dict], None]


def _default_event_handler(event_type: str, data: dict) -> None:
    """Print the pre-formatted ANSI string from *data["fmt"]* to stdout/stderr.

    Events with ``no_newline=True`` are printed without a trailing newline
    (used for streaming deltas so the cursor stays on the same line).
    """
    fmt = data.get("fmt")
    if fmt is None:
        return
    if event_type == "token_limit":
        print(fmt, file=sys.stderr)
    elif data.get("no_newline"):
        print(fmt, end="", flush=True)
    else:
        print(fmt)


def _emit(session: dict, event_type: str, **data) -> None:
    """Fire *event_type* through the session's registered event handlers.

    First calls any per-event-type handlers registered via :func:`subscribe`,
    then calls the generic ``on_event`` handler (or the default).

    .. seealso::

        :ref:`event-types` — full list of event types with descriptions.
    """
    # Call per-event-type handlers first
    handlers = session.get("_event_handlers", {}).get(event_type, [])
    for handler in handlers:
        handler(event_type, data)
    # Then call the generic handler
    handler: EventCallback = session.get("on_event") or _default_event_handler
    handler(event_type, data)


def subscribe(session: dict, event_type: str, handler: EventCallback) -> None:
    """Register an event handler for a specific event type.

    The *handler* will be called with ``(event_type, data)`` whenever an event
    of that type is emitted.  Multiple handlers can be registered for the same
    type; they are called in registration order, before the generic
    ``on_event`` handler passed to :func:`init_session`.

    Example::

        session = init_session(schema)
        subscribe(session, "tool_call", lambda et, d: print(d["fmt"]))
        subscribe(session, "content_delta", lambda et, d: print(d["text"], end=""))

    .. seealso::

        :ref:`event-types` — full list of event types with descriptions.
    """
    if "_event_handlers" not in session:
        session["_event_handlers"] = {}
    session["_event_handlers"].setdefault(event_type, []).append(handler)


def unsubscribe(session: dict, event_type: str, handler: EventCallback) -> None:
    """Unregister a previously registered event handler.

    Does nothing if the *handler* was not registered for *event_type*.
    """
    handlers = session.get("_event_handlers", {}).get(event_type, [])
    if handler in handlers:
        handlers.remove(handler)


# Convenience alias
on = subscribe


# ── api key ──────────────────────────────────────────────────────────────────

def get_api_key() -> str:
    from .keys import ensure_api_key
    return ensure_api_key()


def _get_opencode_token() -> str:
    auth_json = Path.home() / ".local/share/opencode/auth.json"
    import json as _json
    creds = _json.loads(auth_json.read_text())
    token = creds.get("github-copilot", {}).get("access")
    if not token:
        raise AuthenticationError(
            f"No github-copilot.access token found in {auth_json}"
        )
    return token


def safe_model_name(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def _parse_run_uri(endpoint: str) -> str | None:
    """Return binary path if *endpoint* is a run:// URI, else None."""
    if endpoint.startswith("run://"):
        return endpoint[len("run://"):]
    return None


_DEFAULT_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read and return the contents of a file at the specified path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file."},
                    "offset": {"type": "integer", "description": "Line number to start reading from (1-indexed)."},
                    "limit": {"type": "integer", "description": "Maximum number of lines to read."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write (or overwrite) a file at the specified path with the given content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file."},
                    "content": {"type": "string", "description": "Content to write."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "str_replace",
            "description": "Edit an existing file by replacing a specific substring.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_str": {"type": "string"},
                    "new_str": {"type": "string"},
                },
                "required": ["path", "old_str", "new_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_shell_command",
            "description": "Execute a shell command and return its stdout, stderr, and exit code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                },
                "required": ["command"],
            },
        },
    },
]

_DEFAULT_TOOL_DISPATCH = {
    "read_file":            {"python_function": "t_read",        "param_map": {}},
    "write_file":           {"python_function": "t_write",       "param_map": {}},
    "str_replace":          {"python_function": "t_update",      "param_map": {"old_str": "old", "new_str": "new"}},
    "execute_shell_command":{"python_function": "t_run",         "param_map": {}},
}
_DEFAULT_TOOLS = [
    "t_read",
    "t_write",
    "t_update",
    "t_run",
]


# ── probe / cache ─────────────────────────────────────────────────────────────

def load_or_probe(model: str, endpoint: str, force: bool) -> dict:
    # Spec JSON files live in the project root (parent of the package directory).
    here = Path(__file__).resolve().parent.parent

    # run:// URI → subprocess binary; use cached spec or generate a default.
    # Accept run:// in either endpoint or model (CLI convenience).
    binary_path = _parse_run_uri(endpoint) or _parse_run_uri(model)
    if binary_path is not None:
        # Normalise: model gets the bare path, endpoint gets the run:// URI.
        run_endpoint = f"run://{binary_path}"
        if _parse_run_uri(model):
            model = binary_path
        endpoint = run_endpoint
        path = here / f"agent_spec_{safe_model_name(model)}.json"
        if path.exists() and not force:
            print(f"{DIM}Using cached spec at {path.name}{RESET}")
            with path.open() as f:
                return json.load(f)
        data = {
            "model":       model,
            "endpoint":    endpoint,
            "status":      "default",
            "tool_specs":  _DEFAULT_TOOL_SCHEMA,
            "tools":       _DEFAULT_TOOLS,
            "behaviour":   {"call_delivery_mode": "structured_tool_calls"},
        }
        with path.open("w") as f:
            json.dump(data, f, indent=2)
        print(f"{DIM}Generated default spec at {path.name}{RESET}")
        return data

    # Accept a direct path to a JSON schema file.
    if model.endswith(".json"):
        path = Path(model)
        if not path.is_absolute():
            path = here / path
        with path.open() as f:
            data = json.load(f)
        print(f"{DIM}Using schema file {path.name}{RESET}")
        return data

    path = here / f"agent_spec_{safe_model_name(model)}.json"
    if not path.exists():
        path = here / f"inferred_tool_schema_{safe_model_name(model)}.json"
    if not path.exists():
        path = here / f"tool_schema_{safe_model_name(model)}.json"

    if path.exists() and not force:
        print(f"{DIM}Using cached probe at {path.name}{RESET}")
        with path.open() as f:
            return json.load(f)

    if endpoint:
        data = {
            "model": model,
            "endpoint": endpoint,
            "status": "default",
            "tool_specs": _DEFAULT_TOOL_SCHEMA,
            "tools": _DEFAULT_TOOLS,
            "behaviour": {"call_delivery_mode": "structured_tool_calls"},
        }
        with path.open("w") as f:
            json.dump(data, f, indent=2)
        print(f"{DIM}Generated default spec at {path.name}{RESET}")
        return data

    raise AgentSpecInvalidError(
        f"No agent spec found for '{model}'. "
        f"Run `llmprobe {model}` (with --endpoint {endpoint} if needed) "
        f"to probe the model and generate a spec file.",
        model=model,
    )


# ── dispatch ──────────────────────────────────────────────────────────────────

class FatalToolDispatchError(RuntimeError):
    """Raised when the agent requests a tool that cannot be dispatched."""


def _tool_name_from_spec(tool_spec: dict) -> str:
    """Return the model-facing tool name from an OpenAI-compatible tool spec."""
    fn = tool_spec.get("function") or tool_spec
    return fn.get("name", "")


def _tool_param_names(tool_spec: dict) -> list[str]:
    """Return the model-facing parameter names from *tool_spec* in declaration order."""
    fn = tool_spec.get("function") or tool_spec
    params = ((fn.get("parameters") or {}).get("properties") or {})
    return list(params.keys())


def _callable_param_names(fn: Callable) -> list[str]:
    """Return positional/keyword parameter names for *fn* in signature order."""
    import inspect

    sig = inspect.signature(fn)
    result: list[str] = []
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        result.append(name)
    return result


def _derive_param_map(tool_spec: dict, fn_name: str) -> dict[str, str]:
    """Infer model-arg -> Python-kwarg mapping for *fn_name* and *tool_spec*."""
    fn = TOOL_LIBRARY.get(fn_name)
    if fn is None:
        raise AgentSpecInvalidError(
            f"Tool function {fn_name!r} not found in TOOL_LIBRARY.",
            model=fn_name,
        )

    spec_params = _tool_param_names(tool_spec)
    sig_params = _callable_param_names(fn)
    if len(spec_params) > len(sig_params):
        raise AgentSpecInvalidError(
            f"Tool spec for {_tool_name_from_spec(tool_spec)!r} declares "
            f"{len(spec_params)} params but {fn_name} only accepts {len(sig_params)}.",
            model=_tool_name_from_spec(tool_spec) or fn_name,
        )

    return {
        spec_name: sig_params[idx]
        for idx, spec_name in enumerate(spec_params)
        if spec_name != sig_params[idx]
    }


def _build_dispatch_from_tools(tool_specs: list[dict], tools: list[str]) -> dict[str, dict]:
    """Build a tool_dispatch dict from ordered tool specs and TOOL_LIBRARY names."""
    if len(tool_specs) != len(tools):
        raise AgentSpecInvalidError(
            f"'tool_specs' has {len(tool_specs)} entries but 'tools' has {len(tools)}.",
        )

    dispatch: dict[str, dict] = {}
    for tool_spec, fn_name in zip(tool_specs, tools, strict=False):
        if not isinstance(fn_name, str):
            raise AgentSpecInvalidError("'tools' must be a list of TOOL_LIBRARY function names.")
        if fn_name not in TOOL_LIBRARY:
            raise AgentSpecInvalidError(f"Unknown tool function {fn_name!r} in 'tools'.")
        tool_name = _tool_name_from_spec(tool_spec)
        if not tool_name:
            raise AgentSpecInvalidError("Every 'tool_specs' entry must define function.name.")
        dispatch[tool_name] = {
            "python_function": fn_name,
            "param_map": _derive_param_map(tool_spec, fn_name),
        }
    return dispatch


def _default_dispatch_for_tool_specs(tool_specs: list[dict]) -> dict[str, dict]:
    """Return built-in dispatch entries for tool specs whose names are known defaults."""
    dispatch: dict[str, dict] = {}
    for tool_spec in tool_specs:
        tool_name = _tool_name_from_spec(tool_spec)
        entry = _DEFAULT_TOOL_DISPATCH.get(tool_name)
        if entry is not None:
            dispatch[tool_name] = copy.deepcopy(entry)
    return dispatch


def _normalize_schema(schema: dict) -> dict:
    """Return a schema copy with public tool fields normalized for runtime use."""
    normalized = copy.deepcopy(schema)
    tool_specs = normalized.get("tool_specs")
    if tool_specs is None:
        tool_specs = normalized.get("inferred_tool_schema")
    if tool_specs is None:
        tool_specs = []
    normalized["tool_specs"] = tool_specs
    normalized["inferred_tool_schema"] = tool_specs

    tools = normalized.get("tools")
    if tools is not None:
        if not isinstance(tools, list) or any(not isinstance(t, str) for t in tools):
            raise AgentSpecInvalidError("'tools' must be a list of TOOL_LIBRARY function names.")
        normalized["tool_dispatch"] = _build_dispatch_from_tools(tool_specs, tools)
    elif not normalized.get("tool_dispatch"):
        normalized["tool_dispatch"] = _default_dispatch_for_tool_specs(tool_specs)

    return normalized


def _resolve_fn(entry: dict) -> Callable | None:
    """Return the callable from a dispatch entry.

    Supports two shapes:

    * ``{"python_function": "t_read", …}`` — name looked up in
      :data:`~agentknit.tool_library.TOOL_LIBRARY`.
    * ``{"python_function": <callable>, …}`` — used directly.

    Returns ``None`` when the string name is not found.
    """
    pf = entry.get("python_function")
    if pf is None:
        return None
    if callable(pf):
        return pf
    # string name → look up in TOOL_LIBRARY
    return TOOL_LIBRARY.get(pf)


def dispatch(tool_name: str, args: dict, tool_dispatch: dict) -> tuple[str, dict]:
    """Call the Python function mapped to *tool_name* via *tool_dispatch*.

    tool_dispatch entry shape:
      {
        "python_function": "t_update",        # name string (looked up in TOOL_LIBRARY)
        "python_function": <callable>,         # or a direct callable
        "param_map": {"path": "path", "old_str": "old", "new_str": "new"}
      }

    param_map translates model argument names → Python kwarg names.
    Any model arg not in param_map is passed through unchanged.
    """
    entry = tool_dispatch.get(tool_name)
    if not entry:
        raise FatalToolDispatchError(f"ERROR: no dispatch entry for tool '{tool_name}'")

    fn = _resolve_fn(entry)

    if fn is None:
        pf = entry.get("python_function", "")
        r = f"ERROR: python_function '{pf}' not found in TOOL_LIBRARY"
        return r, {"result": r}

    param_map = entry.get("param_map") or {}
    # Translate model param names → Python kwarg names.
    kwargs = {param_map.get(k, k): v for k, v in args.items()}

    # Derive a human-readable name for error messages
    fn_name = getattr(fn, "__name__", str(fn))

    try:
        result = fn(**kwargs)
    except TypeError as e:
        r = f"ERROR: calling {fn_name}(**{kwargs}): {e}"
        return r, {"result": r}

    # All library functions return (str, dict); handle plain str just in case.
    if isinstance(result, tuple):
        return result
    return str(result), {"result": str(result)}


# ── schema helpers ────────────────────────────────────────────────────────────

def schema_props(tool: dict) -> dict:
    fn = tool.get("function") or tool
    params = fn.get("parameters") or {}
    props = params.get("properties")
    if not isinstance(props, dict):
        props = {k: v for k, v in params.items()
                 if isinstance(v, dict) and "type" in v}
    return props


# ── inline-JSON tool-call extraction (multi-call safe) ───────────────────────

_decoder = json.JSONDecoder()

def extract_inline_calls(text: str) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    pos = 0
    while pos < len(text):
        idx = text.find("{", pos)
        if idx == -1:
            break
        try:
            obj, end = _decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            pos = idx + 1
            continue
        if isinstance(obj, dict):
            name = obj.get("name") or obj.get("function_name")
            args = obj.get("arguments") or obj.get("parameters") or {}
            if name and isinstance(args, dict):
                out.append((name, args))
        pos = end
    return out


# ── prompts & display ─────────────────────────────────────────────────────────

def fmt_call(name: str, args: dict) -> str:
    pretty = ", ".join(f"{k}={v!r}" for k, v in args.items())
    if len(pretty) > 200:
        pretty = pretty[:200] + "…"
    return f"{CYAN}{BOLD}▶ {name}({pretty}){RESET}"

def fmt_usage(usage) -> str:
    """One-line, human-readable token/cache breakdown for a single completion."""
    prompt      = getattr(usage, "prompt_tokens", 0) or 0
    completion  = getattr(usage, "completion_tokens", 0) or 0
    total       = getattr(usage, "total_tokens", 0) or 0
    cached      = getattr(usage, "cached_tokens", 0) or 0
    cache_write = getattr(usage, "cache_creation_tokens", 0) or 0

    prompt_part = f"prompt {prompt:,}"
    if cached:
        pct = (cached / prompt * 100) if prompt else 0
        prompt_part += f" ({cached:,} cached, {pct:.0f}%)"
    parts = [prompt_part]
    if cache_write:
        parts.append(f"cache-write {cache_write:,}")
    parts.append(f"completion {completion:,}")
    parts.append(f"total {total:,}")
    return "  |  ".join(parts)


def _enforce_cache_proof(session: dict, usage) -> None:
    """Fail closed when strict cache mode does not observe a cache hit."""
    if not session.get("strict_cache_proof", True):
        return
    if session.get("llm_call_count", 0) <= 1:
        return
    has_cache_proof = getattr(usage, "has_cache_proof", False)
    cached_tokens = getattr(usage, "cached_tokens", 0) or 0
    if not has_cache_proof:
        raise CacheProofError(
            "Strict cache mode requires explicit cache accounting from the server "
            "after the first LLM call, but this response exposed no cache-proof field."
        )
    if cached_tokens <= 0:
        raise CacheProofError(
            "Strict cache mode requires cached_tokens > 0 after the first LLM call, "
            "but the server reported no cache hit."
        )


def fmt_result(text: str, streamed: bool = False) -> str:
    if streamed:
        # Output was already streamed to console in real-time; just show a
        # short summary instead of repeating the full content.
        return DIM + "  (output streamed above)" + RESET
    lines = text.splitlines()
    head = lines[:20]
    tail = f"\n{DIM}  … ({len(lines)-20} more lines){RESET}" if len(lines) > 20 else ""
    return DIM + "\n".join("  " + line for line in head) + RESET + tail


def fmt_read_result_with_command(command: str, text: str, streamed: bool = False) -> str:
    reminder = f"{YEL}{BOLD}  shell output from:{RESET} {YEL}{command}{RESET}\n"
    return reminder + fmt_result(text, streamed=streamed)


def inline_system_prompt(tools: list[dict]) -> str:
    examples = []
    for tool in tools:
        fn = tool.get("function") or tool
        name = fn.get("name", "?")
        arg_obj = {k: f"<{k}>" for k in schema_props(tool)}
        examples.append(json.dumps({"name": name, "arguments": arg_obj}))
    return (
        "To call a tool respond with ONLY a JSON object (no prose, no markdown fence):\n"
        + "\n".join(examples) + "\n\n"
        "After each call you will receive the result. When the task is done, "
        "respond with a plain-text summary (no JSON).\n"
    )


def read_repl_input(prompt: str) -> str:
    """Read one REPL task, coalescing multiline clipboard paste into one turn."""
    text = input(prompt)
    # If paste arrives line-by-line, keep draining until input has been idle briefly.
    while select.select([sys.stdin], [], [], PASTE_IDLE_TIMEOUT_S)[0]:
        text += "\n" + sys.stdin.readline().rstrip("\n")
    return text.rstrip("\n")


def print_session_history(session: dict) -> None:
    """Replay a resumed session's conversation to the console."""
    structured = session["structured"]
    sep = "─" * 56
    print(f"{DIM}{sep}{RESET}\n")
    for msg in session["messages"]:
        role    = msg.get("role")
        content = msg.get("content") or ""

        if role == "system":
            continue

        if role == "user":
            # Non-structured mode injects tool results as user messages.
            if not structured and content.startswith("Tool results:\n"):
                for block in content[len("Tool results:\n"):].split("\n\n"):
                    block = block.strip()
                    if block:
                        body = block.split("] ", 1)[1] if (block.startswith("[") and "] " in block) else block
                        print(fmt_result(body))
            else:
                print(f"{BOLD}>{RESET} {content}")

        elif role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                for tc in tool_calls:
                    fn   = tc.get("function") or {}
                    name = fn.get("name", "?")
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        args = {}
                    print(fmt_call(name, args))
            elif content:
                if not structured:
                    calls = extract_inline_calls(content)
                    if calls:
                        for name, args in calls:
                            print(fmt_call(name, args))
                        continue
                print(f"\n{GREEN}{BOLD}» {RESET}{content.strip()}\n")

        elif role == "tool":
            print(fmt_result(content))

    print(f"{DIM}{sep}{RESET}\n")


# ── logging ───────────────────────────────────────────────────────────────────

def _open_log(model: str, session_id: str) -> Path:
    now = datetime.datetime.now()
    path = (LOG_BASE / safe_model_name(model)
                     / now.strftime("%Y-%m-%d")
                     / f"{now.strftime('%H%M%S')}_{session_id}.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _log(session: dict, record: dict) -> None:
    record["ts"] = datetime.datetime.now().isoformat(timespec="seconds")
    record["cwd"] = os.getcwd()
    with session["log_path"].open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _snapshot_path(model: str, session_id: str) -> Path:
    return LOG_BASE / safe_model_name(model) / f"{session_id}_messages.json"


def _save_messages_snapshot(session: dict) -> None:
    # Only save if there is at least one non-system message worth resuming.
    if not any(m.get("role") != "system" for m in session["messages"]):
        return
    path = _snapshot_path(session["model"], session["session_id"])
    # Annotate each message with a timestamp (backward-compatible: existing
    # messages that already have a "ts" key are left unchanged).
    annotated = []
    for m in session["messages"]:
        entry = dict(m)
        if "ts" not in entry:
            entry["ts"] = datetime.datetime.now().isoformat(timespec="seconds")
        annotated.append(entry)
    with path.open("w") as f:
        json.dump(annotated, f, ensure_ascii=False, indent=2)


def _load_messages_snapshot(model: str, session_id: str) -> list | None:
    path = _snapshot_path(model, session_id)
    if path.exists():
        with path.open() as f:
            return json.load(f)

    # If session_id looks like a trajectoriz short ID (e.g. "ap-<hex8>"),
    # resolve it by hashing each snapshot's actual session ID.
    if "-" in session_id and len(session_id.split("-")[-1]) == 8:
        import hashlib
        snap_dir = LOG_BASE / safe_model_name(model)
        if snap_dir.is_dir():
            target_hash = session_id.split("-")[-1]
            for snap in snap_dir.glob("*_messages.json"):
                stem = snap.stem
                actual_sid = stem.rsplit("_messages", 1)[0]
                h = hashlib.sha256(actual_sid.encode()).hexdigest()[:8]
                if h == target_hash:
                    with snap.open() as f:
                        return json.load(f)

    return None


def _find_snapshot_in_other_models(model: str, session_id: str) -> tuple[list | None, str | None]:
    """Find a snapshot for session_id in model folders other than `model`."""
    current = safe_model_name(model)
    filename = f"{session_id}_messages.json"
    if not LOG_BASE.exists():
        return None, None

    matches = [
        p for p in LOG_BASE.glob(f"*/{filename}")
        if p.parent.name != current and p.is_file()
    ]
    # Also try trajectoriz short-ID resolution across other model dirs
    if not matches and "-" in session_id and len(session_id.split("-")[-1]) == 8:
        import hashlib
        target_hash = session_id.split("-")[-1]
        for model_dir in sorted(LOG_BASE.iterdir()):
            if model_dir.name == current or not model_dir.is_dir():
                continue
            for snap in model_dir.glob("*_messages.json"):
                stem = snap.stem
                actual_sid = stem.rsplit("_messages", 1)[0]
                h = hashlib.sha256(actual_sid.encode()).hexdigest()[:8]
                if h == target_hash:
                    matches.append(snap)

    if not matches:
        return None, None

    # Prefer the most recently updated trajectory when there are collisions.
    best = max(matches, key=lambda p: p.stat().st_mtime)
    with best.open() as f:
        return json.load(f), best.parent.name


# ── agent loop ────────────────────────────────────────────────────────────────

def _expand_aliases(
    tools: list,
    tool_dispatch: dict,
    aliases: dict,
) -> tuple[list, dict]:
    """Expand alias → canonical mappings into the tool schema and dispatch table.

    For each ``alias_name: canonical_name`` pair:

    * If the alias already has its own ``tool_dispatch`` entry it is left alone.
    * Otherwise the canonical dispatch entry is copied under the alias name.
    * If the canonical tool appears in ``tools`` (structured schema) and the
      alias does not, a deep-copy of the canonical tool spec is appended under
      the alias name so the model is aware of it in structured mode.

    Both ``tools`` and ``tool_dispatch`` are copied; originals are not mutated.
    """
    tools         = list(tools)
    tool_dispatch = dict(tool_dispatch)

    # Fast lookup: tool name → index in tools list
    schema_by_name: dict[str, dict] = {}
    for t in tools:
        fn   = t.get("function") or t
        name = fn.get("name")
        if name:
            schema_by_name[name] = t

    for alias_name, canonical_name in aliases.items():
        # Dispatch: skip if the alias already has an explicit entry
        if alias_name not in tool_dispatch:
            canonical_entry = tool_dispatch.get(canonical_name)
            if canonical_entry:
                tool_dispatch[alias_name] = canonical_entry
            else:
                print(
                    f"{YEL}Warning: alias '{alias_name}' → '{canonical_name}' "
                    f"but '{canonical_name}' has no tool_dispatch entry — skipped{RESET}"
                )
                continue

        # Tool schema (structured mode): clone canonical spec under alias name
        if canonical_name in schema_by_name and alias_name not in schema_by_name:
            alias_tool = copy.deepcopy(schema_by_name[canonical_name])
            fn = alias_tool.get("function") or alias_tool
            fn["name"] = alias_name
            tools.append(alias_tool)
            schema_by_name[alias_name] = alias_tool

    return tools, tool_dispatch


def init_session(schema: dict, non_interactive: bool = False,
                 resumed_from: str | None = None,
                 system_prompt_supplement: str = "",
                 cache_key: str | None = None,
                 max_output_tokens: int | None = None,
                 strict_cache_proof: bool = True,
                 on_event: "EventCallback | None" = None,
                 *,
                 compaction_enabled: bool | None = None,
                 compaction_trigger_tokens: int | None = None,
                 compaction_target_tokens: int | None = None,
                 compaction_keep_last_turns: int | None = None,
                 ) -> dict:
    """Build a stateful session dict.

    The cache_key is sent on every call as both `user` and `prompt_cache_key`
    so OpenRouter / the underlying provider can route this session's growing
    prefix to the same cache shard — much faster and cheaper after turn 1.

    By default the cache_key is the session_id, but a caller can pass a stable
    `cache_key` (e.g. derived from the working directory) to keep reusing a
    provider's prefix cache *without* resuming the prior conversation: that
    requires `resumed_from`, which is the only thing that loads past messages.

    Compaction (keyword-only arguments):

    * ``compaction_enabled`` — whether to enable automatic context compaction
      when the prompt token budget is exceeded.  Defaults to the schema's
      ``compaction_enabled`` key or ``True``.
    * ``compaction_trigger_tokens`` — prompt-token threshold that triggers a
      compaction pass.  Defaults to the schema's ``compaction_trigger_tokens``
      or ``100_000``.
    * ``compaction_target_tokens`` — ``max_tokens`` passed to the compaction
      summary call.  Defaults to the schema's ``compaction_target_tokens`` or
      ``20_000``.
    * ``compaction_keep_last_turns`` — number of recent raw turns to keep
      after compaction.  Defaults to the schema's
      ``compaction_keep_last_turns`` or ``2``.
    """
    schema = _normalize_schema(schema)
    tools         = schema.get("inferred_tool_schema") or []
    behaviour     = schema.get("behaviour") or {}
    tool_dispatch = schema.get("tool_dispatch") or {}
    structured    = behaviour.get("call_delivery_mode", "structured_tool_calls") == "structured_tool_calls"
    model         = schema.get("model", "unknown")

    # Expand aliases before any filtering so aliased tools are treated like
    # first-class tools everywhere (non-interactive filtering, inline prompt, …).
    aliases = schema.get("aliases") or {}
    if aliases:
        tools, tool_dispatch = _expand_aliases(tools, tool_dispatch, aliases)

    if non_interactive:
        # Remove tools whose dispatch entry maps to t_ask_user.
        ask_tool_names = {
            tn for tn, e in tool_dispatch.items()
            if e.get("python_function") in _ASK_USER_FNS
        }
        tools = [t for t in tools
                 if ((t.get("function") or t).get("name")) not in ask_tool_names]

    sys_msg = (
        "You are a helpful coding agent. Use the provided tools to complete the task."
        " When finished, reply in plain text."
        if structured else inline_system_prompt(tools)
    )

    # Append any model-specific system prompt supplement.
    if system_prompt_supplement:
        sys_msg += "\n\n" + system_prompt_supplement

    claude_md = Path.home() / ".claude" / "CLAUDE.md"
    if claude_md.exists():
        sys_msg += "\n\n" + claude_md.read_text()

    agents_md = Path.cwd() / "AGENTS.md"
    if agents_md.exists():
        sys_msg += "\n\n" + agents_md.read_text()
    session_id = resumed_from if resumed_from else uuid.uuid4().hex[:12]
    streaming = bool(
        (schema.get("provider_api_support") or {})
        .get("streaming", {})
        .get("supported", False)
    )
    session_start_ts = datetime.datetime.now().isoformat(timespec="seconds")
    session = {
        "messages":        [{"role": "system", "content": sys_msg,
                             "ts": session_start_ts}],
        "tools":           tools,
        "structured":      structured,
        "tool_dispatch":   tool_dispatch,
        "session_id":      session_id,
        "cache_key":       cache_key or session_id,
        "model":           model,
        "endpoint":        schema.get("endpoint", ""),
        "log_path":        _open_log(model, session_id),
        "non_interactive": non_interactive,
        "usage_totals":    {"prompt": 0, "completion": 0, "total": 0,
                            "cached": 0, "cache_write": 0},
        "provider":        schema.get("provider"),
        "max_output_tokens": max_output_tokens or schema.get("max_output_tokens"),
        "strict_cache_proof": strict_cache_proof,
        "llm_call_count":  0,
        "on_event":        on_event or _default_event_handler,
        "streaming":       streaming,
        "options":         schema.get("options") or [],
        "session_start_ts": session_start_ts,
        "compaction_enabled": (
            compaction_enabled if compaction_enabled is not None
            else schema.get("compaction_enabled", True)
        ),
        "compaction_trigger_tokens": (
            compaction_trigger_tokens if compaction_trigger_tokens is not None
            else schema.get("compaction_trigger_tokens", DEFAULT_COMPACTION_TRIGGER_TOKENS)
        ),
        "compaction_target_tokens": (
            compaction_target_tokens if compaction_target_tokens is not None
            else schema.get("compaction_target_tokens", DEFAULT_COMPACTION_TARGET_TOKENS)
        ),
        "compaction_keep_last_turns": (
            compaction_keep_last_turns if compaction_keep_last_turns is not None
            else schema.get("compaction_keep_last_turns", DEFAULT_COMPACTION_KEEP_LAST_TURNS)
        ),
    }
    _log(session, {"type": "session_start", "model": model,
                   "endpoint": schema.get("endpoint", ""),
                   "session_id": session_id,
                   "mode": behaviour.get("call_delivery_mode"),
                   "non_interactive": non_interactive,
                   "cwd": os.getcwd(),
                   "ts": session_start_ts})
    if resumed_from:
        loaded = _load_messages_snapshot(model, resumed_from)
        if loaded:
            session["messages"] = loaded
            _log(session, {"type": "session_resumed", "resumed_from": resumed_from,
                           "messages_loaded": len(loaded),
                           "ts": datetime.datetime.now().isoformat(timespec="seconds")})
            _emit(session, "session_resumed",
                  session_id=resumed_from, messages_loaded=len(loaded),
                  fmt=f"{DIM}Resumed session {resumed_from} "
                      f"({len(loaded)} messages in context){RESET}")
        else:
            loaded_other, source_model = _find_snapshot_in_other_models(model, resumed_from)
            if loaded_other:
                session["messages"] = loaded_other
                _log(session, {"type": "session_resumed", "resumed_from": resumed_from,
                               "resumed_from_model": source_model,
                               "messages_loaded": len(loaded_other),
                               "ts": datetime.datetime.now().isoformat(timespec="seconds")})
                _emit(session, "session_resumed",
                      session_id=resumed_from, messages_loaded=len(loaded_other),
                      source_model=source_model,
                      fmt=(f"{YEL}No snapshot for {resumed_from!r} under "
                           f"{safe_model_name(model)}; loaded from {source_model}{RESET}\n"
                           f"{DIM}Resumed session {resumed_from} "
                           f"({len(loaded_other)} messages in context){RESET}"))
            else:
                _emit(session, "session_resumed",
                      session_id=resumed_from, messages_loaded=0,
                      fmt=(f"{YEL}Warning: no snapshot found for session {resumed_from!r} "
                           f"in this or other model trajectories — starting fresh{RESET}"))
    return session


def _complete(client: openai.OpenAI | SubprocessOpenAI, session: dict, **kwargs) -> object:
    kwargs["user"] = session["cache_key"]
    if session.get("max_output_tokens"):
        kwargs.setdefault("max_tokens", session["max_output_tokens"])

    extra = kwargs.pop("extra_body", None) or {}

    # Some providers (e.g. NVIDIA NIM) reject unknown extra_body fields.
    # The agent spec can list "exclude-prompt_cache_key" in its "options" array
    # to skip sending prompt_cache_key in extra_body.
    if "exclude-prompt_cache_key" not in session.get("options", []):
        extra.setdefault("prompt_cache_key", session["cache_key"])

    is_openrouter = "openrouter.ai" in (client.base_url.host or "")
    if is_openrouter:
        # OpenRouter only reports cache/cost token details when usage
        # accounting is explicitly requested.
        extra.setdefault("usage", {"include": True})
        # Pin provider routing so every turn hits the same backend — prompt
        # caches are provider-specific and unpinned routing load-balances
        # across providers, which destroys cache continuity.
        if session.get("provider"):
            extra.setdefault("provider", session["provider"])

    kwargs["extra_body"] = extra

    session["_content_was_streamed"] = False
    if session.get("streaming"):
        streamed: list[str] = []
        reasoned: list[str] = []

        def _on_delta(piece: str) -> None:
            first = not streamed
            streamed.append(piece)
            if first:
                prefix = "\n\n" if reasoned else "\n"
                fmt = f"{prefix}{GREEN}{BOLD}» {RESET}{piece}"
            else:
                fmt = piece
            _emit(session, "content_delta", text=piece, first=first,
                  no_newline=True, fmt=fmt)

        def _on_reasoning(piece: str) -> None:
            first = not reasoned
            reasoned.append(piece)
            prefix = f"\n{DIM}[thinking]{RESET} " if first else ""
            fmt = f"{prefix}{DIM}{piece}{RESET}"
            _emit(session, "reasoning_delta", text=piece, first=first,
                  no_newline=True, fmt=fmt)

        resp = client.chat.completions.create(
            **kwargs,
            on_content_delta=_on_delta,
            on_reasoning_delta=_on_reasoning,
        )
        if streamed:
            _emit(session, "content_stream_end", no_newline=True, fmt="\n")
            session["_content_was_streamed"] = True
        elif reasoned:
            _emit(session, "reasoning_stream_end", no_newline=True, fmt="\n")
    else:
        resp = client.chat.completions.create(**kwargs)

    # Sticky provider: lock onto whichever provider served the first call so
    # the rest of the session reuses one provider's prefix cache. An explicit
    # per-spec `provider` (set above) takes precedence and skips this.
    if is_openrouter and not session.get("provider"):
        served = getattr(resp, "provider", None)
        if served:
            session["provider"] = {"order": [served], "allow_fallbacks": False}
            _emit(session, "provider_pinned", provider=served,
                  fmt=f"{DIM}{MAG}[provider] pinned to {served} for this session{RESET}")
            _log(session, {"type": "provider_pinned", "provider": served,
                   "ts": datetime.datetime.now().isoformat(timespec="seconds")})
    return resp


def _compact_session(
    client: openai.OpenAI | SubprocessOpenAI,
    model: str,
    session: dict,
) -> None:
    """Replace the oldest portion of the conversation with a continuation-oriented summary.

    Keeps the system prompt and the most recent *compaction_keep_last_turns*
    turns in raw form.  The middle portion is summarized by the model and
    replaced with a single assistant message tagged with
    ``compacted_summary=true`` metadata.
    """
    messages = session["messages"]
    keep = session.get("compaction_keep_last_turns", DEFAULT_COMPACTION_KEEP_LAST_TURNS)

    # Find the boundary: keep system prompt + last `keep` non-system messages.
    non_system_indices = [i for i, m in enumerate(messages) if m.get("role") != "system"]
    if len(non_system_indices) <= keep:
        return  # nothing to compact

    split_idx = non_system_indices[-keep] if keep > 0 else len(messages)
    prefix = messages[:split_idx]
    suffix = messages[split_idx:]

    # Build a temporary message list for the compaction call.
    compaction_messages = list(prefix)
    compaction_messages.append({"role": "user", "content": _COMPACTION_PROMPT})

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=compaction_messages,
            temperature=0,
            max_tokens=session.get("compaction_target_tokens", DEFAULT_COMPACTION_TARGET_TOKENS),
        )
    except Exception as exc:
        _emit(session, "error", text=f"Compaction failed: {exc}",
              fmt=f"\n{RED}Compaction failed: {exc}{RESET}")
        _log(session, {"type": "compaction_error", "error": str(exc),
                       "ts": datetime.datetime.now().isoformat(timespec="seconds")})
        return

    summary = (resp.choices[0].message.content or "").strip()
    if not summary:
        return

    # Replace the compacted prefix with a single summary message.
    system_msgs = [m for m in prefix if m.get("role") == "system"]
    summary_msg = {
        "role": "assistant",
        "content": summary,
        "compacted_summary": True,
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    session["messages"] = system_msgs + [summary_msg] + suffix

    _emit(session, "compaction",
          summary=summary,
          compacted_turns=len(prefix) - len(system_msgs),
          fmt=f"{DIM}{MAG}[compaction] {len(prefix) - len(system_msgs)} turns summarized "
              f"({len(summary)} chars){RESET}")
    _log(session, {"type": "compaction",
                   "compacted_turns": len(prefix) - len(system_msgs),
                   "summary_length": len(summary),
                   "summary": summary,
                   "ts": datetime.datetime.now().isoformat(timespec="seconds")})
    _save_messages_snapshot(session)


def _maybe_compact(
    client: openai.OpenAI | SubprocessOpenAI,
    model: str,
    session: dict,
    usage,
) -> None:
    """Trigger compaction if the session's prompt tokens exceed the threshold."""
    if not session.get("compaction_enabled", True):
        return
    prompt_tok = getattr(usage, "prompt_tokens", 0) or 0
    trigger = session.get("compaction_trigger_tokens", DEFAULT_COMPACTION_TRIGGER_TOKENS)
    if prompt_tok >= trigger:
        _compact_session(client, model, session)


def _handle_tool_call(
    name: str,
    args: dict,
    session: dict,
) -> str:
    """Dispatch one tool call, log it, print it, return the result string."""
    tool_dispatch   = session["tool_dispatch"]
    non_interactive = session["non_interactive"]

    entry = tool_dispatch.get(name) or {}
    is_ask = entry.get("python_function") in _ASK_USER_FNS

    _log(session, {"type": "tool_call", "name": name,
                   "python_function": entry.get("python_function"), "args": args,
                   "ts": datetime.datetime.now().isoformat(timespec="seconds")})
    _emit(session, "tool_call", name=name, args=args, fmt=fmt_call(name, args))

    _tool_module._tool_context.session_id = session.get("session_id")

    if is_ask and non_interactive:
        result = "ERROR: user interaction is disabled (--non-interactive)"
        log_data: dict = {"result": result}
        streamed = False
    else:
        try:
            result, log_data = dispatch(name, args, tool_dispatch)
            streamed = log_data.pop("streamed", False)
        except FatalToolDispatchError as e:
            result = str(e)
            _emit(session, "tool_result", name=name, result=result, streamed=False,
                  fmt=fmt_result(result))
            _log(session, {"type": "fatal_error", "name": name,
                           "python_function": entry.get("python_function"),
                           "result": result,
                           "ts": datetime.datetime.now().isoformat(timespec="seconds")})
            raise SystemExit(2) from e

    fmt = fmt_result(result, streamed=streamed)
    if name == "read_file":
        path = args.get("path")
        if isinstance(path, str):
            command = _tool_module.get_async_command_for_output_path(path)
            if command:
                fmt = fmt_read_result_with_command(command, result, streamed=streamed)

    _emit(session, "tool_result", name=name, result=result, streamed=streamed,
          fmt=fmt)
    _log(session, {"type": "tool_result", "name": name,
                   "python_function": entry.get("python_function"),
                   "result": result, **log_data,
                   "ts": datetime.datetime.now().isoformat(timespec="seconds")})
    return result


class CancelToken:
    """Cooperative cancellation handle for :func:`run_turn`.

    Call :meth:`cancel` from any thread (e.g. a TUI "stop" button) to request
    that the current turn abort at its next iteration boundary.  The turn raises
    ``KeyboardInterrupt`` when it sees the flag, which the REPL and TUI loops
    both already handle.

    Example::

        token = CancelToken()
        threading.Thread(target=lambda: run_turn(client, model, session,
                                                 task, cancel=token)).start()
        # … later, from the TUI:
        token.cancel()
    """

    def __init__(self) -> None:
        self._cancelled = False

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        self._cancelled = True


class _InputCollector:
    """Collect stdin lines typed while run_turn() is executing.

    Call start() before a turn and stop() after. drain() returns any lines that
    arrived while the agent was busy; they are processed as follow-on turns.
    """

    def __init__(self) -> None:
        self._q: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.5)
            self._thread = None

    def pause(self) -> None:
        """Temporarily stop the reader thread so input() can be called directly."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.5)
            self._thread = None

    def resume(self) -> None:
        """Restart the reader thread after a pause()."""
        self._stop.clear()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def drain(self) -> list[str]:
        items: list[str] = []
        while True:
            try:
                items.append(self._q.get_nowait())
            except queue.Empty:
                break
        return items

    def _reader(self) -> None:
        # Show a dim prompt after 0.5 s of idle stdin (5 × 0.1 s poll ticks),
        # giving the agent's initial output time to appear first.
        # No leading \n: agent output already ends with \n so the cursor is at
        # col 0; adding another \n would produce a blank line.
        _IDLE_TICKS = 5
        _idle = 0
        _shown = False
        while not self._stop.is_set():
            if not select.select([sys.stdin], [], [], 0.1)[0]:
                _idle += 1
                if not _shown and _idle >= _IDLE_TICKS:
                    sys.stdout.write(f"{DIM}> {RESET}")
                    sys.stdout.flush()
                    _shown = True
                continue
            line = sys.stdin.readline()
            if not line:  # EOF
                break
            text = line.rstrip("\n")
            _idle = 0
            _shown = False
            if text.strip():
                self._q.put(text)
                # \r ensures we land at col 0 (cursor may be after the dim ">").
                # No trailing "> " — the idle mechanism will re-show it after 0.5 s.
                sys.stdout.write(f"\r{DIM}[queued — will run after current turn]{RESET}\n")
                sys.stdout.flush()


def run_turn(client: openai.OpenAI | SubprocessOpenAI, model: str, session: dict, task: str,
             *, cancel: CancelToken | None = None) -> SessionResult:
    """Run one agent turn and return a :class:`SessionResult`.

    The result reflects the session state at the end of the turn.
    ``final_reply`` is ``None`` if the turn was interrupted before a final answer.

    Pass a :class:`CancelToken` to allow cooperative cancellation from another
    thread (e.g. a TUI stop button).
    """
    global _in_turn
    _in_turn = True
    try:
        return _run_turn(client, model, session, task, cancel=cancel)
    finally:
        _in_turn = False


def _run_turn(client: openai.OpenAI | SubprocessOpenAI, model: str, session: dict, task: str,
              cancel: CancelToken | None = None) -> SessionResult:
    messages   = session["messages"]
    tools      = session["tools"]
    structured = session["structured"]

    messages.append({"role": "user", "content": task,
                     "ts": datetime.datetime.now().isoformat(timespec="seconds")})
    _log(session, {"type": "user", "content": task,
                   "ts": datetime.datetime.now().isoformat(timespec="seconds")})

    total_tokens = 0
    max_tokens   = DEFAULT_MAX_TOKENS
    try:
        while True:
            if cancel is not None and cancel.cancelled:
                raise KeyboardInterrupt()
            kwargs: dict = dict(model=model, messages=messages, temperature=0)
            if structured:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            try:
                resp  = _complete(client, session, **kwargs)
            except Exception as exc:
                err = f"API error: {exc}"
                _emit(session, "error", text=err,
                      fmt=f"\n{RED}Error: {err}{RESET}")
                _log(session, {"type": "error", "error": err,
                               "ts": datetime.datetime.now().isoformat(timespec="seconds")})
                return _session_result(session)
            msg   = resp.choices[0].message

            # Accumulate token usage from the response and surface it to the user.
            usage = getattr(resp, "usage", None)
            if usage:
                session["llm_call_count"] = session.get("llm_call_count", 0) + 1
                _enforce_cache_proof(session, usage)
                prompt_tok     = getattr(usage, "prompt_tokens", 0) or 0
                completion_tok = getattr(usage, "completion_tokens", 0) or 0
                cached_tok     = getattr(usage, "cached_tokens", 0) or 0
                cache_creat    = getattr(usage, "cache_creation_tokens", 0) or 0
                # Only count effective (non-cached) tokens toward the budget —
                # cached tokens were served from a prefix cache and weren't
                # actually generated/processed, so they shouldn't deplete the
                # budget as aggressively as new tokens.
                effective = max(0, prompt_tok - cached_tok) + completion_tok
                total_tokens += effective
                totals = session["usage_totals"]
                totals["prompt"]      += prompt_tok
                totals["completion"]  += completion_tok
                totals["total"]       += getattr(usage, "total_tokens", 0) or 0
                totals["cached"]      += cached_tok
                totals["cache_write"] += cache_creat
                _emit(session, "usage",
                      prompt=getattr(usage, "prompt_tokens", 0) or 0,
                      completion=getattr(usage, "completion_tokens", 0) or 0,
                      total=getattr(usage, "total_tokens", 0) or 0,
                      cached=getattr(usage, "cached_tokens", 0) or 0,
                      cache_write=getattr(usage, "cache_creation_tokens", 0) or 0,
                      fmt=f"{DIM}{MAG}[tokens] {fmt_usage(usage)}{RESET}")
                _log(session, {"type": "usage",
                               "prompt_tokens":      getattr(usage, "prompt_tokens", 0) or 0,
                               "completion_tokens":  getattr(usage, "completion_tokens", 0) or 0,
                               "total_tokens":       getattr(usage, "total_tokens", 0) or 0,
                               "cached_tokens":      getattr(usage, "cached_tokens", 0) or 0,
                               "cache_creation_tokens": getattr(usage, "cache_creation_tokens", 0) or 0,
                               "ts": datetime.datetime.now().isoformat(timespec="seconds")})
            elif session.get("strict_cache_proof", True) and session.get("llm_call_count", 0) >= 1:
                err = ("Strict cache mode requires usage metadata on every LLM call after the first, "
                       "but the server returned no usage block.")
                _emit(session, "error", text=err, fmt=f"\n{RED}Error: {err}{RESET}")
                _log(session, {"type": "error", "error": err,
                               "ts": datetime.datetime.now().isoformat(timespec="seconds")})
                return _session_result(session)
            _maybe_compact(client, model, session, usage)

            if total_tokens > max_tokens:
                totals = session["usage_totals"]
                raw_total = totals["total"]
                cached_total = totals["cached"]
                raw_info = f" ({raw_total:,} raw API total, {cached_total:,} cached)" if cached_total else ""
                _emit(session, "token_limit", used=total_tokens, limit=max_tokens,
                      raw_total=raw_total, cached_total=cached_total,
                      fmt=f"\n[stopped after exceeding {max_tokens:,} effective tokens "
                          f"(used {total_tokens:,} effective{raw_info})]")
                if not session["non_interactive"]:
                    try:
                        ans = input(f"Double the token budget to {max_tokens * 2:,}? [y/N] ").strip().lower()
                    except EOFError:
                        print()
                        ans = "n"
                    if ans in ("y", "yes"):
                        max_tokens *= 2
                        print(f"{DIM}Token budget doubled to {max_tokens:,}{RESET}", file=sys.stderr)
                        continue
                return

            # ── structured tool_calls ────────────────────────────────────────────
            if structured and msg.tool_calls:
                now_ts = datetime.datetime.now().isoformat(timespec="seconds")
                messages.append({
                    "role": "assistant",
                    "tool_calls": [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.function.name,
                                      "arguments": tc.function.arguments}}
                        for tc in msg.tool_calls
                    ],
                    "ts": now_ts,
                })
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    result = _handle_tool_call(tc.function.name, args, session)
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result,
                                     "ts": datetime.datetime.now().isoformat(timespec="seconds")})
                continue

            text = msg.content or ""

            # ── inline JSON tool calls ───────────────────────────────────────────
            if not structured:
                now_ts = datetime.datetime.now().isoformat(timespec="seconds")
                messages.append({"role": "assistant", "content": text, "ts": now_ts})
                calls = extract_inline_calls(text)
                if calls:
                    results = []
                    for name, args in calls:
                        result = _handle_tool_call(name, args, session)
                        results.append(f"[{name}] {result}")
                    messages.append({"role": "user",
                                     "content": "Tool results:\n" + "\n\n".join(results),
                                     "ts": datetime.datetime.now().isoformat(timespec="seconds")})
                    continue

            # ── final answer ─────────────────────────────────────────────────────
            # In structured mode the assistant message wasn't appended above.
            if structured:
                messages.append({"role": "assistant", "content": text,
                                 "ts": datetime.datetime.now().isoformat(timespec="seconds")})
            _log(session, {"type": "assistant", "content": text,
                       "ts": datetime.datetime.now().isoformat(timespec="seconds")})
            already_streamed = session.get("_content_was_streamed", False)
            _emit(session, "final_answer", text=text.strip(),
                  fmt="" if already_streamed else f"\n{GREEN}{BOLD}» {RESET}{text.strip()}\n")
            t = session["usage_totals"]
            cached_part = f", {t['cached']:,} cached" if t["cached"] else ""
            eff = max(0, t['prompt'] - t['cached']) + t['completion']
            eff_part = f"  |  effective {eff:,}" if t['cached'] else ""
            _emit(session, "session_usage", **t,
                  fmt=(f"{DIM}{MAG}[session tokens] prompt {t['prompt']:,}{cached_part}  |  "
                       f"completion {t['completion']:,}  |  total {t['total']:,}{eff_part}{RESET}\n"))
            return _session_result(session)
    except CacheProofError as exc:
        err = str(exc)
        _emit(session, "error", text=err, fmt=f"\n{RED}Error: {err}{RESET}")
        _log(session, {"type": "error", "error": err,
                       "ts": datetime.datetime.now().isoformat(timespec="seconds")})
        return _session_result(session)


# ── pricing check ─────────────────────────────────────────────────────────────

def _fetch_openrouter_price(model: str) -> tuple[float | None, float | None]:
    """Return (input_per_million, output_per_million) from OpenRouter API, or (None, None)."""
    try:
        url = "https://openrouter.ai/api/v1/models"
        req = urllib.request.Request(url, headers={"User-Agent": "agent-probe/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        for m in data.get("data", []):
            if m["id"] == model:
                p = m.get("pricing", {})
                inp = float(p.get("prompt", 0)) * 1_000_000
                out = float(p.get("completion", 0)) * 1_000_000
                return inp, out
    except Exception:
        pass
    return None, None


def _azure_price_cache_path() -> Path:
    """Return the filesystem path for the Azure pricing cache file."""
    cache_dir = LOG_BASE / "azure_pricing_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "prices.json"


def _load_azure_price_cache() -> dict | None:
    """Load cached Azure pricing data if it is less than 7 days old."""
    path = _azure_price_cache_path()
    if not path.exists():
        return None
    try:
        with path.open() as f:
            cache = json.load(f)
        cached_at = cache.get("cached_at")
        if cached_at is None:
            return None
        age = datetime.datetime.now() - datetime.datetime.fromisoformat(cached_at)
        if age.days >= 7:
            return None
        return cache
    except Exception:
        return None


def _save_azure_price_cache(items: list[dict]) -> None:
    """Save Azure pricing data to the cache file with a timestamp."""
    cache = {
        "cached_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "items": items,
    }
    path = _azure_price_cache_path()
    with path.open("w") as f:
        json.dump(cache, f, indent=2)


def _fetch_azure_price(model: str) -> tuple[float | None, float | None]:
    """Return (input_per_million, output_per_million) from Azure Retail Prices API.

    Pricing data is cached on disk for up to 7 days to avoid repeated
    slow API calls on every startup.
    """
    try:
        import urllib.parse

        # Try loading from cache first
        cached = _load_azure_price_cache()
        if cached is not None:
            items = cached["items"]
        else:
            url = (
                "https://prices.azure.com/api/retail/prices"
                "?api-version=2023-01-01-preview"
                "&" + urllib.parse.urlencode({"$filter": "serviceName eq 'Foundry Models' and priceType eq 'Consumption'"})
            )
            items = []
            while url:
                req = urllib.request.Request(url, headers={"User-Agent": "agent-probe/1.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode())
                items.extend(data.get("Items", []))
                url = data.get("NextPageLink")
            _save_azure_price_cache(items)

        norm_model = re.sub(r"[^a-z0-9]", "", model.lower())

        # Collect all matching entries, tagged by zone type
        global_entries: list[tuple[bool, float]] = []  # (is_input, price_per_mtok)
        dz_entries: list[tuple[bool, float]] = []

        for item in items:
            meter_name = item.get("meterName", "")
            meter_lower = meter_name.lower()

            m = re.match(
                r"^(.+?)[\s-]+"
                r"(inp|outp|input|output|inpt|outpt|in|out)\b"
                r"(?:[\s-]+.*)?\s+tokens$",
                meter_lower,
            )
            if not m:
                continue

            model_key_raw = m.group(1).strip()
            norm_key = re.sub(r"[^a-z0-9]", "", model_key_raw)

            if norm_model not in norm_key and norm_key not in norm_model:
                continue

            is_global = "glbl" in meter_lower or "global" in meter_lower
            is_dz = "dz" in meter_lower
            is_cache = "cache" in meter_lower

            price = float(item["retailPrice"])
            unit = item.get("unitOfMeasure", "1M")
            price_per_mtok = price * 1000 if "1K" in unit else price

            direction = m.group(2).lower()
            is_input = direction.startswith(("inp", "input", "in"))

            entry = (is_input, price_per_mtok, is_cache)
            if is_global:
                global_entries.append(entry)
            elif is_dz:
                dz_entries.append(entry)

        # Prefer global prices; fall back to DZ if no globals exist
        entries = global_entries if global_entries else dz_entries

        # Prefer non-cache prices; only use cache prices as fallback
        regular_inp = [p for inp, p, cache in entries if inp and not cache]
        cache_inp = [p for inp, p, cache in entries if inp and cache]
        inp_price = min(regular_inp, default=min(cache_inp, default=None))

        regular_out = [p for inp, p, cache in entries if not inp and not cache]
        cache_out = [p for inp, p, cache in entries if not inp and cache]
        out_price = min(regular_out, default=min(cache_out, default=None))

        return inp_price, out_price
    except Exception:
        pass
    return None, None


def check_and_display_pricing(schema: dict) -> None:
    """Fetch current pricing, display it, and exit if it exceeds spec limits."""
    model    = schema.get("model", "")
    endpoint = schema.get("endpoint", "")
    max_inp  = schema.get("max_input_token_price_per_million")
    max_out  = schema.get("max_output_token_price_per_million")

    is_openrouter = "openrouter" in endpoint
    is_azure = "azure.com" in endpoint or "services.ai.azure.com" in endpoint
    is_local = _parse_run_uri(endpoint) or _parse_run_uri(model)

    if is_local:
        label = "local/subprocess"
        print(f"{MAG}{BOLD}[pricing]{RESET}{MAG}  {model}  |  endpoint: {label}  |  no price check{RESET}")
        return

    if is_azure:
        cur_inp, cur_out = _fetch_azure_price(model)
        source = "Azure"
    elif is_openrouter:
        cur_inp, cur_out = _fetch_openrouter_price(model)
        source = "OpenRouter"
    else:
        print(f"{MAG}{BOLD}[pricing]{RESET}{MAG}  {model}  |  endpoint: {endpoint}  |  no price check{RESET}")
        return

    if cur_inp is None and cur_out is None:
        print(f"{YEL}{BOLD}[pricing]{RESET}{YEL}  {model}  |  could not fetch price from {source}{RESET}")
        return

    inp_ok = max_inp is None or (cur_inp is not None and cur_inp <= max_inp)
    out_ok = max_out is None or (cur_out is not None and cur_out <= max_out)

    color = GREEN if (inp_ok and out_ok) else RED

    inp_str = f"${cur_inp:.4f}/M" if cur_inp is not None else "N/A"
    out_str = f"${cur_out:.4f}/M" if cur_out is not None else "N/A"
    print(f"{color}{BOLD}[pricing]{RESET}{color}  {model}  |  input: {inp_str}  |  output: {out_str}{RESET}")

    if not inp_ok:
        raise PricingLimitExceededError(
            f"Input price {inp_str} exceeds limit ${max_inp}/M for {model}",
            model=model, direction="input",
            current_price=cur_inp or 0.0, limit=max_inp or 0.0,
        )
    if not out_ok:
        raise PricingLimitExceededError(
            f"Output price {out_str} exceeds limit ${max_out}/M for {model}",
            model=model, direction="output",
            current_price=cur_out or 0.0, limit=max_out or 0.0,
        )


# ── public library API ────────────────────────────────────────────────────────

def validate_schema(schema: dict) -> None:
    """Raise a typed exception if *schema* cannot be used to run an agent.

    Raises:
        AgentSpecDisabledError: if ``schema["disabled"]`` is true.
        AgentSpecInvalidError:  if ``schema["inferred_tool_schema"]`` is absent.
    """
    schema = _normalize_schema(schema)
    if schema.get("disabled"):
        comment = schema.get("comment", "This agent spec is disabled.")
        raise AgentSpecDisabledError(comment, comment=comment)
    if not schema.get("inferred_tool_schema"):
        model = schema.get("model", "unknown")
        raise AgentSpecInvalidError(
            f"No tool schema for {model!r} — probe likely failed.",
            model=model,
        )


@dataclasses.dataclass
class SessionResult:
    """Structured result returned by :func:`run_turn` and :func:`run_task`."""
    session_id:  str
    final_reply: str | None
    usage:       dict
    messages:    list[dict]


def _session_result(session: dict) -> SessionResult:
    """Build a SessionResult snapshot from the current session state."""
    final_reply: str | None = None
    for msg in reversed(session["messages"]):
        if msg.get("role") == "assistant" and msg.get("content"):
            final_reply = msg["content"]
            break
    return SessionResult(
        session_id  = session["session_id"],
        final_reply = final_reply,
        usage       = dict(session["usage_totals"]),
        messages    = session["messages"],
    )


def _get_key_for_schema(schema: dict) -> str:
    """Return the API key appropriate for this schema.

    Priority:
      1. keyring_service + keyring_username in the spec → keyring lookup
      2. key_env in the spec → read that env variable
      3. Default OPENROUTER_API_KEY via ensure_api_key()
    """
    ks = schema.get("keyring_service")
    ku = schema.get("keyring_username")
    if ks and ku:
        try:
            import keyring as _kr
            val = _kr.get_password(ks, ku)
            if val:
                return val
        except Exception:
            pass
        # Fall back to env var named by keyring_username uppercased
        env_name = ku.upper().replace("-", "_")
        val = os.environ.get(env_name)
        if val:
            return val

    key_env = schema.get("key_env")
    if key_env:
        val = os.environ.get(key_env)
        if val:
            return val

    return get_api_key()


def create_client(schema: dict) -> "openai.OpenAI | SubprocessOpenAI":
    """Create an API client from a loaded agent spec schema.

    Handles subprocess (run://), OpenCode GitHub-Copilot, and standard
    OpenAI-compatible endpoints.  Call :func:`load_or_probe` first to obtain
    a schema.

    If the schema contains a ``max_rpm`` key, it is passed to the OpenAI
    client constructor to enforce a client-side rate limit (e.g. 40 RPM
    for NVIDIA NIM free-tier endpoints).
    """
    schema = _normalize_schema(schema)
    endpoint    = schema.get("endpoint") or DEFAULT_ENDPOINT
    binary_path = _parse_run_uri(endpoint) or _parse_run_uri(schema.get("model", ""))
    auth        = schema.get("auth")
    max_rpm     = schema.get("max_rpm")
    kwargs: dict = {}
    if max_rpm is not None:
        kwargs["max_rpm"] = max_rpm
    if binary_path is not None:
        return SubprocessOpenAI(binary_path)
    if auth == "opencode-github-copilot":
        return openai.OpenAI(api_key=_get_opencode_token(), base_url=endpoint,
                             auth_header="X-API-Key", **kwargs)
    return openai.OpenAI(api_key=_get_key_for_schema(schema), base_url=endpoint, **kwargs)


def run_task(
    schema: dict,
    task: str,
    *,
    non_interactive: bool = False,
    session_id: str | None = None,
    cache_key: str | None = None,
    system_prompt_supplement: str = "",
    max_output_tokens: int | None = None,
    strict_cache_proof: bool = True,
    on_event: "EventCallback | None" = None,
    compaction_enabled: bool | None = None,
    compaction_trigger_tokens: int | None = None,
    compaction_target_tokens: int | None = None,
    compaction_keep_last_turns: int | None = None,
) -> SessionResult:
    """Run a single task against the agent and return a :class:`SessionResult`.

    This is the primary SDK entry point for one-shot programmatic use — no
    argparse, no stdin reading, no REPL loop.

    Example::

        schema = load_or_probe("qwen/qwen3-8b", "https://openrouter.ai/api/v1", False)
        result = run_task(schema, "List the files in /tmp")
        print(result.final_reply)
        print(result.usage)
    """
    validate_schema(schema)
    client  = create_client(schema)
    session = init_session(
        schema,
        non_interactive=non_interactive,
        resumed_from=session_id,
        system_prompt_supplement=system_prompt_supplement,
        cache_key=cache_key,
        max_output_tokens=max_output_tokens,
        strict_cache_proof=strict_cache_proof,
        on_event=on_event,
        compaction_enabled=compaction_enabled,
        compaction_trigger_tokens=compaction_trigger_tokens,
        compaction_target_tokens=compaction_target_tokens,
        compaction_keep_last_turns=compaction_keep_last_turns,
    )
    try:
        return run_turn(client, schema["model"], session, task)
    finally:
        _save_messages_snapshot(session)
        _log(session, {"type": "session_end", "session_id": session["session_id"],
                       "reason": "run_task_complete"})


def run(
    schema: dict | None = None,
    task: str | None = None,
    *,
    model: str | None = None,
    endpoint: str | None = None,
    reprobe: bool = False,
    non_interactive: bool = False,
    session_id: str | None = None,
    cache_key: str | None = None,
    system_prompt_supplement: str = "",
    max_output_tokens: int | None = None,
    strict_cache_proof: bool = True,
    on_event: "EventCallback | None" = None,
    compaction_enabled: bool | None = None,
    compaction_trigger_tokens: int | None = None,
    compaction_target_tokens: int | None = None,
    compaction_keep_last_turns: int | None = None,
) -> SessionResult:
    """Backward-compatible helper for :func:`run_task`.

    Accepts either a loaded schema dict or a ``model``/``endpoint`` pair for
    wrapper scripts that want to skip ``agentknit.main()``.
    """
    if schema is None:
        if model is None or endpoint is None:
            raise TypeError("run() requires either `schema` or both `model` and `endpoint`.")
        schema = load_or_probe(model, endpoint, reprobe)
    elif model is not None or endpoint is not None:
        raise TypeError("run() accepts either `schema` or `model`/`endpoint`, not both.")
    if task is None:
        raise TypeError("run() missing required argument: 'task'")
    return run_task(
        schema,
        task,
        non_interactive=non_interactive,
        session_id=session_id,
        cache_key=cache_key,
        system_prompt_supplement=system_prompt_supplement,
        max_output_tokens=max_output_tokens,
        strict_cache_proof=strict_cache_proof,
        on_event=on_event,
        compaction_enabled=compaction_enabled,
        compaction_trigger_tokens=compaction_trigger_tokens,
        compaction_target_tokens=compaction_target_tokens,
        compaction_keep_last_turns=compaction_keep_last_turns,
    )


def _build_resume_cmd(model: str, session_id: str, default_program: str | None = None) -> str:
    program = os.environ.get("AGENTKNIT_RESUME_COMMAND")
    if program:
        return f"{program} --session {session_id}"
    if default_program is None:
        default_program = sys.argv[0]
    return f"{default_program} {model} --session {session_id}"


def _repl_setup(
    schema: dict,
    *,
    non_interactive: bool = False,
    session_id: str | None = None,
    cache_key: str | None = None,
    system_prompt_supplement: str = "",
    max_output_tokens: int | None = None,
    strict_cache_proof: bool = True,
    on_event: "EventCallback | None" = None,
    compaction_enabled: bool | None = None,
    compaction_trigger_tokens: int | None = None,
    compaction_target_tokens: int | None = None,
    compaction_keep_last_turns: int | None = None,
) -> tuple:
    """Common REPL setup: validate, create client, init session, return (client, session, model, hist_file)."""
    validate_schema(schema)
    client = create_client(schema)
    session = init_session(
        schema,
        non_interactive=non_interactive,
        resumed_from=session_id,
        system_prompt_supplement=system_prompt_supplement,
        cache_key=cache_key,
        max_output_tokens=max_output_tokens,
        strict_cache_proof=strict_cache_proof,
        on_event=on_event,
        compaction_enabled=compaction_enabled,
        compaction_trigger_tokens=compaction_trigger_tokens,
        compaction_target_tokens=compaction_target_tokens,
        compaction_keep_last_turns=compaction_keep_last_turns,
    )
    model = schema["model"]

    if session_id:
        print_session_history(session)

    import hashlib as _hashlib
    _hist_dir = Path.home() / ".local" / "share" / "agent_probe" / "repl_history"
    _hist_dir.mkdir(parents=True, exist_ok=True)
    _cwd_tag = _hashlib.md5(os.getcwd().encode()).hexdigest()[:12]
    _hist_file = _hist_dir / f"{_cwd_tag}.hist"
    try:
        readline.read_history_file(_hist_file)
    except FileNotFoundError:
        pass
    readline.set_history_length(500)

    return client, session, model, _hist_file


def _repl_teardown(session: dict, hist_file: Path, resume_cmd: str) -> None:
    """Common REPL teardown: save history, snapshot, log."""
    try:
        readline.write_history_file(hist_file)
    except Exception:
        pass
    _save_messages_snapshot(session)
    _log(session, {"type": "session_end", "session_id": session["session_id"],
                   "reason": "repl_exit"})
    print(f"\n{DIM}Resume: {resume_cmd}{RESET}")


def _repl_loop_body(
    t: str,
    client: openai.OpenAI | SubprocessOpenAI,
    session: dict,
    model: str,
    *,
    use_async_input: bool = False,
) -> None:
    """Run one REPL turn, optionally with async input queue.

    When *use_async_input* is True a background ``_InputCollector`` thread
    queues keystrokes typed while the agent is thinking; they are drained
    and run as follow-up turns.  When False (the default) the turn runs
    synchronously with no background reader — simpler and immune to stdin
    races with tools that call ``input()``.
    """
    current_model = session.get("model", model)
    if _slash_registry.dispatch(t, session, client, current_model):
        _save_messages_snapshot(session)
        return

    if use_async_input:
        _async_repl_turn(t, client, session, current_model)
    else:
        _sync_repl_turn(t, client, session, current_model)


def _sync_repl_turn(
    t: str,
    client: openai.OpenAI | SubprocessOpenAI,
    session: dict,
    model: str,
) -> None:
    """Run a single turn synchronously — no background reader thread."""
    try:
        run_turn(client, model, session, t)
    except KeyboardInterrupt:
        print(f"\n{DIM}[interrupted]{RESET}")
    except Exception as exc:
        _emit(session, "error", text=str(exc),
              fmt=f"\n{RED}Error: {exc}{RESET}")
        _log(session, {"type": "error", "error": str(exc),
               "ts": datetime.datetime.now().isoformat(timespec="seconds")})
    _save_messages_snapshot(session)


def _async_repl_turn(
    t: str,
    client: openai.OpenAI | SubprocessOpenAI,
    session: dict,
    model: str,
) -> None:
    """Run a turn with a background ``_InputCollector`` queuing keystrokes."""
    _collector = _InputCollector()
    _tool_module._input_collector = _collector
    _pending = [t]
    try:
        while _pending:
            _task = _pending.pop(0)
            _collector.start()
            _interrupted = False
            try:
                run_turn(client, model, session, _task)
            except KeyboardInterrupt:
                print(f"\n{DIM}[interrupted]{RESET}")
                _interrupted = True
            except Exception as exc:
                _emit(session, "error", text=str(exc),
                      fmt=f"\n{RED}Error: {exc}{RESET}")
                _log(session, {"type": "error", "error": str(exc),
                       "ts": datetime.datetime.now().isoformat(timespec="seconds")})
            finally:
                _collector.stop()
            _save_messages_snapshot(session)
            if _interrupted:
                break
            for _qi in _collector.drain():
                _qs = _qi.strip()
                if not _qs or _qs.lower() in ("exit", "quit", "q"):
                    continue
                if not _slash_registry.dispatch(_qs, session, client, model):
                    _pending.append(_qi)
                else:
                    _save_messages_snapshot(session)
    finally:
        _tool_module._input_collector = None


def run_repl(
    schema: dict,
    *,
    non_interactive: bool = False,
    session_id: str | None = None,
    cache_key: str | None = None,
    system_prompt_supplement: str = "",
    max_output_tokens: int | None = None,
    strict_cache_proof: bool = True,
    on_event: "EventCallback | None" = None,
    compaction_enabled: bool | None = None,
    compaction_trigger_tokens: int | None = None,
    compaction_target_tokens: int | None = None,
    compaction_keep_last_turns: int | None = None,
) -> None:
    """Start an interactive REPL session against the agent (sync, no background thread).

    Reads tasks line-by-line from stdin and runs :func:`run_turn` for each.
    The session snapshot is saved after every turn so it can be resumed with
    ``--session <session_id>``.

    Slash commands (``/clear``, ``/model``, ``/usage``, ``/help``) are
    intercepted before sending input to the model.

    This is the *sync* variant — no background reader thread, so tools that
    call ``input()`` (e.g. ``ask_user_question``) work without stdin races.
    """
    client, session, model, hist_file = _repl_setup(
        schema,
        non_interactive=non_interactive,
        session_id=session_id,
        cache_key=cache_key,
        system_prompt_supplement=system_prompt_supplement,
        max_output_tokens=max_output_tokens,
        strict_cache_proof=strict_cache_proof,
        on_event=on_event,
        compaction_enabled=compaction_enabled,
        compaction_trigger_tokens=compaction_trigger_tokens,
        compaction_target_tokens=compaction_target_tokens,
        compaction_keep_last_turns=compaction_keep_last_turns,
    )
    resume_cmd = _build_resume_cmd(model, session["session_id"], "agent-probe")

    display_name = schema.get("display_name", f"agentknit {model}")
    print(f"{BOLD}{display_name}{RESET}  (type 'exit' to quit)\n")
    try:
        while True:
            try:
                t = read_repl_input(f"{RL_BOLD}>{RL_RESET} ")
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print()
                continue
            cmd = t.strip()
            if cmd.lower() in ("exit", "quit", "q"):
                break
            if cmd:
                _repl_loop_body(cmd, client, session, model, use_async_input=False)
    finally:
        _repl_teardown(session, hist_file, resume_cmd)


def run_async_repl(
    schema: dict,
    *,
    non_interactive: bool = False,
    session_id: str | None = None,
    cache_key: str | None = None,
    system_prompt_supplement: str = "",
    max_output_tokens: int | None = None,
    strict_cache_proof: bool = True,
    on_event: "EventCallback | None" = None,
    compaction_enabled: bool | None = None,
    compaction_trigger_tokens: int | None = None,
    compaction_target_tokens: int | None = None,
    compaction_keep_last_turns: int | None = None,
) -> None:
    """Start an interactive REPL session with a background input queue.

    Same as :func:`run_repl` but spawns a background ``_InputCollector``
    thread that queues keystrokes typed while the agent is thinking.  Those
    queued inputs are drained and run as follow-up turns after the current
    turn completes.

    **Caveat**: tools that call ``input()`` (e.g. ``ask_user_question``) may
    race with the background reader thread for stdin.  Use ``run_repl``
    (sync) if you need those tools.
    """
    client, session, model, hist_file = _repl_setup(
        schema,
        non_interactive=non_interactive,
        session_id=session_id,
        cache_key=cache_key,
        system_prompt_supplement=system_prompt_supplement,
        max_output_tokens=max_output_tokens,
        strict_cache_proof=strict_cache_proof,
        on_event=on_event,
        compaction_enabled=compaction_enabled,
        compaction_trigger_tokens=compaction_trigger_tokens,
        compaction_target_tokens=compaction_target_tokens,
        compaction_keep_last_turns=compaction_keep_last_turns,
    )
    resume_cmd = _build_resume_cmd(model, session["session_id"], "agent-probe")

    display_name = schema.get("display_name", f"agentknit {model}")
    print(f"{BOLD}{display_name}{RESET}  (type 'exit' to quit)\n")
    try:
        while True:
            try:
                t = read_repl_input(f"{RL_BOLD}>{RL_RESET} ")
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print()
                continue
            cmd = t.strip()
            if cmd.lower() in ("exit", "quit", "q"):
                break
            if cmd:
                _repl_loop_body(cmd, client, session, model, use_async_input=True)
    finally:
        _repl_teardown(session, hist_file, resume_cmd)


# ── entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe + agent loop for any OpenRouter model.")
    p.add_argument("model", help="Model ID, e.g. qwen/qwen3-vl-32b-instruct")
    p.add_argument("task", nargs="*", help="Task to run (omit for REPL or stdin)")
    p.add_argument("--reprobe", action="store_true", help="Force re-probing")
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Endpoint base URL")
    p.add_argument("--non-interactive", action="store_true", dest="non_interactive",
                   help="Remove ask_user_question from the tool schema; "
                        "return an error if called anyway")
    p.add_argument("--session", metavar="SESSION_ID",
                   help="Resume a previous session by its ID (loads its message history)")
    p.add_argument("--cache-key", metavar="KEY", dest="cache_key",
                   help="Stable prompt-cache key (user/prompt_cache_key) for prefix-cache "
                        "reuse, WITHOUT resuming any prior conversation. Defaults to the "
                        "session ID. Use --session to actually resume history.")
    p.add_argument("--system-prompt-supplement", default="",
                   help="Extra text appended to the system prompt for this model")
    p.add_argument("--max-tokens", type=int, dest="max_tokens", default=None,
                   help="Cap max output tokens per request. Overrides the spec's "
                        "max_output_tokens. Useful for models with a huge default "
                        "output that would otherwise reserve large credit holds.")
    p.add_argument("--no-strict-cache-proof", action="store_true",
                   help="Disable the default fail-closed cache-proof check that "
                        "requires a nonzero cache hit on every LLM call after the first.")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    try:
        schema = load_or_probe(args.model, args.endpoint, args.reprobe)
        validate_schema(schema)
        check_and_display_pricing(schema)
    except AgentSpecDisabledError as e:
        sys.exit(f"{RED}Agent disabled: {e.comment or e}{RESET}")
    except AgentSpecInvalidError as e:
        sys.exit(f"{RED}{e}{RESET}")
    except PricingLimitExceededError as e:
        sys.exit(f"{RED}ABORT: {e}{RESET}")
    except AuthenticationError as e:
        sys.exit(f"{RED}Authentication error: {e}{RESET}")

    model    = schema["model"]
    behaviour = schema.get("behaviour") or {}
    mode     = behaviour.get("call_delivery_mode", "structured_tool_calls")
    mode_str = f"  |  mode: {mode}" if mode != "structured_tool_calls" else ""

    opts: dict = dict(
        non_interactive          = args.non_interactive,
        resumed_from             = args.session,
        system_prompt_supplement = args.system_prompt_supplement,
        cache_key                = args.cache_key,
        max_output_tokens        = args.max_tokens,
        strict_cache_proof       = not args.no_strict_cache_proof,
    )

    # Print the session header once, before any task runs.
    client  = create_client(schema)
    session = init_session(schema, **opts)
    tool_names = [((t.get("function") or t).get("name", "?")) for t in session["tools"]]
    print(f"{DIM}Model: {model}{mode_str}  |  "
          f"{len(tool_names)} tools: {', '.join(tool_names)}{RESET}\n")
    print(f"{DIM}Session: {session['session_id']}  |  log: {session['log_path']}{RESET}\n")

    if args.session:
        print_session_history(session)

    resume_cmd = _build_resume_cmd(model, session["session_id"])

    if args.task:
        try:
            run_turn(client, model, session, " ".join(args.task))
        finally:
            _save_messages_snapshot(session)
            _log(session, {"type": "session_end", "session_id": session["session_id"],
                           "reason": "one_shot_task"})
            print(f"\n{DIM}Resume: {resume_cmd}{RESET}")
        return

    if not sys.stdin.isatty():
        task = sys.stdin.read().strip()
        if task:
            try:
                run_turn(client, model, session, task)
            finally:
                _save_messages_snapshot(session)
                _log(session, {"type": "session_end", "session_id": session["session_id"],
                               "reason": "stdin_task"})
                print(f"\n{DIM}Resume: {resume_cmd}{RESET}")
        return

    # Interactive REPL — reuse the already-created client + session.
    repl_opts = {k: v for k, v in opts.items() if k != "resumed_from"}
    repl_opts["session_id"] = opts.get("resumed_from")
    run_repl(schema, **repl_opts)


if __name__ == "__main__":
    main()
