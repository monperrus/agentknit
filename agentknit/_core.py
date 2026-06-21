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
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import datetime
import json
import os
import re
import readline  # noqa: F401 — enables arrow keys / history in input()
import select
import signal
import sys
import urllib.request
import uuid
from pathlib import Path
from typing import Callable

from . import openai_compat as openai
from .openai_compat import SubprocessOpenAI

from llmprobe import probe as probe
from . import tool_library as _tool_module
from .tool_library import TOOL_LIBRARY, _ASK_USER_FNS
from .exceptions import (
    AgentSpecDisabledError, AgentSpecInvalidError,
    PricingLimitExceededError, AuthenticationError,
)


DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1"
DEFAULT_MAX_TOKENS = 3_000_000
LOG_BASE = Path.home() / ".local" / "share" / "agent_probe"

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
    """Fire *event_type* through the session's registered event handler."""
    handler: EventCallback = session.get("on_event") or _default_event_handler
    handler(event_type, data)


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
            "model":                model,
            "endpoint":             endpoint,
            "status":               "default",
            "inferred_tool_schema": _DEFAULT_TOOL_SCHEMA,
            "behaviour":            {"call_delivery_mode": "structured_tool_calls"},
            "tool_dispatch":        _DEFAULT_TOOL_DISPATCH,
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

    print(f"{YEL}Probing {model} … this may take ~30 s{RESET}")
    probe.ENDPOINT = endpoint
    probe.MODEL    = model
    client = probe.make_client(get_api_key())

    elicited      = probe.elicit_round(client)
    initial_tools = probe.build_tool_schema(elicited)
    probe.probe_round(client, initial_tools)
    final_probes  = probe.probe_round(client, initial_tools, label="Final probe")
    behaviour     = probe.behavioural_summary(final_probes)
    _tool_lib_path = Path(__file__).resolve().parent / "tool_library.py"
    tool_dispatch = probe.build_tool_dispatch(elicited, final_probes, client,
                                              tool_library_path=_tool_lib_path)

    data = {
        "model":                model,
        "endpoint":             endpoint,
        "status":               "ok",
        "elicited_names":       {op: v["function_name"] for op, v in elicited.items()},
        "inferred_tool_schema": initial_tools,
        "behaviour":            behaviour,
        "tool_dispatch":        tool_dispatch,
    }
    with path.open("w") as f:
        json.dump(data, f, indent=2)
    return data


# ── dispatch ──────────────────────────────────────────────────────────────────

class FatalToolDispatchError(RuntimeError):
    """Raised when the agent requests a tool that cannot be dispatched."""


def dispatch(tool_name: str, args: dict, tool_dispatch: dict) -> tuple[str, dict]:
    """Call the Python function mapped to *tool_name* via *tool_dispatch*.

    tool_dispatch entry shape:
      {
        "python_function": "t_update",
        "param_map": {"path": "path", "old_str": "old", "new_str": "new"}
      }

    param_map translates model argument names → Python kwarg names.
    Any model arg not in param_map is passed through unchanged.
    """
    entry = tool_dispatch.get(tool_name)
    if not entry:
        raise FatalToolDispatchError(f"ERROR: no dispatch entry for tool '{tool_name}'")

    fn_name = entry.get("python_function", "")
    fn = TOOL_LIBRARY.get(fn_name)

    if fn is None:
        r = f"ERROR: python_function '{fn_name}' not found in TOOL_LIBRARY"
        return r, {"result": r}

    param_map = entry.get("param_map") or {}
    # Translate model param names → Python kwarg names.
    kwargs = {param_map.get(k, k): v for k, v in args.items()}

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


def fmt_result(text: str, streamed: bool = False) -> str:
    if streamed:
        # Output was already streamed to console in real-time; just show a
        # short summary instead of repeating the full content.
        return DIM + "  (output streamed above)" + RESET
    lines = text.splitlines()
    head = lines[:20]
    tail = f"\n{DIM}  … ({len(lines)-20} more lines){RESET}" if len(lines) > 20 else ""
    return DIM + "\n".join("  " + line for line in head) + RESET + tail


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
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


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
                 on_event: "EventCallback | None" = None) -> dict:
    """Build a stateful session dict.

    The cache_key is sent on every call as both `user` and `prompt_cache_key`
    so OpenRouter / the underlying provider can route this session's growing
    prefix to the same cache shard — much faster and cheaper after turn 1.

    By default the cache_key is the session_id, but a caller can pass a stable
    `cache_key` (e.g. derived from the working directory) to keep reusing a
    provider's prefix cache *without* resuming the prior conversation: that
    requires `resumed_from`, which is the only thing that loads past messages.
    """
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
        "log_path":        _open_log(model, session_id),
        "non_interactive": non_interactive,
        "usage_totals":    {"prompt": 0, "completion": 0, "total": 0,
                            "cached": 0, "cache_write": 0},
        "provider":        schema.get("provider"),
        "max_output_tokens": max_output_tokens or schema.get("max_output_tokens"),
        "on_event":        on_event or _default_event_handler,
        "streaming":       streaming,
        "options":         schema.get("options") or [],
        "session_start_ts": session_start_ts,
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

    _emit(session, "tool_result", name=name, result=result, streamed=streamed,
          fmt=fmt_result(result, streamed=streamed))
    _log(session, {"type": "tool_result", "name": name,
                   "python_function": entry.get("python_function"), **log_data,
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
            total_tokens += getattr(usage, "total_tokens", 0) or 0
            totals = session["usage_totals"]
            totals["prompt"]      += getattr(usage, "prompt_tokens", 0) or 0
            totals["completion"]  += getattr(usage, "completion_tokens", 0) or 0
            totals["total"]       += getattr(usage, "total_tokens", 0) or 0
            totals["cached"]      += getattr(usage, "cached_tokens", 0) or 0
            totals["cache_write"] += getattr(usage, "cache_creation_tokens", 0) or 0
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
        if total_tokens > max_tokens:
            _emit(session, "token_limit", used=total_tokens, limit=max_tokens,
                  fmt=f"\n[stopped after exceeding {max_tokens:,} tokens "
                      f"(used {total_tokens:,})]")
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
        _emit(session, "session_usage", **t,
              fmt=(f"{DIM}{MAG}[session tokens] prompt {t['prompt']:,}{cached_part}  |  "
                   f"completion {t['completion']:,}  |  total {t['total']:,}{RESET}\n"))
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
    """
    endpoint    = schema.get("endpoint") or DEFAULT_ENDPOINT
    binary_path = _parse_run_uri(endpoint) or _parse_run_uri(schema.get("model", ""))
    auth        = schema.get("auth")
    if binary_path is not None:
        return SubprocessOpenAI(binary_path)
    if auth == "opencode-github-copilot":
        return openai.OpenAI(api_key=_get_opencode_token(), base_url=endpoint,
                             auth_header="X-API-Key")
    return openai.OpenAI(api_key=_get_key_for_schema(schema), base_url=endpoint)


def run_task(
    schema: dict,
    task: str,
    *,
    non_interactive: bool = False,
    session_id: str | None = None,
    cache_key: str | None = None,
    system_prompt_supplement: str = "",
    max_output_tokens: int | None = None,
    on_event: "EventCallback | None" = None,
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
        on_event=on_event,
    )
    try:
        return run_turn(client, schema["model"], session, task)
    finally:
        _save_messages_snapshot(session)
        _log(session, {"type": "session_end", "session_id": session["session_id"],
                       "reason": "run_task_complete"})


def run_repl(
    schema: dict,
    *,
    non_interactive: bool = False,
    session_id: str | None = None,
    cache_key: str | None = None,
    system_prompt_supplement: str = "",
    max_output_tokens: int | None = None,
    on_event: "EventCallback | None" = None,
) -> None:
    """Start an interactive REPL session against the agent.

    Reads tasks line-by-line from stdin and runs :func:`run_turn` for each.
    The session snapshot is saved after every turn so it can be resumed with
    ``--session <session_id>``.
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
        on_event=on_event,
    )
    model      = schema["model"]
    resume_cmd = f"agent-probe {model} --session {session['session_id']}"

    if session_id:
        print_session_history(session)

    # Per-directory history so arrow-up recalls prompts from the same folder.
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

    print(f"{BOLD}probe-agent {model}{RESET}  (type 'exit' to quit)\n")
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
    finally:
        try:
            readline.write_history_file(_hist_file)
        except Exception:
            pass
        _save_messages_snapshot(session)
        _log(session, {"type": "session_end", "session_id": session["session_id"],
                       "reason": "repl_exit"})
        print(f"\n{DIM}Resume: {resume_cmd}{RESET}")


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

    resume_cmd = f"{sys.argv[0]} {model} --session {session['session_id']}"

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
    # Per-directory history so arrow-up recalls prompts from the same folder.
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

    print(f"{BOLD}probe-agent {model}{RESET}  (type 'exit' to quit)\n")
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
    finally:
        try:
            readline.write_history_file(_hist_file)
        except Exception:
            pass
        _save_messages_snapshot(session)
        _log(session, {"type": "session_end", "session_id": session["session_id"],
                       "reason": "repl_exit"})
        print(f"\n{DIM}Resume: {resume_cmd}{RESET}")


if __name__ == "__main__":
    main()
