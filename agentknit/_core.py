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
    API or dispatch error.  Data: ``text``, ``error_class``, ``http_status``,
    ``error_code``, ``error_message``, ``elapsed_s``, ``adapter``, ``fmt``.
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
``cache_cold``
    A resumed turn was served with no cache hit because the prefix cache
    had expired (last message older than ``CACHE_COLD_GAP_SECONDS``).
    Strict cache-proof enforcement is relaxed for this case.  Data:
    ``age``, ``fmt``.
``cache_proof_missing``
    A call after the first exposed no cache accounting / no cache hit while
    strict cache mode is on.  The turn continues automatically (the tokens
    are already paid); the warning is meant to be shown temporarily, e.g. in
    a status bar, until a later call reports a cache read or write (which
    sets ``session["_cache_status"]`` back to ``"ok"``).  Data:
    ``cached_tokens`` / ``prompt_tokens`` when available, ``fmt``.
``rate_limit_wait``
    Emitted before sleeping through a retryable HTTP 429.  Data:
    ``delay`` (seconds), ``resume_at`` (ISO timestamp), ``fmt``.

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
import time
import traceback
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, TypeAlias, TypedDict, cast

if TYPE_CHECKING:
    from typing import NotRequired, TextIO
    from .sandbox import ToolExecutor

from . import openai_compat as openai
from .openai_compat import SubprocessOpenAI

from . import tool_library as _tool_module
from .tool_library import TOOL_LIBRARY, _ASK_USER_FNS
from .tool import Tool, build_tool_spec, register_tools_in_library
from .exceptions import (
    AgentSpecDisabledError, AgentSpecInvalidError,
    PricingLimitExceededError, AuthenticationError, CacheProofError,
    ContextWindowExceededError, RateLimitError,
)
from .slash_commands import REGISTRY as _slash_registry
from .hooks import (
    HookDecision,
    HookEntry,
    canonical_tool_name,
    translate_updated_input,
    load_hooks as _load_hooks,
    run_hooks as _run_hooks,
)
from ._journal import (
    DurableSink,
    SessionJournal,
    new_call_id,
    replay_journal,
)


DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1"
DEFAULT_MAX_TOKENS = 3_000_000
LOG_BASE = Path.home() / ".local" / "share" / "agent_probe"

# Recovery note injected into the conversation when a resumed session's
# journal shows tool calls that started but never finished (crash mid-tool).
# Their side effects are unknown, so the model must verify rather than redo.
_PENDING_TOOL_NOTE = (
    "SYSTEM RECOVERY NOTE: The previous session crashed while these tool "
    "calls were in flight; their side effects are UNKNOWN:\n{calls}\n"
    "Verify the resulting state (inspect files, re-run read-only checks) "
    "before re-running any of them."
)

# Cap for the unreceived-results recovery note: the note exists to restore
# awareness of a handful of lost results, never to replay raw tool output —
# unbounded results (a long-running session can finish hundreds of calls)
# would otherwise dwarf the resumed context itself.
_UNRECEIVED_RESULTS_MAX_CHARS = 4000
_UNRECEIVED_RESULT_MAX_CHARS = 200


def _agentknit_commit() -> str:
    """Return the agentknit commit id (git HEAD) behind this process.

    Resolved from the package directory's enclosing git repository, so a
    snapshot can be traced back to the exact tool / runtime definitions in
    force when it was written.  Returns ``"unknown"`` when not run from a
    git checkout (e.g. pip-installed wheel).
    """
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parent.parent), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip() or "unknown"
    except Exception:
        return "unknown"

# Compaction threshold: chosen conservatively so it works across arbitrary
# endpoints (many models still have 128K–200K context windows).  For
# large-context models (1M+) this may trigger earlier than necessary; raise
# it in the agent spec or via compaction_trigger_tokens=….
DEFAULT_COMPACTION_TRIGGER_TOKENS = 100_000
DEFAULT_COMPACTION_TARGET_TOKENS = 20_000
DEFAULT_COMPACTION_KEEP_LAST_TURNS = 2

# Prefix-cache "cold resume" threshold.  Provider prefix caches expire after
# a few minutes (OpenAI ~5–10 min, Anthropic 5 min).  If the last message in
# a resumed session is older than this, the cache has almost certainly
# evaporated through no fault of the caller, so strict cache-proof
# enforcement would spuriously abort the turn.  Such turns are marked "cold"
# and warn instead of failing — see :func:`_enforce_cache_proof`.
CACHE_COLD_GAP_SECONDS = 3600

# Default minimum cacheable prompt size (in prompt tokens) used by
# _enforce_cache_proof() when a session doesn't declare its own
# min_cacheable_tokens.  0 means "no minimum" — any zero-cache response after
# the first call is treated as a real cache miss.  Providers enforce their
# own floor below which nothing is cached regardless of prompt content, e.g.
# Anthropic Claude Haiku ~4096 input tokens, GPT-5.6-class models ~1024.
# Set schema["min_cacheable_tokens"] (or pass min_cacheable_tokens=... to
# init_session/run_task/run) to the provider's documented floor so small
# prompts don't spuriously trip strict cache-proof mode.
DEFAULT_MIN_CACHEABLE_TOKENS = 0

# Token awareness (model-facing): inject the true token count into the
# model's own context so it can pace itself and checkpoint before
# compaction.  The default budget is the compaction trigger — literally
# the window the model experiences between compactions, so the number is
# both true and operative.  100k is a "reasonable" budget in TALE's terms
# (their token-elasticity backfires happened at 10–250 tokens).  The
# reminder threshold matches Codex's reminder_threshold_tokens.
DEFAULT_TOKEN_AWARENESS_REMINDER_TOKENS = 6144

_TOKEN_AWARENESS_SYSTEM = (
    "<budget:token_budget>{budget}</budget:token_budget>\n"
    "Your context window holds {budget} tokens; the usage counter after each "
    "tool call shows how full it currently is. When it fills, agentknit "
    "automatically compacts older context into a summary and you continue in "
    "fresh space — this is normal operation, not a deadline. Do not stop "
    "tasks early or take shortcuts due to context capacity; there is always "
    "enough room to finish properly."
)

# Matches an injected token-awareness warning inside a message, for
# stripping stale readings on resume (see _normalise_for_resume).
_TA_WARNING_RE = re.compile(
    r"\n*<system_warning>Token usage:[^<]*(?:</system_warning>)?"
    r"(?:\s*<context_window_reminder>.*?</context_window_reminder>)?",
    re.DOTALL,
)

_TOKEN_AWARENESS_REMINDER = (    "<context_window_reminder>\n"
    "Your context window is nearly full; {remaining} tokens remain before "
    "compaction. Write concise progress notes in your next reply — goal, "
    "decisions, progress, learnings, next steps — then keep working. "
    "Agentknit will compact earlier context into a summary and you continue "
    "in fresh space. Do not stop the task; compaction is normal operation, "
    "not a deadline.\n"
    "</context_window_reminder>"
)

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
    "Style:\n"
    "- Plain text with clear sections. Be concise but complete.\n"
    "- Summarize outcomes; quote logs only where the exact bytes matter.\n"
    "- Report each fact once. Keep uncertainty expressed as uncertainty.\n"
)

_PRE_COMPACTION_PROMPT = (
    "Context compaction is about to run: the earlier part of this "
    "conversation will soon be replaced by a summary. Before that happens, "
    "write down everything you will need to keep working on this task "
    "afterwards — key facts, decisions made, current state of the work, "
    "important file contents or identifiers, and the immediate next steps. "
    "Anything you do not record now may be lost. Reply briefly once done."
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

# ── OSC 8 terminal hyperlinks ─────────────────────────────────────────────────

_OSC8_URL_RE = re.compile(r"(https?://\S+)")


def _osc8_url(url: str) -> str:
    return f"\033]8;;{url}\033\\{url}\033]8;;\033\\"


class _Osc8StdoutWrapper:
    """Line-buffered stdout wrapper that rewrites bare URLs as OSC 8 hyperlinks."""

    def __init__(self, wrapped: "TextIO") -> None:
        self._w = wrapped
        self._buf = ""

    def _rewrite(self, text: str) -> str:
        return _OSC8_URL_RE.sub(lambda m: _osc8_url(m.group(1)), text)

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._w.write(self._rewrite(line) + "\n")
        return len(s)

    def flush(self) -> None:
        if self._buf:
            self._w.write(self._rewrite(self._buf))
            self._buf = ""
        self._w.flush()

    def __getattr__(self, name: str) -> object:
        return getattr(self._w, name)


def enable_osc8_hyperlinks() -> None:
    """Wrap sys.stdout so bare URLs are rendered as OSC 8 clickable hyperlinks."""
    if not isinstance(sys.stdout, _Osc8StdoutWrapper):
        sys.stdout = _Osc8StdoutWrapper(sys.stdout)


# ── Ctrl-C handling ───────────────────────────────────────────────────────────
# True while run_turn() is executing; False at the REPL prompt.
_in_turn: bool = False


def _sigint_handler(sig: int, frame: object) -> None:
    """SIGINT handler: kill the active subprocess and abort the turn.

    When the agent is executing a tool (run_turn is active), immediately
    SIGKILL the current subprocess (if any) then raise KeyboardInterrupt so
    run_turn unwinds back to the REPL.  When idle at the prompt, do nothing.

    Before aborting, Interrupt hooks fire (advisory, 1 s budget): they can
    record the interruption or clean up work a hook started, but cannot
    prevent the interrupt.
    """
    if not _in_turn:
        return
    _active_session = getattr(_sigint_handler, "session", None)
    if _active_session is not None:
        try:
            _fire_hooks(_active_session, "Interrupt",
                        turn_id=(_active_session.get("_hook_state") or {})
                        .get("turn_id"))
        except Exception:
            pass
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
EventCallback: TypeAlias = Callable[[str, "dict[str, Any]"], None]


def _default_event_handler(event_type: str, data: dict[str, Any]) -> None:
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


def _emit(session: Session, event_type: str, **data: Any) -> None:
    """Fire *event_type* through the session's registered event handlers.

    First calls any per-event-type handlers registered via :func:`subscribe`,
    then calls the generic ``on_event`` handler (or the default).

    .. seealso::

        :ref:`event-types` — full list of event types with descriptions.
    """
    # A durable sink is intentionally before every subscriber, including the
    # default terminal renderer.  A sink failure propagates and prevents the
    # event from being consumed.
    _persist_record(session, {"type": "event", "event_type": event_type,
                              "data": data})
    # Call per-event-type handlers first
    handlers = session.get("_event_handlers", {}).get(event_type, [])
    for handler in handlers:
        handler(event_type, data)
    # Then call the generic handler
    handler = session.get("on_event") or _default_event_handler  # type: ignore[truthy-function]
    handler(event_type, data)


def _rate_limit_wait_callback(session: Session) -> "Callable[[float, datetime.datetime, str], None]":
    """Build an ``on_rate_limit_wait`` callback that emits a ``rate_limit_wait`` event."""
    def _on_rate_limit_wait(delay: float, resume_at: datetime.datetime, fmt: str) -> None:
        _emit(session, "rate_limit_wait", delay=delay, resume_at=resume_at.isoformat(), fmt=fmt)
    return _on_rate_limit_wait


def subscribe(session: Session, event_type: str, handler: EventCallback) -> None:
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


def unsubscribe(session: Session, event_type: str, handler: EventCallback) -> None:
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
    creds = cast(dict[str, Any], _json.loads(auth_json.read_text()))
    gh = creds.get("github-copilot")
    token = str(gh.get("access")) if isinstance(gh, dict) and gh.get("access") else None
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


_DEFAULT_TOOL_SCHEMA: list[dict[str, Any]] = [
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
            "description": (
                "Edit an existing file by replacing a specific substring. "
                "By default replaces only the first occurrence; pass "
                "replace_all=true to replace every occurrence."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_str": {"type": "string"},
                    "new_str": {"type": "string"},
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace every occurrence of old_str instead of only the first (default false).",
                    },
                },
                "required": ["path", "old_str", "new_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "exec_shell",
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
    "exec_shell":           {"python_function": "t_run",         "param_map": {}},
}

# Retired tool names still accepted as aliases of their renamed successor.
# Specs and sessions built before the rename keep working: the alias is
# expanded into the canonical dispatch entry at session start.
_LEGACY_TOOL_ALIASES: "dict[str, str]" = {
    "execute_shell_command": "exec_shell",
}
_DEFAULT_TOOLS = [
    "t_read",
    "t_write",
    "t_update",
    "t_run",
]


# ── spec loading ──────────────────────────────────────────────────────────────

def _load_spec_file(path: Path) -> "dict[str, Any]":
    """Read a spec JSON *path*, raising a typed error if unreadable."""
    try:
        with path.open() as f:
            data = cast(dict[str, Any], json.load(f))
    except FileNotFoundError:
        raise AgentSpecInvalidError(
            f"Spec file not found: {path}", model=path.name,
        ) from None
    except json.JSONDecodeError as e:
        raise AgentSpecInvalidError(
            f"Spec file {path} is not valid JSON: {e}", model=path.name,
        ) from None
    print(f"{DIM}Using schema file {path.name}{RESET}")
    return data


def load_specification(model: str, endpoint: str, spec_path: str | None = None) -> "dict[str, Any]":
    """Load an agent spec for `model`, without ever probing the model itself.

    agentknit only consumes specs — it does not generate them by talking to a
    model. This resolves, in order:

    1. An explicit ``spec_path``: that file is read (absolute, or relative
       to the current working directory) and returned as-is, skipping all
       name-based lookup. Composes with `run://` endpoints.
    2. A ``run://`` URI (in `endpoint` or `model`): looks for a cached
       ``agent_spec_<model>.json`` next to the binary; if absent returns an
       in-memory default spec (structured tool calls, default tool set).
    3. A direct path to a ``.json`` schema file (`model` ending in
       ``.json``): loaded and returned as-is. Relative paths resolve
       against the package root, so prefer ``spec_path``.
    4. A cached spec file for `model`, checked in order under the package
       directory as ``agent_spec_<model>.json``,
       ``inferred_tool_schema_<model>.json``, then
       ``tool_schema_<model>.json``.
    5. A spec file for `model` in the current working directory.
    6. If none exists and `endpoint` is given: returns an in-memory default
       spec.
    7. Otherwise: raises `AgentSpecInvalidError` telling the caller to run
       an external probing tool (e.g. `llmprobe`) to generate a real spec.

    :param model: Model identifier, a `run://` URI, or a path to a `.json`
        spec file.
    :param endpoint: Base URL of the OpenAI-compatible endpoint, or a
        `run://` URI.
    :param spec_path: Optional explicit path to a spec JSON file. When set,
        no API probing or name-based lookup happens.
    :returns: The loaded (or freshly generated default) spec as a dict.
    :raises AgentSpecInvalidError: If ``spec_path`` does not exist or is not
        valid JSON; or if no cached spec exists and `endpoint` is falsy, so
        no default spec can be generated either.
    """
    # Spec JSON files live in the project root (parent of the package directory).
    here = Path(__file__).resolve().parent.parent

    # Explicit spec path: read it, skip every other resolution step.
    if spec_path:
        return _load_spec_file(Path(spec_path).expanduser())

    # run:// URI → subprocess binary; use cached spec or return in-memory default.
    # Accept run:// in either endpoint or model (CLI convenience).
    binary_path = _parse_run_uri(endpoint) or _parse_run_uri(model)
    if binary_path is not None:
        # Normalise: model gets the bare path, endpoint gets the run:// URI.
        run_endpoint = f"run://{binary_path}"
        if _parse_run_uri(model):
            model = binary_path
        endpoint = run_endpoint
        # Check for an existing spec next to the binary.
        spec_dir = Path(binary_path).resolve().parent
        path = spec_dir / f"agent_spec_{safe_model_name(model)}.json"
        if path.exists():
            print(f"{DIM}Using specification from {path.name}{RESET}")
            with path.open() as f:
                return cast(dict[str, Any], json.load(f))
        return {
            "model":       model,
            "endpoint":    endpoint,
            "status":      "default",
            "tool_specs":  _DEFAULT_TOOL_SCHEMA,
            "tools":       _DEFAULT_TOOLS,
            "behaviour":   {"call_delivery_mode": "structured_tool_calls"},
        }

    # Accept a direct path to a JSON schema file.
    if model.endswith(".json"):
        path = Path(model)
        if not path.is_absolute():
            path = here / path
        return _load_spec_file(path)

    # Check known shipped spec-files in the package directory first.
    for candidate in (
        here / f"agent_spec_{safe_model_name(model)}.json",
        here / f"inferred_tool_schema_{safe_model_name(model)}.json",
        here / f"tool_schema_{safe_model_name(model)}.json",
    ):
        if candidate.exists():
            print(f"{DIM}Using cached probe at {candidate.name}{RESET}")
            with candidate.open() as f:
                return cast(dict[str, Any], json.load(f))

    # Also check the working directory for a user-created spec.
    cwd_path = Path.cwd() / f"agent_spec_{safe_model_name(model)}.json"
    if cwd_path.exists():
        print(f"{DIM}Using specification from {cwd_path.name}{RESET}")
        with cwd_path.open() as f:
            return cast(dict[str, Any], json.load(f))

    if endpoint:
        return {
            "model": model,
            "endpoint": endpoint,
            "status": "default",
            "tool_specs": _DEFAULT_TOOL_SCHEMA,
            "tools": _DEFAULT_TOOLS,
            "behaviour": {"call_delivery_mode": "structured_tool_calls"},
        }

    raise AgentSpecInvalidError(
        f"No agent spec found for '{model}'. "
        f"Run `llmprobe {model}` (with --endpoint {endpoint} if needed) "
        f"to probe the model and generate a spec file.",
        model=model,
    )


# ── dispatch ──────────────────────────────────────────────────────────────────

class FatalToolDispatchError(RuntimeError):
    """Raised when the agent requests a tool that cannot be dispatched."""


def _tool_name_from_spec(tool_spec: dict[str, Any]) -> str:
    """Return the model-facing tool name from an OpenAI-compatible tool spec.

    Handles function specs (``{"type": "function", "function": {"name"}}``)
    and custom specs (``{"type": "custom", "name"}``).
    """
    if tool_spec.get("type") == "custom":
        return str(tool_spec.get("name", ""))
    fn = tool_spec.get("function") or tool_spec
    if not isinstance(fn, dict):
        return ""
    return str(fn.get("name", ""))


def _tool_param_names(tool_spec: "dict[str, Any]") -> list[str]:
    """Return the model-facing parameter names from *tool_spec* in declaration order.

    Custom tools (``type == "custom"``) take a single raw-text argument and
    report no JSON-Schema properties.
    """
    if tool_spec.get("type") == "custom":
        return []
    fn = tool_spec.get("function") or tool_spec
    params = ((fn.get("parameters") or {}).get("properties") or {})
    return list(params.keys())


def _callable_param_names(fn: Callable[..., object]) -> list[str]:
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


def _derive_param_map(tool_spec: "dict[str, Any]", fn_name: str) -> dict[str, str]:
    """Infer model-arg -> Python-kwarg mapping for *fn_name* and *tool_spec*."""
    fn = TOOL_LIBRARY.get(fn_name)
    if fn is None:
        raise AgentSpecInvalidError(
            f"Tool function {fn_name!r} not found in TOOL_LIBRARY.",
            model=fn_name,
        )

    if tool_spec.get("type") == "custom":
        # Custom tool: the model's argument is raw text delivered as the
        # single `input` kwarg (see Tool(custom_format=...)).
        return {"input": "input"}

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


def _build_dispatch_from_tools(tool_specs: list[dict[str, Any]], tools: list[str]) -> dict[str, dict[str, Any]]:
    """Build a tool_dispatch dict from ordered tool specs and TOOL_LIBRARY names."""
    if len(tool_specs) != len(tools):
        raise AgentSpecInvalidError(
            f"'tool_specs' has {len(tool_specs)} entries but 'tools' has {len(tools)}.",
        )

    dispatch: dict[str, dict[str, Any]] = {}
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


def _default_dispatch_for_tool_specs(tool_specs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return built-in dispatch entries for tool specs whose names are known defaults."""
    dispatch: dict[str, dict[str, Any]] = {}
    for tool_spec in tool_specs:
        tool_name = _tool_name_from_spec(tool_spec)
        entry = _DEFAULT_TOOL_DISPATCH.get(tool_name)
        if entry is None:
            # Legacy tool names resolve to the renamed successor's dispatch entry.
            alias_target = _LEGACY_TOOL_ALIASES.get(tool_name)
            entry = _DEFAULT_TOOL_DISPATCH.get(alias_target) if alias_target else None
        if entry is not None:
            dispatch[tool_name] = copy.deepcopy(entry)
    return dispatch


def _normalize_schema(schema: "dict[str, Any]") -> "dict[str, Any]":
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


def _resolve_fn(entry: dict[str, Any]) -> Callable[..., Any] | None:
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
        return cast(Callable[..., Any], pf)
    # string name → look up in TOOL_LIBRARY
    return TOOL_LIBRARY.get(str(pf))


def dispatch(tool_name: str, args: dict[str, Any], tool_dispatch: dict[str, Any]) -> tuple[str, dict[str, Any]]:
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
        # Argument mismatch (missing/wrongly-named/unexpected parameter) is
        # the model's fault — describe the expected signature so it can fix
        # and retry the call in the next iteration.
        import inspect
        try:
            sig = str(inspect.signature(fn))
        except (TypeError, ValueError):
            sig = "(...)"
        r = (f"ERROR: invalid arguments for tool {tool_name!r}: {e}. "
             f"Expected signature: {fn_name}{sig}. "
             f"You supplied: {sorted(args.keys())}. "
             f"Correct the argument names/values and call the tool again.")
        return r, {"result": r}
    except Exception as e:
        # Internal tool failure — include the exception type (not just
        # str(e)) plus the innermost frames so bugs like "'bool' object has
        # no attribute 'splitlines'" are immediately locatable.
        tb = traceback.extract_tb(sys.exc_info()[2])
        inner = ", ".join(f"{os.path.basename(f.filename)}:{f.lineno} in {f.name}"
                          for f in tb[-3:])
        r = (f"ERROR: tool {tool_name!r} raised {type(e).__name__}: {e} "
             f"({fn_name}(**{kwargs!r}); {inner})")
        return r, {"result": r}

    # All library functions return (str, dict); handle plain str just in case.
    if isinstance(result, tuple):
        text, meta = result
        # Coerce non-str results (e.g. a tool returning True instead of a
        # string) — downstream formatters assume str and would crash.
        if not isinstance(text, str):
            text = str(text)
            meta = {**meta, "result": text}
        return text, meta
    return str(result), {"result": str(result)}


# ── schema helpers ────────────────────────────────────────────────────────────

def schema_props(tool: dict[str, Any]) -> dict[str, Any]:
    if tool.get("type") == "custom":
        # Custom tools have a single raw-text `input` argument.
        return {"input": {"type": "string"}}
    fn = tool.get("function") or tool
    params = fn.get("parameters") or {}
    props = params.get("properties")
    if not isinstance(props, dict):
        props = {k: v for k, v in params.items()
                 if isinstance(v, dict) and "type" in v}
    return props


# ── inline-JSON tool-call extraction (multi-call safe) ───────────────────────

_decoder = json.JSONDecoder()

def extract_inline_calls(text: str) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
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

def fmt_call(name: str, args: dict[str, Any]) -> str:
    pretty = ", ".join(f"{k}={v!r}" for k, v in args.items())
    if len(pretty) > 400:
        pretty = pretty[:400] + "…"
    return f"{CYAN}{BOLD}▶ {name}({pretty}){RESET}"

def fmt_usage(usage: object, *, compaction_trigger: int | None = None) -> str:
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
    if compaction_trigger:
        compact_pct = (prompt / compaction_trigger * 100) if compaction_trigger else 0
        parts.append(f"compact {compact_pct:.0f}%")
    parts.append(f"completion {completion:,}")
    parts.append(f"total {total:,}")
    return "  |  ".join(parts)


def _last_message_age_seconds(session: Session) -> float | None:
    """Seconds since the last message in the session, or None if unmeasurable.

    When resuming a session after a break, the provider's prefix cache may
    have expired (OpenAI caches live ~5–10 min, Anthropic ~5 min).  This
    lets the caller distinguish a genuine cache miss from a cold resume.
    """
    messages = session.get("messages") or []
    last_ts = None
    for msg in reversed(messages):
        ts = msg.get("ts") if isinstance(msg, dict) else None
        if ts:
            last_ts = ts
            break
    if not last_ts:
        return None
    try:
        last_dt = datetime.datetime.fromisoformat(last_ts)
    except (ValueError, TypeError):
        return None
    now = datetime.datetime.now()
    return (now - last_dt).total_seconds()


def _enforce_cache_proof(session: Session, usage: object) -> None:
    """Fail closed at the start of the session; warn-and-continue afterwards.

    Strict cache mode only *raises* when the very first LLM call exposes no
    cache-proof field: caching is then not working at all, and aborting is
    cheap because nothing beyond one call has been paid for.  From the
    second call on, the token price has already been paid, so a missing
    cache proof is downgraded to a ``cache_proof_missing`` warning event and
    the turn continues automatically.  ``session["_cache_status"]`` is set
    to ``"missing"`` so a UI can show a temporary status-bar warning; it
    flips to ``"ok"`` (clearing the warning) as soon as any call reports a
    cache read or write.

    A resumed session whose last message is older than
    :data:`CACHE_COLD_GAP_SECONDS` is assumed to be a *cold resume*: the
    provider's prefix cache has expired through no fault of the caller, so
    the first post-resume call is allowed to miss without even a warning
    escalation.  A dim notice is emitted instead so the output is not broken.

    Below ``session["min_cacheable_tokens"]`` prompt tokens, providers cache
    nothing by design (e.g. Anthropic Claude Haiku ~4096, GPT-5.6-class
    ~1024) — such calls report ``cached_tokens == 0`` even though caching
    works fine for larger prompts.  If the current call's prompt is below
    that floor, a zero-cache response is treated as expected and skipped
    rather than warned about.  See :data:`DEFAULT_MIN_CACHEABLE_TOKENS`.
    """
    if not session.get("strict_cache_proof", True):
        return

    has_cache_proof = getattr(usage, "has_cache_proof", False)
    cached_tokens = getattr(usage, "cached_tokens", 0) or 0
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    cache_creation = getattr(usage, "cache_creation_tokens", 0) or 0

    # A genuine cache read or write proves prefix caching works; remember it
    # so any temporary "no cache proof" status-bar warning can clear itself.
    if has_cache_proof and (cached_tokens > 0 or cache_creation > 0):
        session["_cache_status"] = "ok"

    # Beginning of the session: the first call must expose cache accounting,
    # otherwise strict cache mode cannot work at all.  Aborting here is
    # cheap — nothing beyond this call has been paid for.  Providers with a
    # minimum cacheable prefix (session["min_cacheable_tokens"]) are exempt
    # when the first prompt is below that floor: their cache silently skips
    # small prompts, so the absence of accounting is expected, not broken.
    if session.get("llm_call_count", 0) <= 1:
        if not has_cache_proof and cache_creation <= 0:
            min_cacheable = session.get("min_cacheable_tokens", DEFAULT_MIN_CACHEABLE_TOKENS) or 0
            if min_cacheable and prompt_tokens < min_cacheable:
                return
            raise CacheProofError(
                "Strict cache mode requires explicit cache accounting from the server "
                "on the first LLM call, but this response exposed no cache-proof field."
            )
        return

    # Detect a cold resume: the caller paused long enough that the provider's
    # prefix cache has surely expired.  Don't break the turn for something
    # outside the caller's control; warn once instead.
    age = _last_message_age_seconds(session)
    cold_resume = age is not None and age > CACHE_COLD_GAP_SECONDS
    if cold_resume and not (has_cache_proof and cached_tokens > 0):
        notice = (
            f"{DIM}Prefix cache expired (last message {int(age or 0)}s old); "
            f"this turn was not served from cache and will re-process the prompt.{RESET}"
        )
        _emit(session, "cache_cold", age=int(age or 0), fmt=notice)
        session["_cache_cold_warned"] = True
        return

    if not has_cache_proof:
        # Past the first call the token price is already paid; aborting would
        # only waste it.  Continue automatically with a temporary warning.
        session["_cache_status"] = "missing"
        notice = (
            f"{YEL}⚠ No cache accounting from the server after the first call; "
            f"continuing without strict cache proof (tokens for this turn are "
            f"already paid). Warning clears on the next observed cache hit.{RESET}"
        )
        _emit(session, "cache_proof_missing", cached_tokens=cached_tokens,
              prompt_tokens=prompt_tokens, fmt=notice)
        return
    if cached_tokens <= 0 and cache_creation <= 0:
        min_cacheable = session.get("min_cacheable_tokens", DEFAULT_MIN_CACHEABLE_TOKENS) or 0
        if min_cacheable and prompt_tokens < min_cacheable:
            notice = (
                f"{DIM}No cache hit, but prompt ({prompt_tokens} tokens) is below the "
                f"provider's minimum cacheable prefix ({min_cacheable} tokens); "
                f"not a caching failure.{RESET}"
            )
            _emit(session, "cache_below_minimum", prompt_tokens=prompt_tokens,
                  min_cacheable_tokens=min_cacheable, fmt=notice)
            return
        # Some servers only cache eligible prefixes of at least N tokens (e.g.
        # Anthropic Claude Haiku ~4096, GPT-5.6-class models ~1024). Configure
        # session["min_cacheable_tokens"] to that floor to avoid warning on
        # legitimately small prompts.  As above, warn instead of aborting —
        # the turn's tokens are already paid for.
        session["_cache_status"] = "missing"
        notice = (
            f"{YEL}⚠ No cache hit after the first call "
            f"(prompt {prompt_tokens:,} tokens); continuing without strict cache "
            f"proof. Warning clears on the next observed cache hit.{RESET}"
        )
        _emit(session, "cache_proof_missing", cached_tokens=cached_tokens,
              prompt_tokens=prompt_tokens, fmt=notice)
        return
    if cached_tokens <= 0 and cache_creation > 0:
        # A cache WRITE is just as much proof that prefix caching works: this is
        # the first call whose prefix crossed the provider's minimum cacheable
        # size, so the cache is being populated rather than read. The next call
        # in the same session will read it back.
        notice = (
            f"{DIM}Prefix cache written this call ({cache_creation:,} tokens); "
            f"the next call will read it back.{RESET}"
        )
        _emit(session, "cache_written", cache_creation_tokens=cache_creation, fmt=notice)


def fmt_result(text: str, streamed: bool = False) -> str:
    if streamed:
        # Output was already streamed to console in real-time; just show a
        # short summary instead of repeating the full content.
        return DIM + "  (output streamed above)" + RESET
    lines = text.splitlines()
    head = lines[:40]
    tail = f"\n{DIM}  … ({len(lines)-40} more lines){RESET}" if len(lines) > 40 else ""
    return DIM + "\n".join("  " + line for line in head) + RESET + tail


def fmt_read_result_with_command(command: str, text: str, streamed: bool = False) -> str:
    reminder = f"{YEL}{BOLD}  shell output from:{RESET} {YEL}{command}{RESET}\n"
    return reminder + fmt_result(text, streamed=streamed)


def inline_system_prompt(tools: list[dict[str, Any]]) -> str:
    examples = []
    for tool in tools:
        if tool.get("type") == "custom":
            # Custom tools have a raw-text argument; the inline JSON
            # protocol carries it as the "input" string.
            examples.append(json.dumps(
                {"name": tool.get("name", "?"), "arguments": {"input": "<text>"}}))
            continue
        fn = tool.get("function") or tool
        name = fn.get("name", "?")
        arg_obj = {k: f"<{k}>" for k in schema_props(tool)}
        examples.append(json.dumps({"name": name, "arguments": arg_obj}))
    return (
        "Tool calls are made by responding with a single JSON object on its "
        "own line:\n"
        + "\n".join(examples) + "\n\n"
        "After each call you will receive the result. When the task is done, "
        "respond with a plain-text summary.\n"
    )


def _git_config_value(key: str) -> "str | None":
    """Read one git config value; None when git or the value is absent."""
    import subprocess
    try:
        out = subprocess.run(
            ["git", "config", "--get", key],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = out.stdout.strip()
    return value or None


def _git_root() -> Path:
    """Root of the enclosing git work tree, or cwd when not in one."""
    import subprocess
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return Path.cwd()
    root = out.stdout.strip()
    return Path(root) if root else Path.cwd()


def _git_status_block() -> "str | None":
    """Git status lines for the system prompt; None outside a git repo."""
    import subprocess
    try:
        inside = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, timeout=5,
        )
        if inside.returncode != 0:
            return None
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        changed = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        ).stdout.splitlines()
        last_commit = subprocess.run(
            ["git", "log", "-1", "--pretty=format:%s"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = [f"Git: on branch {branch or '(unknown)'}"]
    if last_commit:
        lines.append(f"  last commit: {last_commit}")
    if changed:
        lines.append(f"  changed files ({len(changed)}):")
        lines.extend(f"    {line}" for line in changed[:20])
        if len(changed) > 20:
            lines.append(f"    … ({len(changed) - 20} more)")
    else:
        lines.append("  working tree clean")
    return "\n".join(lines)


def _scratchpad_dir(cwd: Path) -> Path:
    """Per-working-directory scratchpad under the temp dir.

    Slugged with the basename plus a short hash of the full path so two
    projects sharing a directory name never collide.
    """
    import hashlib
    import tempfile
    digest = hashlib.sha1(str(cwd).encode()).hexdigest()[:8]
    return Path(tempfile.gettempdir()) / f"agentknit-scratchpad-{cwd.name}-{digest}"


# ── hooks integration ─────────────────────────────────────────────────────────

def _hooks_enabled(session: Session) -> bool:
    """True when the session has hooks and they are not disabled."""
    return bool(session.get("hooks_enabled", True)) and bool(session.get("hooks"))


def _hook_notify(session: Session) -> "Callable[[str, dict[str, Any]], None]":
    """Event callback for hook infrastructure messages (status/warning)."""
    def _notify(event_type: str, data: dict[str, Any]) -> None:
        _emit(session, event_type, **data)
    return _notify


def _fire_hooks(session: Session, event: str, matcher_values: "list[str]" = [],
                **extra: Any) -> HookDecision:
    """Run the session's matching hooks for *event*; never raises.

    Builds the common input payload (the shared Claude/Codex field set plus
    agentknit aliases), dispatches through :func:`agentknit.hooks.run_hooks`,
    surfaces ``systemMessage`` and errors as ``hook_warning`` events and
    ``hook_error`` log records, and queues ``additionalContext`` as pending
    model-facing context.  Returns the combined decision (an empty
    ``HookDecision`` when hooks are disabled or none match).
    """
    if not _hooks_enabled(session):
        return HookDecision()
    cwd = str(session.get("_cwd") or Path.cwd())
    payload: dict[str, Any] = {
        "session_id": session.get("session_id", ""),
        "transcript_path": str(session.get("log_path") or "") or None,
        "cwd": cwd,
        "hook_event_name": event,
        "model": session.get("model", ""),
        "permission_mode": "dontAsk" if session.get("non_interactive") else "default",
        "scratchpad_dir": str(_scratchpad_dir(Path(cwd))),
        "prompt_id": (session.get("_hook_state") or {}).get("turn_id"),
    }
    payload.update(extra)
    state = session.setdefault("_hook_state", {})
    decision = _run_hooks(
        session.get("hooks") or [], event, payload,
        cwd=cwd, matcher_values=matcher_values, state=state,
        spill_dir=session.get("session_dir") or (session.get("log_path") or Path()).parent,
        notify=_hook_notify(session),
    )
    if decision.error:
        _emit(session, "hook_warning", text=decision.error,
              fmt=f"{YEL}⚠ hook error: {decision.error}{RESET}")
        _log(session, {"type": "hook_error", "event": event,
                       "error": decision.error,
                       "ts": datetime.datetime.now().isoformat(timespec="seconds")})
    if decision.system_message and decision.error != decision.system_message:
        _emit(session, "hook_warning", text=decision.system_message,
              fmt=f"{YEL}[hook] {decision.system_message}{RESET}")
    if decision.additional_context:
        state.setdefault("pending_context", []).append(decision.additional_context)
    return decision


def _hook_tool_payload_fields(session: Session, name: str, args: dict[str, Any],
                              call_id: str) -> dict[str, Any]:
    """Event-specific fields for PreToolUse / PostToolUse."""
    return {
        "tool_name": canonical_tool_name(name),
        "agentknit_tool_name": name,
        "tool_input": args,
        "tool_use_id": call_id,
    }


def _drain_pending_hook_context(session: Session) -> str | None:
    """Pop queued ``additionalContext`` from async hooks, if any."""
    state = session.get("_hook_state") or {}
    queued = state.pop("async_results", None) or []
    texts = [q["context"] for q in queued if q.get("context")]
    for q in queued:
        if q.get("system_message") and not _hooks_enabled(session):
            pass
    if not texts:
        return None
    return "\n".join(texts)


def _cpu_count() -> "int | None":
    """Usable cores: scheduler affinity when available, else os.cpu_count()."""
    import os
    try:  # Linux containers: affinity may be lower than the machine's cores.
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


def _total_ram_gib(meminfo: "Path | None" = None) -> "float | None":
    """Total physical RAM in GiB, best-effort and dependency-free."""
    try:  # Linux.
        with (meminfo or Path("/proc/meminfo")).open() as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kib = float(line.split()[1])
                    return kib / (1024 * 1024)
    except OSError:
        pass
    try:  # macOS.
        import subprocess
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                             text=True, timeout=5)
        if out.returncode == 0:
            return int(out.stdout.strip()) / (1024 ** 3)
    except Exception:
        pass
    return None


def environment_context(model: str, version: "str | None" = None) -> str:
    """Build the environment-awareness block appended to the system prompt.

    Covers (per issue #31): user identity, git status, working directory,
    OS/architecture, CPU/RAM, current date & timezone, scratchpad dir,
    model identity.
    """
    import getpass
    import platform

    lines = ["## Environment"]

    # User identity: unix name + git identity when configured.
    try:
        unix_name = getpass.getuser()
    except Exception:
        unix_name = None
    git_name = _git_config_value("user.name")
    git_email = _git_config_value("user.email")
    identity = ""
    if unix_name:
        identity += f"unix user: {unix_name}"
    if git_name or git_email:
        git_id = " ".join(x for x in (git_name, f"<{git_email}>" if git_email else "") if x)
        identity += ("; " if identity else "") + f"git identity: {git_id}"
    if identity:
        lines.append(f"User: {identity}")

    git_block = _git_status_block()
    if git_block:
        lines.append(git_block)

    lines.append(f"Working directory: {Path.cwd()}")
    lines.append(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    cpus = _cpu_count()
    if cpus:
        lines.append(f"CPU cores: {cpus}")
    ram = _total_ram_gib()
    if ram:
        lines.append(f"RAM: {ram:.1f} GiB")

    now = datetime.datetime.now().astimezone()
    lines.append(f"Current date/time: {now.strftime('%Y-%m-%d %H:%M:%S')} "
                 f"({now.tzname() or 'local'} timezone)")

    scratchpad = _scratchpad_dir(Path.cwd())
    scratchpad.mkdir(parents=True, exist_ok=True)
    lines.append(f"Scratchpad (for temporary files): {scratchpad}")

    model_line = f"Model: {model}"
    if version:
        model_line += f" (version {version})"
    lines.append(model_line)

    # Harness identity: local import avoids a circular import with __init__.
    from . import __version__ as _harness_version
    lines.append(f"Harness: agentknit {_harness_version}")

    lines.append("")
    lines.append(attribution_block(model))

    return "\n".join(lines)


def attribution_block(model: str) -> str:
    """Git/PR attribution instructions appended to the system prompt."""
    return (
        "## Attribution\n"
        "Attribution for git commits and pull requests you create from here on:\n"
        "- End git commit messages with:\n"
        f"Co-Authored-By: agentknit+{model} <agentknit+{model}@monperrus.com>\n"
        "- End pull request descriptions with:\n"
        "🤖 Generated with [agentknit](https://github.com/monperrus/agentknit)"
    )


def read_repl_input(prompt: str) -> str:
    """Read one REPL task, coalescing multiline clipboard paste into one turn."""
    # Do not use input() here. With readline enabled, input() can read ahead
    # into readline's private buffer. select() cannot see those buffered paste
    # lines, so a multiline paste was split into separate REPL turns.
    print(prompt.replace("\x01", "").replace("\x02", ""), end="", flush=True)
    first_line = sys.stdin.readline()
    if first_line == "":
        raise EOFError

    text = first_line.rstrip("\n")
    # If paste arrives line-by-line, keep draining until input has been idle briefly.
    while select.select([sys.stdin], [], [], PASTE_IDLE_TIMEOUT_S)[0]:
        line = sys.stdin.readline()
        if line == "":
            break
        text += "\n" + line.rstrip("\n")
    text = text.rstrip("\n")
    if text:
        readline.add_history(text)
    return text


def print_session_history(session: Session) -> None:
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
                    custom = tc.get("custom")
                    if custom:
                        # Custom tool call: raw text input, no JSON args.
                        print(fmt_call(custom.get("name", "?"),
                                       {"input": custom.get("input", "")}))
                        continue
                    fn   = tc.get("function") or {}
                    name = fn.get("name", "?")
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        args = {}
                    if not isinstance(args, dict):
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

def _open_log(model: str, session_id: str, session_dir: str | Path | None = None) -> Path:
    now = datetime.datetime.now()
    if session_dir is not None:
        path = Path(session_dir) / "events.jsonl"
    else:
        path = (LOG_BASE / safe_model_name(model)
                         / now.strftime("%Y-%m-%d")
                         / f"{now.strftime('%H%M%S')}_{session_id}.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _token_awareness_injection(session: Session, usage: object) -> str | None:
    """Model-facing countdown text to suffix onto the next tool result.

    Derived exclusively from the server-reported prompt size of the last
    call — never padded or fabricated (a fake counter is a bug, not a
    feature).  Returns None when the feature is off, when usage data is
    missing, or when this call is skipped by ``update_every``.
    """
    if not session.get("token_awareness_enabled"):
        return None
    prompt_tok = getattr(usage, "prompt_tokens", 0) or 0
    if prompt_tok <= 0:
        return None
    budget = session.get("token_awareness_budget_tokens", 0) or 0
    if budget <= 0:
        return None
    update_every = max(1, session.get("token_awareness_update_every", 1) or 1)
    if session.get("llm_call_count", 0) % update_every != 0:
        return None
    remaining = max(0, budget - prompt_tok)
    reminder_threshold = session.get("token_awareness_reminder_tokens", 0) or 0
    # Edge-triggered: fire the checkpoint reminder once when remaining
    # crosses below the threshold; re-arm after it rises back above
    # (post-compaction, the countdown re-opens naturally).
    last_remaining = session.get("token_awareness_last_remaining")
    below = remaining < reminder_threshold
    crossed = below and not (
        last_remaining is not None and last_remaining < reminder_threshold)
    session["token_awareness_last_remaining"] = remaining
    _emit(session, "token_budget", used=prompt_tok, budget=budget,
          remaining=remaining, below_reminder_threshold=below,
          fmt=f"{DIM}{MAG}[budget] {remaining:,}/{budget:,} tokens remaining"
              f"{' (below reminder threshold)' if below else ''}{RESET}")
    warning = (f"<system_warning>Token usage: {prompt_tok}/{budget}; "
               f"{remaining} remaining</system_warning>")
    if crossed:
        warning += "\n" + _TOKEN_AWARENESS_REMINDER.format(remaining=remaining)
    return warning


def _log(session: Session, record: "dict[str, Any]") -> None:
    record["ts"] = datetime.datetime.now().isoformat(timespec="seconds")
    record["cwd"] = os.getcwd()
    with session["log_path"].open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if session.get("_durable_capture"):
            f.flush()
            os.fsync(f.fileno())


def _persist_record(session: Session, record: "dict[str, Any]") -> None:
    """Commit one lifecycle record before its producer exposes it.

    The built-in journal is the authoritative recovery stream.  An optional
    caller sink receives the same record synchronously, allowing a consumer
    to mirror or replace storage without patching agent internals.
    """
    journal = session.get("_journal")
    if session.get("_durable_capture") and journal is not None:
        journal.append(record)
    sink = session.get("durable_sink")
    if sink is not None and sink is not journal:
        sink.append(dict(record))


def _write_journal_record(session: Session, record: "dict[str, Any]") -> None:
    """Write a recovery record and mirror it to the optional public sink."""
    journal = session.get("_journal")
    if journal is not None:
        journal.append(record)
    sink = session.get("durable_sink")
    if sink is not None and sink is not journal:
        sink.append(dict(record))


def _snapshot_path(model: str, session_id: str,
                   session_dir: str | Path | None = None) -> Path:
    if session_dir is not None:
        return Path(session_dir) / "messages.json"
    return LOG_BASE / safe_model_name(model) / f"{session_id}_messages.json"


def _journal_path(model: str, session_id: str,
                  session_dir: str | Path | None = None) -> Path:
    if session_dir is not None:
        return Path(session_dir) / "journal.jsonl"
    return LOG_BASE / safe_model_name(model) / f"{session_id}_journal.jsonl"


def _save_messages_snapshot(session: Session) -> None:
    # Only save if there is at least one non-system message worth resuming.
    if not any(m.get("role") != "system" for m in session["messages"]):
        return
    path = _snapshot_path(session["model"], session["session_id"],
                          session.get("session_dir"))
    path.parent.mkdir(parents=True, exist_ok=True)
    # Annotate each message with a timestamp (backward-compatible: existing
    # messages that already have a "ts" key are left unchanged).
    annotated = []
    for m in session["messages"]:
        entry = dict(m)
        if "ts" not in entry:
            entry["ts"] = datetime.datetime.now().isoformat(timespec="seconds")
        annotated.append(entry)
    # Tool provenance: whether the default tool schema was used (no explicit
    # tools given) plus the resolved tool-name list, so a snapshot records
    # exactly which tools the model had.  Traceability of their definitions
    # is provided by the agentknit commit id below.
    tool_specs: Any = session.get("tools")
    default_names = [t["function"]["name"] for t in _DEFAULT_TOOL_SCHEMA]
    if tool_specs:
        tool_names = [((t.get("function") or t).get("name") or "?") for t in tool_specs]
        default_tools = tool_names == default_names
    else:
        # No tools given → the runtime used the default tool set; record it
        # explicitly so the snapshot is self-describing.
        tool_names = list(default_names)
        default_tools = True
    payload = {
        "metadata": {
            "endpoint": session.get("endpoint", ""),
            "model": session["model"],
            "session_id": session["session_id"],
            "default_tools": default_tools,
            "tools": tool_names,
            "agentknit_commit": _agentknit_commit(),
            "auth": dict(session.get("auth") or {}),
            # Compaction knobs so a resumed session keeps the trigger that
            # matches its context window (informational; runtime state lives
            # in the session dict itself).
            "compaction": {
                "enabled": bool(session.get("compaction_enabled", True)),
                "trigger_tokens": session.get("compaction_trigger_tokens"),
                "target_tokens": session.get("compaction_target_tokens"),
            },
            # Token-awareness knobs so a resumed session keeps identical
            # countdown semantics (informational; runtime state lives in
            # the session dict itself).
            "token_awareness": {
                "enabled": bool(session.get("token_awareness_enabled")),
                "budget_tokens": session.get("token_awareness_budget_tokens"),
                "reminder_tokens": session.get("token_awareness_reminder_tokens"),
                "update_every": session.get("token_awareness_update_every"),
            },
        },
        "messages": annotated,
    }
    with path.open("w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        if session.get("_durable_capture"):
            f.flush()
            os.fsync(f.fileno())
    if session.get("_durable_capture"):
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


_MAX_ARGS_PREVIEW = 200


def _compact_tool_args(raw: Any) -> str:
    """Return a short, single-line preview of tool-call arguments."""
    text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
    text = " ".join(text.split())
    if len(text) > _MAX_ARGS_PREVIEW:
        text = text[:_MAX_ARGS_PREVIEW] + "…"
    return text


def _summarise_tool_outcome(content: Any) -> str:
    """Classify a tool result as 'ok' or 'error' without echoing its payload.

    Heuristic only (the raw content string carries no structured status
    field); used purely to give the flattened history a compact outcome
    marker, never to reproduce believable result payloads.
    """
    text = str(content).strip()
    if not text:
        return "ok"
    low = text.lower()
    if low.startswith(("error", "traceback")) or "exception" in low:
        return "error"
    return "ok"


def _repair_tool_call_pairing(
    msgs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Guarantee every ``tool_call`` has a ``tool`` message and vice versa.

    Walks the transcript; after each assistant message carrying
    ``tool_calls`` ensures the following messages include one ``tool``
    response per call id, inserting a placeholder result for any id that
    never got one (crash between the API reply and tool execution).
    ``tool`` messages whose id matches no pending call are dropped.
    """
    repaired: list[dict[str, Any]] = []
    outstanding: list[str] = []
    outstanding_names: dict[str, str] = {}
    for m in msgs:
        role = m.get("role")
        if role == "assistant":
            # A new assistant message closes any prior tool-call block;
            # ids never answered must be backfilled before it.
            for cid in outstanding:
                repaired.append({
                    "role": "tool",
                    "tool_call_id": cid,
                    "content": ("[tool result lost: the session was "
                                "interrupted before this tool call "
                                "completed; verify state before retrying]"),
                    "ts": m.get("ts"),
                })
            outstanding = []
            outstanding_names = {}
            for tc in m.get("tool_calls") or []:
                cid = tc.get("id")
                if cid:
                    outstanding.append(str(cid))
                    fn = tc.get("function") or {}
                    custom = tc.get("custom") or {}
                    outstanding_names[str(cid)] = (
                        custom.get("name") or fn.get("name") or "?")
            repaired.append(m)
        elif role == "tool":
            cid = str(m.get("tool_call_id") or "")
            if cid in outstanding:
                outstanding.remove(cid)
                repaired.append(m)
            # else: dangling tool message (its assistant call was lost or
            # already answered) — drop it.
        else:
            # user/system message: backfill first so the tool block stays
            # contiguous.
            for cid in outstanding:
                repaired.append({
                    "role": "tool",
                    "tool_call_id": cid,
                    "content": (f"[tool result lost: {outstanding_names.get(cid, '?')} "
                                "was interrupted before completing; verify "
                                "state before retrying]"),
                    "ts": m.get("ts"),
                })
            outstanding = []
            outstanding_names = {}
            repaired.append(m)
    for cid in outstanding:
        repaired.append({
            "role": "tool",
            "tool_call_id": cid,
            "content": (f"[tool result lost: {outstanding_names.get(cid, '?')} "
                        "was interrupted before completing; verify state "
                        "before retrying]"),
            "ts": None,
        })
    return repaired


def _normalise_for_resume(
    msgs: list[dict[str, Any]], *, flatten: bool = False,
) -> list[dict[str, Any]]:
    """Normalise resumed messages into an API-safe user/assistant transcript.

    Always merges consecutive same-role user messages so the API's strict
    user/assistant alternation is preserved.

    ``flatten`` controls whether structured ``tool_calls`` / ``tool``
    results are converted into plain assistant text.  Most providers accept
    tool-call IDs minted in a previous API session and should keep the real
    structured history — passing real tool calls back is what disambiguates
    "the tool ran" from "the model is narrating a call" in the model's own
    context.  Set ``flatten=True`` only for providers that reject stale
    tool-call IDs on resume (400 "Upstream request failed" — seen on
    opencode.ai / deepseek-v4-flash-free), via the agent spec's
    ``behaviour.resume_rejects_stale_tool_call_ids``.

    When flattening, results are rendered as a compact ``name(args) ->
    outcome`` line with no bracket/callable syntax and no raw result
    payload: a transcript that shows the model writing out full
    ``[Tool result: {...}]`` JSON teaches it that producing tool-result
    payloads is its own job, and it will start fabricating them (see
    issue #25).

    Stale token-awareness warnings (``<system_warning>Token usage: …``)
    are stripped from every message except the one carrying the most
    recent reading: mid-history readings are superseded by the latest
    one, and re-injecting them on every resume would fabricate
    progressively wrong counters.
    """
    # Keep only the latest token-awareness reading across the transcript.
    last_ta_idx = -1
    for i, m in enumerate(msgs):
        if _TA_WARNING_RE.search(m.get("content") or ""):
            last_ta_idx = i
    if last_ta_idx >= 0:
        msgs = [
            (dict(m, content=_TA_WARNING_RE.sub("", m["content"]).rstrip())
             if i != last_ta_idx and isinstance(m.get("content"), str)
             else m)
            for i, m in enumerate(msgs)
        ]
    # Normalise: merge consecutive user messages so the API's strict
    # user/assistant alternation is preserved.
    normalised: list[dict[str, Any]] = []
    for m in msgs:
        if normalised and m.get("role") == "user" and normalised[-1].get("role") == "user":
            old = normalised[-1].get("content", "")
            new = m.get("content", "")
            normalised[-1]["content"] = f"{old}\n\n{new}" if old else new
            normalised[-1]["ts"] = m.get("ts")
        else:
            normalised.append(dict(m))
    if flatten:
        normalised = _flatten_tool_calls(normalised)
    # Structural repair: strict providers (DeepSeek, OpenAI) reject any
    # assistant ``tool_calls`` not followed by a ``tool`` message per call
    # id, and any ``tool`` message without a preceding matching call.  A
    # crash between the assistant turn and the tool results (or a torn
    # journal) produces exactly those orphans — synthesize placeholder
    # results and drop dangling tool messages so resume is always API-safe.
    normalised = _repair_tool_call_pairing(normalised)
    return normalised


def _flatten_tool_calls(
    normalised: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    # Flatten tool call / result pairs into a single neutral summary line
    # per call, keyed on tool_call_id so the call and its outcome merge
    # even though they arrive as separate messages.
    pending: dict[str, tuple[str, str, Any]] = {}
    converted: list[dict[str, Any]] = []
    for m in normalised:
        if m.get("role") == "assistant" and "tool_calls" in m:
            for tc in m["tool_calls"]:
                custom = tc.get("custom")
                if custom:
                    name = custom.get("name", "?")
                    args = _compact_tool_args(custom.get("input", ""))
                else:
                    fn = tc.get("function") or {}
                    name = fn.get("name", "?")
                    args = _compact_tool_args(fn.get("arguments", ""))
                call_id = tc.get("id")
                if call_id:
                    pending[call_id] = (name, args, m.get("ts"))
                else:
                    converted.append({
                        "role": "assistant",
                        "content": f"prior tool use: {name}({args}) -> unknown",
                        "ts": m.get("ts"),
                    })
        elif m.get("role") == "tool":
            call_id = str(m.get("tool_call_id") or "")
            name, args, ts = pending.pop(call_id, ("?", "", m.get("ts")))
            outcome = _summarise_tool_outcome(m.get("content", ""))
            converted.append({
                "role": "assistant",
                "content": f"prior tool use: {name}({args}) -> {outcome}",
                "ts": m.get("ts") or ts,
            })
        else:
            converted.append(m)
    # Any call left without a matching result (shouldn't happen in a
    # well-formed snapshot, but don't silently drop it).
    for name, args, ts in pending.values():
        converted.append({
            "role": "assistant",
            "content": f"prior tool use: {name}({args}) -> unknown",
            "ts": ts,
        })
    # Merge consecutive assistant messages to preserve user/assistant
    # alternation required by the API.
    merged: list[dict[str, Any]] = []
    for m in converted:
        if merged and m.get("role") == "assistant" and merged[-1].get("role") == "assistant":
            old = merged[-1].get("content", "")
            new = m.get("content", "")
            merged[-1]["content"] = f"{old}\n\n{new}" if old else new
            merged[-1]["ts"] = m.get("ts")
        else:
            merged.append(dict(m))
    return merged


def _load_messages_snapshot(
    model: str, session_id: str, *, flatten: bool = False,
) -> list[dict[str, Any]] | None:
    path = _snapshot_path(model, session_id)
    if not path.exists():
        return None
    with path.open() as f:
        data = json.load(f)
    msgs = data["messages"] if isinstance(data, dict) and "messages" in data else data
    if not isinstance(msgs, list):
        return None
    return _normalise_for_resume(msgs, flatten=flatten)

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
                        data = json.load(f)
                        if isinstance(data, dict) and "messages" in data:
                            return data["messages"]
                        return data

    return None


def _find_snapshot_in_other_models(
    model: str, session_id: str, *, flatten: bool = False,
) -> tuple[list[dict[str, Any]] | None, str | None]:
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
        data = json.load(f)
    msgs = data["messages"] if isinstance(data, dict) and "messages" in data else data
    if not isinstance(msgs, list):
        return None, None
    return _normalise_for_resume(msgs, flatten=flatten), best.parent.name


def _load_snapshot_metadata(model: str, session_id: str) -> "dict[str, Any] | None":
    """Return the metadata describing the endpoint *model*'s session ran on.

    Two sources are combined:

    * the snapshot's ``metadata`` block in *model*'s own directory (legacy
      snapshots without the wrapper contribute only what they have);
    * the session logs (``<date>/<HHMMSS>_<id>.jsonl``): the endpoint of the
      **earliest** ``session_start`` record.  Snapshots are rewritten at
      every turn boundary, so a buggy resume that ran the transcript against
      the wrong provider overwrites the snapshot with *that* endpoint; the
      logs are append-only and keep the endpoint the session was actually
      created on.  The log wins whenever the two disagree.

    A snapshot found under a *different* model directory is deliberately not
    consulted: that is the cross-model resume path, where the caller has
    chosen a new provider (see :func:`_port_snapshot_to_model`).  Returns
    ``None`` when neither source knows anything about the session.
    """
    meta: "dict[str, Any] | None" = None
    path = _snapshot_path(model, session_id)
    if path.exists():
        try:
            with path.open() as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            data = None
        if isinstance(data, dict) and isinstance(data.get("metadata"), dict):
            meta = data["metadata"]

    origin = _session_origin_endpoint(model, session_id)
    if origin is not None:
        combined = dict(meta or {})
        combined["endpoint"] = origin
        return combined
    return meta


def _session_origin_endpoint(model: str, session_id: str) -> "str | None":
    """Return the endpoint the session was created on, from its logs.

    Scans every dated log file for *session_id* and returns the endpoint of
    the earliest ``session_start`` record (files sort chronologically:
    ``<YYYY-MM-DD>/<HHMMSS>_<id>.jsonl``).  ``None`` when no log survives.
    """
    model_dir = LOG_BASE / safe_model_name(model)
    if not model_dir.is_dir():
        return None
    starts: list[tuple[str, str]] = []
    for day in model_dir.iterdir():
        if not day.is_dir() or not day.name[:4].isdigit():
            continue
        for log in day.glob(f"*_{session_id}.jsonl"):
            try:
                with log.open() as f:
                    for line in f:
                        rec = json.loads(line)
                        if rec.get("type") == "session_start" and rec.get("endpoint"):
                            starts.append((f"{day.name}/{log.name}", rec["endpoint"]))
                        break  # session_start is always the first record
            except (OSError, json.JSONDecodeError):
                continue
    return min(starts)[1] if starts else None


def _bind_schema_to_resumed_session(
    schema: "dict[str, Any]", session_id: str,
) -> "dict[str, Any]":
    """Pin a schema to the endpoint and key source a session was run on.

    Resume must continue on the endpoint that produced the history: tool-call
    IDs, prefix-cache keys and model behaviour are provider-specific, and the
    key that funded the session lives with that provider — replaying a
    z.ai transcript against OpenRouter fails with a bogus 402 rather than a
    useful error.  The snapshot's recorded ``endpoint`` and ``auth``
    configuration therefore win over whatever the CLI default or wrapper
    resolved.  Returns a copy; the caller's schema is never mutated.

    No-op when the model has no session of its own for *session_id*: that is
    a deliberate provider switch (cross-model resume, handled by
    :func:`_port_snapshot_to_model` re-stamping the copied snapshot), where
    the caller's endpoint and key source are the whole point.
    """
    meta = _load_snapshot_metadata(schema.get("model", "unknown"), session_id)
    if not meta:
        return schema
    bound = dict(schema)
    changes: list[str] = []
    endpoint = str(meta.get("endpoint") or "")
    if endpoint and endpoint != (bound.get("endpoint") or ""):
        changes.append(f"endpoint {bound.get('endpoint') or '∅'!r} → {endpoint!r}")
        bound["endpoint"] = endpoint
    auth = meta.get("auth")
    if isinstance(auth, dict) and auth:
        # Key sources the resumed session never used are dropped, not merged:
        # resolution order (keyring → key_env → OPENROUTER_API_KEY) would
        # otherwise resurrect a key for the wrong provider.
        for key in ("auth", "keyring_service", "keyring_username", "key_env"):
            if key in auth:
                if bound.get(key) != auth[key]:
                    changes.append(f"{key} → {auth[key]!r}")
                bound[key] = auth[key]
            elif bound.get(key) is not None:
                changes.append(f"{key} dropped")
                del bound[key]
    if changes:
        print(f"{YEL}Resuming session {session_id} on its original "
              f"endpoint {endpoint!r} ({'; '.join(changes)}){RESET}")
    return bound


def _port_snapshot_to_model(
    schema: "dict[str, Any]", session_id: str,
) -> "Path | None":
    """Copy a session's snapshot into the new model's directory, re-stamped.

    Cross-model resume (``agentknit <new-model> --session <id>``) must not
    drag the old provider along: tool-call IDs, prefix-cache keys and the
    funding key are provider-specific.  Instead of replaying another
    provider's transcript in place, the snapshot file is **copied** under
    the new model with ``metadata.model`` / ``endpoint`` / ``auth``
    re-stamped from *schema* — the new session file records the new
    provider, so a later resume of it binds to that provider, not the old
    one.  The origin is kept in ``metadata.ported_from`` for traceability
    and the original file is left untouched.

    Idempotent: an existing snapshot for *session_id* under the new model
    wins and nothing is copied.  Returns the (new or pre-existing) snapshot
    path, or ``None`` when no snapshot exists under any other model.
    """
    model = schema.get("model", "unknown")
    dest = _snapshot_path(model, session_id)
    if dest.exists():
        return dest
    if not LOG_BASE.exists():
        return None
    matches = [
        p for p in LOG_BASE.glob(f"*/{session_id}_messages.json")
        if p.parent.name != safe_model_name(model) and p.is_file()
    ]
    if not matches:
        return None
    src = max(matches, key=lambda p: p.stat().st_mtime)
    try:
        with src.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not (isinstance(data, dict) and isinstance(data.get("messages"), list)):
        return None
    src_meta: "dict[str, Any]" = (
        data["metadata"] if isinstance(data.get("metadata"), dict) else {})
    auth = {k: schema[k] for k in
            ("auth", "keyring_service", "keyring_username", "key_env")
            if schema.get(k) is not None}
    data["metadata"] = {
        "model":        model,
        "endpoint":     schema.get("endpoint", ""),
        "session_id":   session_id,
        "ported_from":  {"model": src_meta.get("model"),
                         "endpoint": src_meta.get("endpoint")},
        **{k: v for k, v in src_meta.items()
           if k in ("default_tools", "tools", "agentknit_commit")},
        "auth":         auth,
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"{YEL}Ported session {session_id} from {src.parent.name} to "
          f"{safe_model_name(model)}; continuing on "
          f"{schema.get('endpoint') or 'the new endpoint'}{RESET}")
    return dest


# ── agent loop ────────────────────────────────────────────────────────────────

def _tool_call_history_item(tc: Any) -> "dict[str, Any]":
    """Serialize a tool call for the assistant history message.

    Function calls round-trip as ``{"type": "function", "function": {...}}``.
    Custom calls are re-emitted in their original shape
    (``{"type": "custom", "custom": {"name", "input"}}``) so grammar-tool
    conversations survive multi-turn replay.
    """
    if tc.type == "custom":
        raw = tc.custom_input if tc.custom_input is not None else tc.function.arguments
        return {"id": tc.id, "type": "custom",
                "custom": {"name": tc.function.name, "input": raw}}
    return {"id": tc.id, "type": "function",
            "function": {"name": tc.function.name,
                         "arguments": tc.function.arguments}}


def _expand_aliases(
    tools: list[dict[str, Any]],
    tool_dispatch: dict[str, Any],
    aliases: dict[str, str],
    *,
    advertise: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Expand alias → canonical mappings into the tool schema and dispatch table.

    For each ``alias_name: canonical_name`` pair:

    * If the alias already has its own ``tool_dispatch`` entry it is left alone.
    * Otherwise the canonical dispatch entry is copied under the alias name.
    * If ``advertise`` and the canonical tool appears in ``tools`` (structured
      schema) and the alias does not, a deep-copy of the canonical tool spec is
      appended under the alias name so the model is aware of it too.

    ``advertise=False`` makes the alias dispatch-only: the retired name still
    works if a model or a restored session emits it, but it is not offered to
    the model as a second tool.  Advertising a retired name alongside its
    successor would show the model two identical tools (see
    ``_LEGACY_TOOL_ALIASES``).

    Both ``tools`` and ``tool_dispatch`` are copied; originals are not mutated.
    """
    tools         = list(tools)
    tool_dispatch = dict(tool_dispatch)

    # Fast lookup: tool name → index in tools list
    schema_by_name: dict[str, dict[str, Any]] = {}
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
        if advertise and canonical_name in schema_by_name and alias_name not in schema_by_name:
            alias_tool = copy.deepcopy(schema_by_name[canonical_name])
            # Custom tools carry their name at the top level; function tools
            # nest it under "function".
            if alias_tool.get("type") == "custom":
                alias_tool["name"] = alias_name
            else:
                fn = alias_tool.get("function") or alias_tool
                fn["name"] = alias_name
            tools.append(alias_tool)
            schema_by_name[alias_name] = alias_tool

    return tools, tool_dispatch


_REQUIRED_SESSION_KEYS = frozenset({
    "messages", "tools", "structured", "tool_dispatch", "session_id",
    "cache_key", "model", "endpoint", "non_interactive", "usage_totals",
    "provider", "max_output_tokens", "strict_cache_proof", "llm_call_count",
    "on_event", "streaming", "options", "session_start_ts",
    "compaction_enabled", "compaction_trigger_tokens", "compaction_target_tokens",
    "compaction_keep_last_turns", "compaction_policy", "compaction_min_chars",
    "compaction_last_prompt_tokens", "min_cacheable_tokens",
})


def _validate_session_dict(session: "Session") -> None:
    """Check that *session* has all keys required for restoration.

    Raises ``ValueError`` if any required key is missing.
    """
    missing = _REQUIRED_SESSION_KEYS - set(session.keys())
    if missing:
        raise ValueError(
            f"Session dict is missing required keys: {', '.join(sorted(missing))}"
        )


class Session(TypedDict):
    """Typed view of the stateful dict returned by :func:`init_session`.

    A ``TypedDict`` (issue #3, lighter alternative to a dataclass): zero
    runtime change — the session stays a plain ``dict`` — while IDEs and
    mypy gain autocomplete and typo detection on every ``session["..."]``
    access, and this class is the single source of truth for which fields
    exist and what they hold.

    Required keys mirror :data:`_REQUIRED_SESSION_KEYS`; runtime state that
    only exists mid-session (underscore-prefixed) and optional auth metadata
    are ``NotRequired``.
    """
    # conversation & tools
    messages: list[dict[str, Any]]             # role/content dicts; [0] is the system prompt
    tools: list[dict[str, Any]]                # model-facing tool specs (OpenAI format)
    tool_dispatch: dict[str, Any]              # tool name → {python_function, param_map}
    structured: bool                 # structured tool calls (vs inline text)
    # identity
    session_id: str
    cache_key: str                   # prefix-cache key sent as user/prompt_cache_key
    model: str
    endpoint: str                    # base URL or run:// URI ("" if unknown)
    non_interactive: bool
    # accounting
    usage_totals: dict[str, int]               # {prompt, completion, total, cached, cache_write}
    llm_call_count: int
    session_start_ts: str            # ISO timestamp
    # provider & limits
    provider: dict[str, Any] | None            # OpenRouter routing hints
    max_output_tokens: int | None
    strict_cache_proof: bool
    min_cacheable_tokens: int
    streaming: bool
    options: list[str]               # extra request options passed verbatim
    # events
    on_event: EventCallback
    # compaction
    compaction_enabled: bool
    compaction_trigger_tokens: int
    compaction_target_tokens: int
    compaction_keep_last_turns: int
    compaction_policy: "str | Callable[..., bool]"
    compaction_min_chars: int
    compaction_last_prompt_tokens: int
    # hooks (Claude Code / Codex-compatible lifecycle hooks)
    # NotRequired: sessions snapshotted before hooks existed lack these keys;
    # the restore path backfills defaults (empty list, enabled).
    hooks: NotRequired[list[HookEntry]]
    hooks_enabled: NotRequired[bool]
    # token awareness (model-facing countdown)
    # NotRequired: sessions snapshotted before token awareness existed lack
    # these keys; the restore path backfills defaults (enabled, counting
    # down from the compaction trigger).  New sessions always set all five.
    token_awareness_enabled: NotRequired[bool]          # master switch (default on)
    token_awareness_budget_tokens: NotRequired[int]     # countdown denominator
    token_awareness_reminder_tokens: NotRequired[int]   # checkpoint-protocol threshold
    token_awareness_update_every: NotRequired[int]      # inject every Nth LLM call
    token_awareness_last_remaining: NotRequired[int | None]  # edge-trigger state; None = no observation yet
    # durability
    # NotRequired: sessions saved before the durable-recovery feature lack
    # this key; the restore path defaults it to True.
    durable: NotRequired[bool]
    session_dir: NotRequired[Path]
    durable_sink: NotRequired[DurableSink | None]
    # NotRequired: absent (None) means the tool library's built-in default TTL.
    tool_ttl_seconds: NotRequired[int | None]  # TTL budget (s) for one sync tool exec
    # ── runtime-only state (set after construction) ──────────────────
    log_path: NotRequired[Path]      # JSONL transcript path (always set in practice)
    auth: NotRequired[dict[str, Any]]          # auth *configuration* (never the key itself)
    tool_executor: NotRequired["ToolExecutor | None"]
    _journal: NotRequired["SessionJournal | None"]
    _event_handlers: NotRequired[dict[str, list[EventCallback]]]
    _content_was_streamed: NotRequired[bool]
    _durable_capture: NotRequired[bool]
    # hooks runtime state: turn id (prompt_id), stop_hook_active guard,
    # pending additionalContext, and results of async hooks.
    _hook_state: NotRequired[dict[str, Any]]
    _cwd: NotRequired[Path]
    _cache_cold_warned: NotRequired[bool]
    # "ok" once a cache read/write has been observed, "missing" when a
    # post-first-call response exposed no cache proof.  A UI can surface
    # "missing" as a temporary status-bar warning and clear it on "ok".
    _cache_status: NotRequired[str]
    _continue_requested: NotRequired[bool]
    # runtime tool management (/tool remove → parked, /tool activate → restore)
    _removed_tools: NotRequired[dict[str, dict[str, Any]]]
    _removed_dispatch: NotRequired[dict[str, dict[str, Any]]]


def init_session(schema: "dict[str, Any]", non_interactive: bool = False,
                 resumed_from: str | None = None,
                 session: "Session | None" = None,
                 system_prompt_supplement: str = "",
                 cache_key: str | None = None,
                 max_output_tokens: int | None = None,
                 strict_cache_proof: bool = True,
                 on_event: "EventCallback | None" = None,
                 tool_executor: "ToolExecutor | None" = None,
                 *,
                 compaction_enabled: bool | None = None,
                 compaction_trigger_tokens: int | None = None,
                 compaction_target_tokens: int | None = None,
                 compaction_keep_last_turns: int | None = None,
                 compaction_policy: "str | Callable[..., bool] | None" = None,
                 compaction_min_chars: int | None = None,
                 min_cacheable_tokens: int | None = None,
                 tool_ttl_seconds: int | None = None,
                 token_awareness_enabled: bool | None = None,
                 token_awareness_budget_tokens: int | None = None,
                 token_awareness_reminder_tokens: int | None = None,
                 token_awareness_update_every: int | None = None,
                 durable: bool | None = None,
                 session_dir: str | Path | None = None,
                 durable_sink: DurableSink | None = None,
                 hooks: "str | Path | dict[str, Any] | list[Any] | None" = None,
                 hooks_enabled: bool | None = None,
                 ) -> "Session":
    """Build a stateful session dict (:class:`Session`).

    The cache_key is sent on every call as both `user` and `prompt_cache_key`
    so OpenRouter / the underlying provider can route this session's growing
    prefix to the same cache shard — much faster and cheaper after turn 1.

    By default the cache_key is the session_id, but a caller can pass a stable
    `cache_key` (e.g. derived from the working directory) to keep reusing a
    provider's prefix cache *without* resuming the prior conversation: that
    requires `resumed_from`, which is the only thing that loads past messages.

    When *session* is provided (a dict previously returned by this function),
    the function restores that session's state — messages, usage totals,
    tool dispatch, event subscriptions, etc. — into a freshly initialized
    session.  A new log file is opened and compaction state is reset.
    Keyword arguments act as overrides on the restored session.

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
    * ``compaction_keep_last_turns`` — number of recent raw non-system
      *messages* to keep after compaction (the boundary is snapped back to a
      user message so tool-call pairs are never split).  Defaults to the
      schema's ``compaction_keep_last_turns`` or ``2``.
    * ``compaction_policy`` — when automatic compaction runs:
      ``"threshold"`` (default, token threshold with hysteresis),
      ``"every_turn"`` (unconditional at each turn end, threshold as mid-turn
      backstop), ``"never"``, or a callable
      ``policy(session, usage, phase) -> bool`` with *phase* one of
      ``"mid_turn"`` / ``"turn_end"``.  Defaults to the schema's
      ``compaction_policy`` or ``"threshold"``.
    * ``compaction_min_chars`` — skip compaction while the compactable text
      is shorter than this many characters (guards against summaries that
      are longer than what they replace).  Defaults to the schema's
      ``compaction_min_chars`` or ``0`` (no minimum).

    ``min_cacheable_tokens`` — the provider's minimum cacheable prompt size,
    in prompt tokens. Below this size the provider caches nothing by design
    (e.g. Anthropic Claude Haiku ~4096, GPT-5.6-class models ~1024), so
    strict cache-proof mode treats a zero-cache response as expected rather
    than a failure when the prompt is below this floor. Defaults to the
    schema's ``min_cacheable_tokens`` or ``0`` (no minimum — any zero-cache
    call after the first is treated as a genuine miss).

    ``tool_ttl_seconds`` — time-to-live budget (seconds) for one synchronous
    tool execution.  The synchronous shell tool (``exec_shell``) is capped a
    little below this (TTL minus a safety margin) so a timed-out command
    still leaves room to stream partial output back.  Defaults to the
    schema's ``tool_ttl_seconds``; when neither is set, a built-in default
    applies (see ``DEFAULT_TOOL_TTL_S`` in :mod:`agentknit.tool_library`).

    ``durable`` — enable durable recovery (default ``True``).  Every message
    append, tool call and tool result inside a turn is written to an
    append-only, fsync-per-record journal
    (``<session_id>_journal.jsonl``) *as it happens*, so a crashed session
    can be recovered to the exact point of failure: completed tool results
    are re-injected instead of re-run, and in-flight tool calls whose
    outcome is unknown are flagged for verification.  Set to ``False`` to
    fall back to turn-boundary snapshots only.

    ``session_dir`` — explicit directory for this session's journal,
    snapshots, and logs.  It is independent of the model name and working
    directory and is also used when resuming the session.

    ``durable_sink`` — synchronous ``append(record)`` destination for the
    ordered lifecycle stream.  It is called only after the built-in journal
    commits, and before any event handler, model request, or tool dispatch
    consumes that record.  Exceptions stop the operation that produced it.

    ``hooks`` — Claude Code / Codex-compatible lifecycle hooks.  Accepts a
    path to a ``hooks.json``-shaped file, an inline config dict, or a list
    of either; layers merge additively with the user-level
    (``~/.agentknit/hooks.json``) and project-level
    (``<git-root>/.agentknit/hooks.json``) files discovered automatically.
    Python hooks can be added programmatically via
    :func:`agentknit.hooks.register_hook` (strictly equivalent to a command
    hook — both go through the same normalization).

    ``hooks_enabled`` — master switch (default ``True``).  ``False``
    disables all hooks for the session.
    """
    # A resumed session must run (and re-save its snapshot) on the endpoint
    # it was created on — bind here so every caller is covered, including
    # ones that build their own client and call init_session directly.
    # A cross-model resume ports the snapshot first: the copy under the new
    # model makes the provider switch explicit and keeps the original file
    # as the record of where the history came from.
    if resumed_from and session_dir is None:
        _port_snapshot_to_model(schema, resumed_from)
        schema = _bind_schema_to_resumed_session(schema, resumed_from)
    schema = _normalize_schema(schema)
    tools         = schema.get("inferred_tool_schema") or []
    behaviour     = schema.get("behaviour") or {}
    tool_dispatch = schema.get("tool_dispatch") or {}
    structured    = behaviour.get("call_delivery_mode", "structured_tool_calls") == "structured_tool_calls"
    model         = schema.get("model", "unknown")

    # Expand aliases before any filtering so aliased tools are treated like
    # first-class tools everywhere (non-interactive filtering, inline prompt, …).
    #
    # Legacy names (e.g. "execute_shell_command") are expanded dispatch-only:
    # a restored session or a model that still emits the retired name keeps
    # working, but the model is never shown both it and its successor.  A spec
    # that names the retired tool itself already gets a dispatch entry from
    # _normalize_schema, so nothing needs advertising here.  Aliases declared
    # in the spec are the caller's explicit choice and are advertised.
    declared_aliases = dict(schema.get("aliases") or {})
    legacy_aliases = {a: c for a, c in _LEGACY_TOOL_ALIASES.items() if a not in declared_aliases}
    # Only expand legacy aliases when their canonical tool is actually part of
    # this session — a custom minimal toolset (e.g. a single renamed "shell"
    # tool) must not inherit retired-name plumbing for tools it never had.
    legacy_aliases = {a: c for a, c in legacy_aliases.items() if c in tool_dispatch}
    if legacy_aliases:
        tools, tool_dispatch = _expand_aliases(tools, tool_dispatch, legacy_aliases, advertise=False)
    if declared_aliases:
        tools, tool_dispatch = _expand_aliases(tools, tool_dispatch, declared_aliases)

    if non_interactive:
        # Remove tools whose dispatch entry maps to t_ask_user.
        ask_tool_names = {
            tn for tn, e in tool_dispatch.items()
            if e.get("python_function") in _ASK_USER_FNS
        }
        tools = [t for t in tools
                 if ((t.get("function") or t).get("name")) not in ask_tool_names]

    # ── Restore from an existing session dict ──────────────────────────
    if session is not None:
        _validate_session_dict(session)
        # Messages, usage totals, call count, session id, cache key, etc.
        # are preserved from the saved session.
        restored: Session = dict(session)  # type: ignore[assignment]  # shallow copy – we override several keys below
        if session_dir is not None:
            restored["session_dir"] = Path(session_dir)
        if durable_sink is not None:
            restored["durable_sink"] = durable_sink
        restored["_durable_capture"] = bool(restored.get("session_dir") or restored.get("durable_sink"))
        restored["log_path"] = _open_log(model, restored.get("session_id") or uuid.uuid4().hex[:12],
                                           restored.get("session_dir"))
        # Reset compaction state so the new session starts fresh.
        restored["compaction_last_prompt_tokens"] = 0
        # Old snapshots predate token awareness: backfill defaults so the
        # restored session stays valid (enabled, counting down from the
        # compaction trigger — the window actually experienced).
        restored.setdefault("token_awareness_enabled", True)
        restored.setdefault("token_awareness_budget_tokens",
                            restored.get("compaction_trigger_tokens",
                                         DEFAULT_COMPACTION_TRIGGER_TOKENS))
        restored.setdefault("token_awareness_reminder_tokens",
                            DEFAULT_TOKEN_AWARENESS_REMINDER_TOKENS)
        restored.setdefault("token_awareness_update_every", 1)
        restored.setdefault("token_awareness_last_remaining", None)
        # Replace event handler if a new one was provided.
        if on_event is not None:
            restored["on_event"] = on_event
        # Apply keyword overrides that the caller explicitly passed.
        if cache_key is not None:
            restored["cache_key"] = cache_key
        if max_output_tokens is not None:
            restored["max_output_tokens"] = max_output_tokens
        if tool_executor is not None:
            restored["tool_executor"] = tool_executor
        # Compaction overrides.
        if compaction_enabled is not None:
            restored["compaction_enabled"] = compaction_enabled
        if compaction_trigger_tokens is not None:
            restored["compaction_trigger_tokens"] = compaction_trigger_tokens
        if compaction_target_tokens is not None:
            restored["compaction_target_tokens"] = compaction_target_tokens
        if compaction_keep_last_turns is not None:
            restored["compaction_keep_last_turns"] = compaction_keep_last_turns
        if compaction_policy is not None:
            restored["compaction_policy"] = compaction_policy
        if compaction_min_chars is not None:
            restored["compaction_min_chars"] = compaction_min_chars
        if min_cacheable_tokens is not None:
            restored["min_cacheable_tokens"] = min_cacheable_tokens
        if token_awareness_enabled is not None:
            restored["token_awareness_enabled"] = token_awareness_enabled
        if token_awareness_budget_tokens is not None:
            restored["token_awareness_budget_tokens"] = token_awareness_budget_tokens
        if token_awareness_reminder_tokens is not None:
            restored["token_awareness_reminder_tokens"] = token_awareness_reminder_tokens
        if token_awareness_update_every is not None:
            restored["token_awareness_update_every"] = token_awareness_update_every
        if durable is not None:
            restored["durable"] = durable
        # Hooks: restored sessions keep their configured hooks unless the
        # caller passes an explicit override; load-from-source happens in
        # _collect_hook_layers for fresh sessions, so a restore keeps the
        # entries that were already parsed into the session dict.
        if hooks_enabled is not None:
            restored["hooks_enabled"] = hooks_enabled
        if hooks is not None:
            _load_hooks(cast("dict[str, Any]", restored), hooks)
        restored.setdefault("hooks_enabled", True)
        # Reopen (or start) the session journal on restore.
        restored["_journal"] = (
            SessionJournal(_journal_path(restored.get("model") or "unknown",
                                         restored.get("session_id") or "",
                                         restored.get("session_dir")),
                           exclusive=bool(restored.get("session_dir")))
            if (restored.get("durable", True) or restored.get("session_dir")) else None
        )
        # Log the restoration event.
        _log(restored, {"type": "session_restored",
                        "session_id": restored.get("session_id"),
                        "ts": datetime.datetime.now().isoformat(timespec="seconds")})
        _emit(restored, "session_restored",
              session_id=restored.get("session_id"),
              fmt=f"{DIM}Restored session {restored.get('session_id')} "
                  f"({len(restored.get('messages', []))} messages){RESET}")
        return restored

    # ── Build a brand-new session ──────────────────────────────────────
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

    # Environment awareness: user identity, git status, cwd, OS, date, scratchpad.
    sys_msg += "\n\n" + environment_context(model, schema.get("version"))

    # Token awareness: resolve the four knobs (explicit kwarg → schema →
    # default) before building the system prompt, which declares the budget.
    ta_enabled = (
        token_awareness_enabled if token_awareness_enabled is not None
        else schema.get("token_awareness_enabled", True)
    )
    ta_budget = (
        token_awareness_budget_tokens if token_awareness_budget_tokens is not None
        else schema.get("token_awareness_budget_tokens")
        or schema.get("context_window")
        or (compaction_trigger_tokens if compaction_trigger_tokens is not None
            else schema.get("compaction_trigger_tokens", DEFAULT_COMPACTION_TRIGGER_TOKENS))
    )
    ta_reminder = (
        token_awareness_reminder_tokens if token_awareness_reminder_tokens is not None
        else schema.get("token_awareness_reminder_tokens",
                        DEFAULT_TOKEN_AWARENESS_REMINDER_TOKENS)
    )
    ta_update_every = max(1, int(
        token_awareness_update_every if token_awareness_update_every is not None
        else schema.get("token_awareness_update_every", 1)
    ))
    if ta_enabled:
        sys_msg += "\n\n" + _TOKEN_AWARENESS_SYSTEM.format(budget=ta_budget)

    session_id = resumed_from if resumed_from else uuid.uuid4().hex[:12]
    streaming = bool(
        (schema.get("provider_api_support") or {})
        .get("streaming", {})
        .get("supported", False)
    )
    session_start_ts = datetime.datetime.now().isoformat(timespec="seconds")
    # ── hooks: discover config layers, merge additively ───────────────────
    from .hooks import parse_hooks_config as _parse_hooks_config
    behaviour_hooks = behaviour.get("hooks")
    hook_sources: list[Any] = []
    # Explicit kwarg first, then the spec's behaviour hooks, then the
    # project layer (<git-root>/.agentknit/hooks.json) and the user layer.
    if hooks is not None:
        hook_sources.append(hooks)
    if behaviour_hooks:
        hook_sources.append(behaviour_hooks)
    hook_sources.append(_git_root() / ".agentknit" / "hooks.json")
    hook_sources.append(Path.home() / ".agentknit" / "hooks.json")
    session_hooks: list[HookEntry] = []
    for source in hook_sources:
        entries, warnings = _parse_hooks_config(source)
        session_hooks.extend(entries)
        for w in warnings:
            # Missing files are normal (layers are optional); anything else
            # is a real config problem worth surfacing at startup.
            if "not found" not in w:
                print(f"{YEL}⚠ hooks: {w}{RESET}", file=sys.stderr)
    session = cast(Session, {
        "messages":        [{"role": "system", "content": sys_msg,
                             "ts": session_start_ts}],
        "tools":           tools,
        "structured":      structured,
        "tool_dispatch":   tool_dispatch,
        "tool_executor":   tool_executor,
        "session_id":      session_id,
        "cache_key":       cache_key or session_id,
        "model":           model,
        "endpoint":        schema.get("endpoint", ""),
        # Auth *configuration* (never the resolved key itself) so a snapshot
        # can be replayed without the wrapper that injected it.  One key
        # source only: keyring (keyring_service+keyring_username) takes
        # precedence over key_env, mirroring _get_key_for_schema's resolution
        # order.  "auth" (the scheme, e.g. "opencode-github-copilot") is
        # orthogonal and always recorded when present.
        "auth":            ({k: schema[k] for k in
                            ("auth", "keyring_service", "keyring_username")
                            if schema.get(k) is not None}
                           if schema.get("keyring_service") and schema.get("keyring_username")
                           else {k: schema[k] for k in ("auth", "key_env")
                                 if schema.get(k) is not None}),
        "session_dir":     Path(session_dir) if session_dir is not None else None,
        "log_path":        _open_log(model, session_id, session_dir),
        "non_interactive": non_interactive,
        "usage_totals":    {"prompt": 0, "completion": 0, "total": 0,
                            "cached": 0, "cache_write": 0},
        "provider":        schema.get("provider"),
        "max_output_tokens": max_output_tokens or schema.get("max_output_tokens"),
        "strict_cache_proof": strict_cache_proof,
        "llm_call_count":  0,
        "_cache_status":   "ok",
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
        "compaction_policy": (
            compaction_policy if compaction_policy is not None
            else schema.get("compaction_policy", "threshold")
        ),
        "compaction_min_chars": (
            compaction_min_chars if compaction_min_chars is not None
            else schema.get("compaction_min_chars", 0)
        ),
        "compaction_last_prompt_tokens": 0,
        "min_cacheable_tokens": (
            min_cacheable_tokens if min_cacheable_tokens is not None
            else schema.get("min_cacheable_tokens", DEFAULT_MIN_CACHEABLE_TOKENS)
        ),
        "tool_ttl_seconds": (
            tool_ttl_seconds if tool_ttl_seconds is not None
            else schema.get("tool_ttl_seconds")
        ),
        "token_awareness_enabled": bool(ta_enabled),
        "token_awareness_budget_tokens": int(ta_budget),
        "token_awareness_reminder_tokens": int(ta_reminder),
        "token_awareness_update_every": ta_update_every,
        "token_awareness_last_remaining": None,
        # Durable recovery: append-only WAL of every in-turn state change.
        "durable": (
            True if session_dir is not None else
            (schema.get("durable", True) if durable is None else durable)
        ),
        "_journal": SessionJournal(
            _journal_path(model, session_id, session_dir),
            exclusive=session_dir is not None) if (
                session_dir is not None or
                (durable if durable is not None else schema.get("durable", True))
            ) else None,
        "durable_sink": durable_sink,
        "_durable_capture": session_dir is not None or durable_sink is not None,
        # ── hooks ────────────────────────────────────────────────────────
        "hooks": session_hooks,
        "hooks_enabled": (
            hooks_enabled if hooks_enabled is not None
            else schema.get("hooks_enabled", True)
        ),
        "_hook_state": {"turn_id": None, "stop_hook_active": False,
                        "pending_context": [], "async_results": []},
        "_cwd": Path.cwd(),
    })
    # The system prompt becomes durable before the session can send it or
    # report any startup event.
    _persist_record(session, {"type": "message", "msg": session["messages"][0]})
    _log(session, {"type": "session_start", "model": model,
                   "endpoint": schema.get("endpoint", ""),
                   "session_id": session_id,
                   "mode": behaviour.get("call_delivery_mode"),
                   "non_interactive": non_interactive,
                   "cwd": os.getcwd(),
                   "sandbox_policy": (tool_executor.policy.metadata()  # type: ignore[union-attr]
                                      if getattr(tool_executor, "policy", None) else None),
                   "ts": session_start_ts})
    # SessionStart hooks: context-only, cannot block.  additionalContext is
    # appended to the system prompt (developer context at conversation
    # start); the pending-queue path is for mid-conversation events.
    _ss = _fire_hooks(session, "SessionStart",
                      matcher_values=["resume" if resumed_from else "startup"],
                      source="resume" if resumed_from else "startup")
    _ss_ctx = _ss.additional_context
    _async_ctx = _drain_pending_hook_context(session)
    for _ctx in (_ss_ctx, _async_ctx):
        if _ctx:
            session["messages"][0]["content"] += "\n\n" + _ctx
    if resumed_from:
        flatten_resume = bool(behaviour.get("resume_rejects_stale_tool_call_ids"))
        if session_dir is not None:
            custom_snapshot = _snapshot_path(model, resumed_from, session_dir)
            if custom_snapshot.exists():
                with custom_snapshot.open() as f:
                    raw_snapshot = json.load(f)
                raw_messages = raw_snapshot.get("messages", raw_snapshot) if isinstance(raw_snapshot, dict) else raw_snapshot
                loaded = _normalise_for_resume(raw_messages, flatten=flatten_resume) if isinstance(raw_messages, list) else None
            else:
                loaded = None
        else:
            loaded = _load_messages_snapshot(model, resumed_from, flatten=flatten_resume)
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
            loaded_other, source_model = _find_snapshot_in_other_models(
                model, resumed_from, flatten=flatten_resume)
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
        # ── durable recovery ─────────────────────────────────────────────
        # The journal records state changes the turn-boundary snapshot cannot:
        # messages appended after the last snapshot and the outcome of every
        # tool call.  Prefer it whenever it knows at least as much.
        journal_state = replay_journal(_journal_path(model, resumed_from, session_dir))
        if journal_state.entries_replayed and (
            not session["messages"] or journal_state.mid_turn
            or len(journal_state.messages) > 1
        ):
            snap_count = len(session["messages"])
            # Reuse the snapshot normalisation (merge same-role messages,
            # optionally flatten stale tool-call IDs) so the recovered
            # history is API-safe, then layer the pending-tool recovery
            # note on top.
            session["messages"] = _normalise_for_resume(
                journal_state.messages, flatten=flatten_resume)
            _log(session, {"type": "journal_recovered", "resumed_from": resumed_from,
                           "entries_replayed": journal_state.entries_replayed,
                           "messages_loaded": len(journal_state.messages),
                           "pending_tool_calls": [
                               {"call_id": p.call_id, "name": p.name}
                               for p in journal_state.pending_tool_calls],
                           "mid_turn": journal_state.mid_turn,
                           "ts": datetime.datetime.now().isoformat(timespec="seconds")})
            _emit(session, "journal_recovered",
                  entries_replayed=journal_state.entries_replayed,
                  messages_loaded=len(journal_state.messages),
                  pending=len(journal_state.pending_tool_calls),
                  mid_turn=journal_state.mid_turn,
                  fmt=(f"{DIM}Recovered {len(journal_state.messages)} messages from the "
                       f"durable journal (snapshot had {snap_count}){RESET}"))
            if journal_state.pending_tool_calls:
                calls = "\n".join(
                    f"- {p.name}({json.dumps(p.args, ensure_ascii=False)})"
                    for p in journal_state.pending_tool_calls)
                session["messages"].append({
                    "role": "user",
                    "content": _PENDING_TOOL_NOTE.format(calls=calls),
                    "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                })
            if journal_state.unreceived_results:
                # Only results for calls still part of the resumed
                # conversation are relevant, and only as digests — full
                # payloads would blow the context window (a session with
                # hundreds of finished calls produced a >2.5 MB note that
                # survived compaction and exceeded DeepSeek's 1M limit).
                relevant = [
                    r for r in journal_state.unreceived_results
                    if r.call_id in {
                        str(tc.get("id"))
                        for m in session["messages"]
                        for tc in (m.get("tool_calls") or [])
                    }
                ]
                if relevant:
                    results = "\n".join(
                        f"- {r.name or r.call_id}: "
                        f"{r.summary(_UNRECEIVED_RESULT_MAX_CHARS)}"
                        for r in relevant)
                    if len(results) > _UNRECEIVED_RESULTS_MAX_CHARS:
                        results = (results[:_UNRECEIVED_RESULTS_MAX_CHARS]
                                   + "\n…[truncated]")
                    session["messages"].append({
                        "role": "user",
                        "content": (
                            "SYSTEM RECOVERY NOTE: These tool calls completed "
                            "just before the crash but their results were "
                            "never shown to you; treat these result digests "
                            "as observed (re-run the tool if you need the "
                            "full output):\n"
                            f"{results}"
                        ),
                        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                    })
    return session


def _error_metadata(
    exc: BaseException,
    session: Session,
    client: object | None = None,
    request_started_at: float | None = None,
) -> "dict[str, Any]":
    """Return stable, machine-readable fields for an error event/log record."""
    http_status = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if http_status is None and response is not None:
        http_status = getattr(response, "status_code", None)
    if http_status is None:
        # urllib.error.HTTPError exposes the status as ``code`` rather than
        # ``status_code``; accepting it also makes wrapped HTTP clients useful.
        http_status = getattr(exc, "code", None)
    if not isinstance(http_status, int):
        http_status = None

    is_subprocess = (
        isinstance(client, SubprocessOpenAI)
        or str(session.get("endpoint", "")).startswith("run://")
    )
    elapsed_s = None
    if request_started_at is not None:
        elapsed_s = round(max(0.0, time.monotonic() - request_started_at), 3)
    return {
        "error_class": type(exc).__name__,
        "http_status": http_status,
        "error_code": getattr(exc, "error_code", None),
        "error_message": getattr(exc, "error_message", None),
        "elapsed_s": elapsed_s,
        "adapter": "subprocess" if is_subprocess else "http",
    }


def _emit_and_log_error(
    session: Session,
    exc: BaseException,
    text: str,
    fmt: str,
    *,
    client: object | None = None,
    request_started_at: float | None = None,
    log_type: str = "error",
    error_kind: str | None = None,
) -> None:
    """Emit and persist an error with consistent structured diagnostics."""
    fields = _error_metadata(exc, session, client, request_started_at)
    _emit(session, "error", text=text, **fields, fmt=fmt)
    record = {"type": log_type, "error": text, **fields,
              "ts": datetime.datetime.now().isoformat(timespec="seconds")}
    if error_kind is not None:
        record["error_kind"] = error_kind
    _log(session, record)


def _complete(client: openai.OpenAI | SubprocessOpenAI, session: Session, **kwargs: Any) -> Any:
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
    kwargs["on_rate_limit_wait"] = _rate_limit_wait_callback(session)

    # Do not include callbacks (or any authentication transport state) in the
    # durable request.  This is the exact model request payload otherwise.
    request = {key: value for key, value in kwargs.items()
               if not key.startswith("on_")}
    _persist_record(session, {"type": "model_request", "request": request})

    def _on_raw_response(frame: Any) -> None:
        _persist_record(session, {"type": "model_response_frame", "frame": frame})

    def _on_request_attempt(attempt: Any) -> None:
        _persist_record(session, {"type": "model_request_attempt", "attempt": attempt})

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

        stream_callbacks: dict[str, Any] = {}
        if isinstance(client, (openai.OpenAI, SubprocessOpenAI)):
            stream_callbacks["on_raw_response"] = _on_raw_response
            stream_callbacks["on_request_attempt"] = _on_request_attempt
        resp = client.chat.completions.create(
            **kwargs,
            on_content_delta=_on_delta,
            on_reasoning_delta=_on_reasoning,
            **stream_callbacks,
        )
        if streamed:
            # Reasoning always precedes content in the SSE stream, so flush
            # the reasoning sequence first: event consumers that buffer
            # deltas (e.g. the TUI) wait for `reasoning_stream_end` to write
            # the trace, and would otherwise drop it whenever content also
            # streamed.  The empty `fmt` keeps the terminal renderer a no-op —
            # there, the first content delta already terminated the
            # `[thinking]` line with its own newline prefix.
            if reasoned:
                _emit(session, "reasoning_stream_end", no_newline=True, fmt="")
            _emit(session, "content_stream_end", no_newline=True, fmt="\n")
            session["_content_was_streamed"] = True
        elif reasoned:
            _emit(session, "reasoning_stream_end", no_newline=True, fmt="\n")
    else:
        response_callbacks: dict[str, Any] = {}
        if isinstance(client, (openai.OpenAI, SubprocessOpenAI)):
            response_callbacks["on_raw_response"] = _on_raw_response
            response_callbacks["on_request_attempt"] = _on_request_attempt
        resp = client.chat.completions.create(**kwargs, **response_callbacks)

    _persist_record(session, {"type": "model_response", "response": {
        "content": getattr(resp.choices[0].message, "content", None),
        "reasoning": getattr(resp, "reasoning", None),
        "tool_calls": [
            {"id": tc.id, "type": tc.type, "name": tc.function.name,
             "arguments": tc.function.arguments}
            for tc in (getattr(resp.choices[0].message, "tool_calls", None) or [])
        ],
    }})

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


def compact_session(
    client: openai.OpenAI | SubprocessOpenAI,
    model: str,
    session: Session,
    prompt_tokens: int | None = None,
) -> bool:
    """Replace the oldest portion of the conversation with a continuation-oriented summary.

    *prompt_tokens* is the server-reported prompt size that triggered the
    compaction (when known — threshold/every-turn policies); it is shown in
    the ``compaction`` event as ``triggered at N tokens (P% to max size)``
    and omitted when unknown (manual ``/compact``, overflow retry).

    Keeps the system prompt and the most recent *compaction_keep_last_turns*
    non-system messages in raw form.  The kept boundary is snapped back to a
    user message so an assistant ``tool_calls`` message is never separated
    from its ``tool`` results (which would produce an API-invalid sequence)
    and the summary is always followed by a user turn.  The compacted portion
    is summarized by the model and replaced with a single assistant message
    tagged with ``compacted_summary=true`` metadata.

    Compaction runs in two phases: first a *pre-compaction* prompt asks the
    model to record everything it needs to keep working (the reply is kept
    as an assistant message), then the summary phase summarizes the prefix
    *including* that recorded state.

    If ``compaction_min_chars`` is set on the session, compaction is skipped
    when the compactable text is shorter than that — summarizing a tiny
    history costs an LLM call and can *grow* the context.

    Returns ``True`` if the history was compacted, ``False`` if skipped.
    """
    messages = session["messages"]
    # ── PreCompact hooks ───────────────────────────────────────────────
    # A block (or continue:false) skips this compaction; the hysteresis
    # watermark is not updated, so the next growth cycle re-triggers.
    trigger_kind = ("manual" if prompt_tokens is None and
                    not session.get("compaction_last_prompt_tokens") else "auto")
    pre = _fire_hooks(session, "PreCompact", matcher_values=[trigger_kind],
                      trigger=trigger_kind)
    if pre.block or pre.stop:
        _emit(session, "compaction_skipped", reason="PreCompact hook",
              fmt=f"{DIM}[compaction] skipped by PreCompact hook{RESET}")
        return False
    keep = session.get("compaction_keep_last_turns", DEFAULT_COMPACTION_KEEP_LAST_TURNS)
    # Server-reported prompt size at the trigger moment, for the user-facing
    # message.  Passed by the caller when known; falls back to the session
    # watermark (0 = never triggered, so unknown) so manual /compact and
    # overflow-retry paths simply omit the number.
    prompt_tokens = (prompt_tokens if prompt_tokens is not None
                     else session.get("compaction_last_prompt_tokens", 0) or None)
    # Adaptively shrink the kept suffix when the provider rejects the summary
    # request as too large: with a mis-sized trigger the history can be so big
    # that even keep_last_turns=2 overflows the window.  The loop below drops
    # to keep=0 (only the system prompt + summary survive) if needed.
    # Phase 2 can legitimately reject an oversized summary request; the
    # retry loop in compact_session() then shrinks `keep`.
    while True:
        try:
            if _compact_once(client, model, session, messages, keep,
                             trigger_kind=trigger_kind):
                return True
        except ContextWindowExceededError:
            pass  # fall through to the keep-shrink below
        else:
            return False  # skipped, nothing compactable
        if keep <= 0:
            return False
        keep = 0 if keep <= 1 else keep // 2
        _emit(session, "compaction_retry", keep=keep,
              fmt=f"{DIM}{MAG}[compaction] request too large; retrying with "
                  f"keep_last_turns={keep}{RESET}")


def _compact_once(
    client: openai.OpenAI | SubprocessOpenAI,
    model: str,
    session: Session,
    messages: list[dict[str, Any]],
    keep: int,
    trigger_kind: str = "auto",
) -> bool:
    """One compaction attempt keeping the last *keep* raw messages."""

    # Find the boundary: keep system prompt + last `keep` non-system messages.
    non_system_indices = [i for i, m in enumerate(messages) if m.get("role") != "system"]
    if len(non_system_indices) <= keep:
        return False  # nothing to compact

    split_idx = non_system_indices[-keep] if keep > 0 else len(messages)
    if keep > 0 and messages[split_idx].get("role") != "user":
        # Snap to a user-message boundary.  Prefer snapping back (keeping
        # more); if that would empty the compactable prefix, snap forward
        # (keeping fewer) instead.
        back = split_idx
        while back > non_system_indices[0] and messages[back].get("role") != "user":
            back -= 1
        if messages[back].get("role") == "user" and back > non_system_indices[0]:
            split_idx = back
        else:
            fwd = split_idx
            while fwd < len(messages) and messages[fwd].get("role") != "user":
                fwd += 1
            if fwd == len(messages):
                return False  # no clean boundary to split at
            split_idx = fwd

    prefix = messages[:split_idx]
    suffix = messages[split_idx:]

    if not any(m.get("role") != "system" for m in prefix):
        return False  # boundary snapping left nothing to compact

    min_chars = session.get("compaction_min_chars", 0) or 0
    if min_chars:
        compactable_chars = sum(
            len(m.get("content") or "") for m in prefix if m.get("role") != "system"
        )
        if compactable_chars < min_chars:
            return False

    # ── Phase 1: pre-compaction preservation prompt ─────────────────────────
    # Give the model a chance to record important state as an assistant
    # message *before* the prefix is summarized away.  Failure here is
    # non-fatal — we still proceed to the summary phase.
    pre_compaction_messages = list(prefix)
    pre_compaction_messages.append({"role": "user", "content": _PRE_COMPACTION_PROMPT})
    request_started_at = time.monotonic()
    try:
        _persist_record(session, {"type": "model_request", "purpose": "pre_compaction",
                                  "request": {"model": model, "messages": pre_compaction_messages,
                                              "temperature": 0,
                                              "max_tokens": session.get(
                                                  "compaction_target_tokens",
                                                  DEFAULT_COMPACTION_TARGET_TOKENS)}})
        pre_resp = client.chat.completions.create(
            model=model,
            messages=pre_compaction_messages,
            temperature=0,
            max_tokens=session.get("compaction_target_tokens", DEFAULT_COMPACTION_TARGET_TOKENS),
            on_rate_limit_wait=_rate_limit_wait_callback(session),
        )
        _persist_record(session, {"type": "model_response", "purpose": "pre_compaction",
                                  "response": {"content": pre_resp.choices[0].message.content}})
        pre_content = (pre_resp.choices[0].message.content or "").strip()
    except Exception as exc:
        _emit(session, "compaction_preparation_skipped", reason=str(exc))
        pre_content = ""
    if pre_content:
        prefix = prefix + [
            {"role": "user", "content": _PRE_COMPACTION_PROMPT},
            {"role": "assistant", "content": pre_content},
        ]

    # ── Phase 2: summarize the (now augmented) prefix ───────────────────────
    compaction_messages = list(prefix)
    compaction_messages.append({"role": "user", "content": _COMPACTION_PROMPT})

    request_started_at = time.monotonic()
    try:
        _persist_record(session, {"type": "model_request", "purpose": "compaction",
                                  "request": {"model": model, "messages": compaction_messages,
                                              "temperature": 0,
                                              "max_tokens": session.get(
                                                  "compaction_target_tokens",
                                                  DEFAULT_COMPACTION_TARGET_TOKENS)}})
        resp = client.chat.completions.create(
            model=model,
            messages=compaction_messages,
            temperature=0,
            max_tokens=session.get("compaction_target_tokens", DEFAULT_COMPACTION_TARGET_TOKENS),
            on_rate_limit_wait=_rate_limit_wait_callback(session),
        )
    except RateLimitError as exc:
        err = f"Compaction rate limited: {exc}"
        _emit_and_log_error(session, exc, err, f"\n{RED}{err}{RESET}",
                            client=client, request_started_at=request_started_at,
                            log_type="compaction_error", error_kind="rate_limit")
        return False
    except Exception as exc:
        if _is_context_window_error(exc):
            # Summary request itself is too large: signal the caller to
            # retry with a smaller kept suffix.
            raise ContextWindowExceededError(str(exc)) from exc
        err = f"Compaction failed: {exc}"
        _emit_and_log_error(session, exc, err, f"\n{RED}{err}{RESET}",
                            client=client, request_started_at=request_started_at,
                            log_type="compaction_error")
        return False

    summary = (resp.choices[0].message.content or "").strip()
    _persist_record(session, {"type": "model_response", "purpose": "compaction",
                              "response": {"content": resp.choices[0].message.content}})
    if not summary:
        return False

    # Replace the compacted prefix with a single summary message.
    system_msgs = [m for m in prefix if m.get("role") == "system"]
    summary_msg = {
        "role": "assistant",
        "content": summary,
        "compacted_summary": True,
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    session["messages"] = system_msgs + [summary_msg] + suffix
    _write_journal_record(session, {"type": "reset_messages", "reason": "compaction",
                                    "messages": list(session["messages"])})

    compacted_turns = len(prefix) - len(system_msgs)
    compacted_chars = sum(
        len(m.get("content") or "") for m in prefix if m.get("role") != "system"
    )
    summary_chars = len(summary)
    ratio = compacted_chars / summary_chars if summary_chars else 0.0

    _emit(session, "compaction",
          summary=summary,
          compacted_turns=compacted_turns,
          fmt=(f"{DIM}{MAG}[compaction] {compacted_turns} turns summarized "
               f"({len(summary)} chars){RESET}  "
               f"{CYAN}{BOLD}ratio {ratio:.1f}x{RESET}"))
    _log(session, {"type": "compaction",
                   "compacted_turns": compacted_turns,
                   "summary_length": len(summary),
                   "compaction_ratio": round(ratio, 2),
                   "summary": summary,
                   "ts": datetime.datetime.now().isoformat(timespec="seconds")})
    _save_messages_snapshot(session)
    # ── PostCompact hooks (advisory + additionalContext) ────────────────
    _fire_hooks(session, "PostCompact", matcher_values=[trigger_kind],
                trigger=trigger_kind)
    return True


_CONTEXT_LIMIT_RE = re.compile(
    r"context.?length|context.?window|token.?limit|maximum.?context|"
    r"too.?many.?tokens|prompt.?is.?too.?long|exceeds?.?(the)?.?model|"
    r"request.?too.?large|input.?too.?long|reduce.?the.?length",
    re.IGNORECASE,
)


def _is_context_window_error(exc: BaseException) -> bool:
    """Return True when *exc* is a provider rejection for an over-limit prompt.

    Detects, in order: our own :class:`ContextWindowExceededError`, then any
    exception carrying a 400/413 status whose message mentions a context /
    token limit (covers the OpenAI SDK's ``BadRequestError`` and simple
    ``RuntimeError("... [HTTP 400] ...")`` wrappers raised by subprocess
    clients).  Rate limits (429) never match.
    """
    if isinstance(exc, ContextWindowExceededError):
        return True
    status = getattr(exc, "status_code", None)
    if status is None:
        # urllib.error.HTTPError exposes .status / .code, not .status_code.
        status = getattr(exc, "status", None) or getattr(exc, "code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    try:
        status = int(status) if status is not None else None
    except (TypeError, ValueError):
        status = None
    text = str(exc)
    if status is None:
        # Subprocess/simple clients embed the status in the message, e.g.
        # ``RuntimeError("... [HTTP 400] ...")`` or the OpenAI SDK's
        # ``"Error code: 413 - ..."`` — recover it from the text.
        m = re.search(r"\bHTTP\s*(\d{3})\b|\bError code:\s*(\d{3})\b", text,
                      re.IGNORECASE)
        if m:
            status = int(m.group(1) or m.group(2))
    if status not in (400, 413):
        return False
    if _CONTEXT_LIMIT_RE.search(text):
        return True
    # The plain-urllib transport (openai_compat.OpenAI) raises
    # ``HTTPError("400 Client Error: Bad Request for url: ...")`` whose
    # message carries no body.  Read the provider's error body from the
    # exception itself and match against that too.
    try:
        body = exc.read()  # type: ignore[attr-defined]
        if isinstance(body, bytes):
            body = body.decode("utf-8", "replace")
        if body and _CONTEXT_LIMIT_RE.search(str(body)):
            return True
    except Exception:
        pass
    return False


# Backward-compatible alias for the pre-public name.
_compact_session = compact_session


def _maybe_compact(
    client: openai.OpenAI | SubprocessOpenAI,
    model: str,
    session: Session,
    usage: object,
) -> None:
    """Trigger compaction if the session's prompt tokens exceed the threshold.

    Compaction only fires once per growth cycle.  After a compaction we
    record the pre-compaction prompt count in
    ``compaction_last_prompt_tokens``.  The next compaction is refused
    until the count has grown by at least 25 % of the trigger threshold
    above that recorded value.  This prevents the "compaction storm"
    where every subsequent turn re-triggers because the threshold is
    still exceeded or only slightly exceeded.
    """
    if not session.get("compaction_enabled", True):
        return
    prompt_tok = getattr(usage, "prompt_tokens", 0) or 0
    trigger = session.get("compaction_trigger_tokens", DEFAULT_COMPACTION_TRIGGER_TOKENS)
    last = session.get("compaction_last_prompt_tokens", 0)
    hysteresis = trigger // 4
    if prompt_tok >= trigger and prompt_tok > last + hysteresis:
        _compact_session(client, model, session)
        session["compaction_last_prompt_tokens"] = prompt_tok


def _apply_compaction_policy(
    client: openai.OpenAI | SubprocessOpenAI,
    model: str,
    session: Session,
    usage: object,
    phase: str,
) -> None:
    """Run the session's compaction policy at *phase*.

    *phase* is ``"mid_turn"`` (after each completion inside the tool loop) or
    ``"turn_end"`` (after a successful final answer).  Policies:

    * ``"threshold"`` (default) — token-threshold compaction with hysteresis
      mid-turn (the historical behaviour); nothing extra at turn end.
    * ``"every_turn"`` — unconditional compaction at turn end, with the
      threshold check kept mid-turn as an overflow backstop.
    * ``"never"`` — no automatic compaction.
    * a callable ``policy(session, usage, phase) -> bool`` — compaction runs
      whenever it returns true.  The callable bypasses ``compaction_enabled``.
    """
    policy = session.get("compaction_policy", "threshold")
    if callable(policy):
        if policy(session, usage, phase):
            compact_session(client, model, session)
        return
    if policy == "never":
        return
    if phase == "mid_turn":
        _maybe_compact(client, model, session, usage)
    elif policy == "every_turn":
        compact_session(client, model, session)


def _handle_tool_call(
    name: str,
    args: dict[str, Any],
    session: Session,
    *,
    call_id: str | None = None,
) -> str:
    """Dispatch one tool call, log it, print it, return the result string.

    ``call_id`` identifies the call in the durable journal; when omitted a
    fresh id is generated (used by inline mode, which has no API tool ids).
    """
    tool_dispatch   = session["tool_dispatch"]
    non_interactive = session["non_interactive"]
    call_id = call_id or new_call_id()

    entry = tool_dispatch.get(name) or {}
    is_ask = entry.get("python_function") in _ASK_USER_FNS

    # ── PreToolUse hooks ───────────────────────────────────────────────
    # Fired *before* the journal write-ahead so a rewritten updatedInput is
    # what gets journaled in tool_start — durable replay then shows the
    # args that actually ran, with no extra journal machinery.  A deny
    # (exit 2 / permissionDecision deny) writes tool_end with
    # outcome=hook_deny and returns the reason as the tool result.
    pre = _fire_hooks(session, "PreToolUse",
                      matcher_values=[canonical_tool_name(name), name],
                      **_hook_tool_payload_fields(session, name, args, call_id))
    _ask_hook_confirm = None
    if pre.permission_decision == "deny" or pre.block:
        reason = (pre.reason or pre.permission_decision_reason
                  or "tool call blocked by PreToolUse hook")
        result = f"ERROR: tool call denied by hook: {reason}"
        journal = session.get("_journal")
        if journal is not None:
            _write_journal_record(session, {"type": "tool_start", "call_id": call_id,
                                            "name": name, "args": args})
            _write_journal_record(session, {"type": "tool_end", "call_id": call_id,
                                            "name": name, "result": result,
                                            "outcome": "hook_deny"})
        _emit(session, "tool_call", name=name, args=args, fmt=fmt_call(name, args))
        _emit(session, "tool_result", name=name, result=result, streamed=False,
              files=None, diff_summary=None, fmt=fmt_result(result))
        _log(session, {"type": "tool_result", "name": name,
                       "python_function": getattr(entry.get("python_function"),
                                                  "__name__",
                                                  entry.get("python_function")),
                       "result": result, "hook": "PreToolUse:deny",
                       "ts": datetime.datetime.now().isoformat(timespec="seconds")})
        return result
    if pre.updated_input is not None:
        # Claude/Codex updatedInput replaces the whole input object.  Alias
        # translation maps Claude argument names onto agentknit's native
        # ones (file_path→path, old_string→old_str, …).
        try:
            args = translate_updated_input(pre.updated_input)
        except Exception as exc:  # non-blocking: proceed with original args
            _emit(session, "hook_warning", text=str(exc),
                  fmt=f"{YEL}⚠ hook updatedInput rejected: {exc}{RESET}")
        else:
            _log(session, {"type": "tool_call_rewritten", "name": name,
                           "args": args,
                           "ts": datetime.datetime.now().isoformat(timespec="seconds")})
    if pre.permission_decision == "ask" and not non_interactive:
        # Map "ask" onto agentknit's interactive surface: confirm with the
        # user before dispatch (the ask_user machinery pauses the input
        # collector when one is running).
        _ask_hook_confirm = pre.permission_decision_reason or (
            f"hook asks for confirmation before {name}")

    journal = session.get("_journal")
    if journal is not None:
        # Write-ahead: persisted before the tool runs, so a crash between
        # here and tool_end marks the side effects as unknown on recovery.
        _write_journal_record(session, {"type": "tool_start", "call_id": call_id,
                                        "name": name, "args": args})

    pf_name = getattr(entry.get("python_function"), "__name__",
                      entry.get("python_function"))
    _log(session, {"type": "tool_call", "name": name,
                   "python_function": pf_name, "args": args,
                   "ts": datetime.datetime.now().isoformat(timespec="seconds")})
    _emit(session, "tool_call", name=name, args=args, fmt=fmt_call(name, args))

    _tool_module._tool_context.session_id = session.get("session_id")
    _tool_module._tool_context.tool_dispatch = tool_dispatch
    _tool_module._tool_context.tool_ttl_seconds = session.get("tool_ttl_seconds")

    if is_ask and non_interactive:
        result = "ERROR: user interaction is disabled (--non-interactive)"
        log_data: dict[str, Any] = {"result": result}
        streamed = False
    else:
        if _ask_hook_confirm is not None:
            # PreToolUse "ask": confirm with the user before dispatch.
            collector = _tool_module._input_collector
            if collector is not None:
                collector.pause()
            try:
                answer = input(f"{YEL}? {_ask_hook_confirm}{RESET} "
                               f"[y/N] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = ""
                print()
            finally:
                if collector is not None:
                    collector.resume()
            if answer not in ("y", "yes"):
                result = ("ERROR: tool call denied by user "
                          "(PreToolUse hook asked)")
                log_data = {"result": result}
                streamed = False
                journal = session.get("_journal")
                if journal is not None:
                    _write_journal_record(session, {"type": "tool_end",
                                                    "call_id": call_id,
                                                    "name": name,
                                                    "result": result,
                                                    "outcome": "hook_ask_deny"})
                _emit(session, "tool_result", name=name, result=result,
                      streamed=False, files=None, diff_summary=None,
                      fmt=fmt_result(result))
                _log(session, {"type": "tool_result", "name": name,
                               "python_function": getattr(
                                   entry.get("python_function"), "__name__",
                                   entry.get("python_function")),
                               "result": result, "hook": "PreToolUse:ask-deny",
                               "ts": datetime.datetime.now().isoformat(
                                   timespec="seconds")})
                return result
        try:
            executor = session.get("tool_executor")
            if executor is None:
                result, log_data = dispatch(name, args, tool_dispatch)
            else:
                result, log_data = executor.execute(
                    name, args, entry, session={"session_id": session.get("session_id", "")})
            streamed = bool(log_data.pop("streamed", False))
        except FatalToolDispatchError as e:
            # Unknown tool name: the model's mistake, not a crash. Feed the
            # error back as the tool result so the model can retry with a
            # tool that actually exists; the loop must continue.
            # Dispatch-only legacy aliases are hidden from the list: the model
            # was never offered them, so recommending one back would be noise.
            # A pre-rename spec whose *only* shell tool is the retired name
            # (canonical absent) keeps it listed.
            callable_names = [
                t for t in tool_dispatch
                if not (t in _LEGACY_TOOL_ALIASES
                        and _LEGACY_TOOL_ALIASES[t] in tool_dispatch)
            ]
            available = ", ".join(sorted(repr(t) for t in callable_names)) or "(none)"
            result = (f"{e}. Available tools: {available}. "
                      f"Call one of the available tools instead.")
            log_data = {"result": result}
            streamed = False
        except Exception as exc:
            # The caller may choose how to surface the exception, but the
            # durable stream must record the observable failure first.
            # Include the exception type and innermost frames so the model
            # (and the human) can locate tool bugs from the result alone.
            tb = traceback.extract_tb(sys.exc_info()[2])
            inner = ", ".join(f"{os.path.basename(f.filename)}:{f.lineno} in {f.name}"
                              for f in tb[-3:])
            failure = (f"ERROR: tool {name!r} raised {type(exc).__name__}: {exc} "
                       f"({inner})")
            if journal is not None:
                _write_journal_record(session, {"type": "tool_end", "call_id": call_id,
                                                "name": name, "result": failure,
                                                "outcome": "error"})
            _emit(session, "tool_result", name=name, result=failure, streamed=False,
                  files=None, diff_summary=None, fmt=fmt_result(failure))
            _log(session, {"type": "tool_error", "name": name,
                           "python_function": pf_name, "result": failure,
                           "ts": datetime.datetime.now().isoformat(timespec="seconds")})
            raise

    fmt = fmt_result(result, streamed=streamed)
    if name == "read_file":
        path = args.get("path")
        if isinstance(path, str):
            command = _tool_module.get_async_command_for_output_path(path)
            if command:
                fmt = fmt_read_result_with_command(command, result, streamed=streamed)

    # ── PostToolUse hooks ──────────────────────────────────────────────
    # Cannot undo the tool; decision:block / exit 2 replaces the result the
    # model sees with the reason, updatedToolOutput replaces the text.
    post = _fire_hooks(session, "PostToolUse",
                       matcher_values=[canonical_tool_name(name), name],
                       **_hook_tool_payload_fields(session, name, args, call_id),
                       tool_response=result)
    if post.updated_tool_output is not None:
        result = post.updated_tool_output
        streamed = False
        fmt = fmt_result(result, streamed=False)
    if post.block:
        reason = post.reason or "tool output rejected by PostToolUse hook"
        result = f"ERROR: PostToolUse hook rejected this tool result: {reason}"
        streamed = False
        fmt = fmt_result(result, streamed=False)

    if journal is not None:
        # Persisted only after the side effects have happened: a tool_end
        # without a later assistant message tells recovery the result is
        # known and must be re-used, not re-computed.
        _write_journal_record(session, {"type": "tool_end", "call_id": call_id,
                                        "name": name, "result": result})

    # Pass file-change metadata from log_data to the event payload
    # so consumers (e.g. Telegram controller) can show "Changed path +5 -2".
    _emit(session, "tool_result",
          name=name, result=result, streamed=streamed,
          files=log_data.get("files"),
          diff_summary=log_data.get("diff_summary"),
          fmt=fmt)
    _log(session, {"type": "tool_result", "name": name,
                   "python_function": pf_name,
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


def run_turn(client: openai.OpenAI | SubprocessOpenAI, model: str, session: Session, task: str | None,
             *, cancel: CancelToken | None = None,
             timeout: float | None = None,
             tool_executor: "ToolExecutor | None" = None) -> SessionResult:
    """Run one agent turn and return a :class:`SessionResult`.

    The result reflects the session state at the end of the turn.
    ``final_reply`` is ``None`` if the turn was interrupted before a final answer.
    Pass ``task=None`` to retry the current conversation state without
    appending a synthetic user message (used by the REPL's ``/c`` command).

    Pass a :class:`CancelToken` to allow cooperative cancellation from another
    thread (e.g. a TUI stop button).

    ``timeout`` requests cooperative cancellation after the given number of
    seconds.  It does not interrupt an in-flight LLM request or tool call; a
    :class:`TimeoutError` is raised when execution next reaches a cancellation
    boundary.
    """
    if timeout is not None and timeout < 0:
        raise ValueError("timeout must be non-negative or None")

    active_cancel = cancel or CancelToken()
    timeout_expired: threading.Event | None = None
    timer: threading.Timer | None = None
    if timeout is not None:
        timeout_expired = threading.Event()

        def _expire() -> None:
            timeout_expired.set()
            active_cancel.cancel()

        timer = threading.Timer(timeout, _expire)

    global _in_turn
    _in_turn = True
    # Interrupt hooks need the live session; SIGINT may arrive mid-turn.
    _sigint_handler.session = session  # type: ignore[attr-defined]
    try:
        if tool_executor is not None:
            session["tool_executor"] = tool_executor
        if timer is not None:
            timer.start()
        return _run_turn(client, model, session, task, cancel=active_cancel,
                         timeout_expired=timeout_expired)
    finally:
        if timer is not None:
            timer.cancel()
        _in_turn = False
        _sigint_handler.session = None  # type: ignore[attr-defined]


def _run_turn(client: openai.OpenAI | SubprocessOpenAI, model: str, session: Session, task: str | None,
              cancel: CancelToken | None = None,
              timeout_expired: threading.Event | None = None) -> SessionResult:
    messages   = session["messages"]
    tools      = session["tools"]
    structured = session["structured"]
    journal    = session.get("_journal")

    # A turn that died between the model's tool_calls and their results
    # (Ctrl-C, crash, a tool thread killed) leaves an assistant message whose
    # calls were never answered.  Every provider rejects that transcript
    # ("No tool output found for function call …"), so the live session would
    # be poisoned for good — each later turn failing the same way.  Repair it
    # here, in-process, exactly as resume does when loading from disk.
    repaired = _repair_tool_call_pairing(messages)
    if repaired != messages:
        _log(session, {"type": "repair_tool_call_pairing",
                       "delta": len(repaired) - len(messages),
                       "ts": datetime.datetime.now().isoformat(timespec="seconds")})
        messages[:] = repaired

    # Per-turn hook state: fresh prompt_id, reset the Stop-continuation guard.
    hook_state = session.setdefault("_hook_state", {})
    hook_state["turn_id"] = uuid.uuid4().hex
    hook_state["stop_hook_active"] = False

    # ── UserPromptSubmit hooks ─────────────────────────────────────────
    # Fires before the task message is appended; block rejects the prompt
    # (nothing is sent to the model) and the reason becomes final_reply.
    if task is not None:
        ups = _fire_hooks(session, "UserPromptSubmit", prompt=task)
        if ups.block or ups.stop:
            reason = (ups.stop_reason or ups.reason
                      or "prompt blocked by UserPromptSubmit hook")
            notice = f"Prompt rejected by hook: {reason}"
            _log(session, {"type": "user_prompt_blocked", "reason": reason,
                           "ts": datetime.datetime.now().isoformat(timespec="seconds")})
            _emit(session, "hook_warning", text=notice,
                  fmt=f"{YEL}{notice}{RESET}")
            return _session_result_with_reply(session, notice)

    if journal is not None:
        _write_journal_record(session, {"type": "turn_start", "task": task})

    def _append_message(msg: dict[str, Any]) -> None:
        """Append a message to the history and durably journal it."""
        if journal is not None:
            _write_journal_record(session, {"type": "message", "msg": msg})
        messages.append(msg)

    now_ts = datetime.datetime.now().isoformat(timespec="seconds")
    if task is None:
        # The prior user message was already journaled before the failed
        # request.  Send that exact transcript again rather than turning a
        # retry into a misleading "go" or "proceed" message.
        pass
    elif messages and messages[-1].get("role") == "user":
        # Merge consecutive user messages to keep the API-valid
        # user/assistant alternation.
        old = messages[-1]["content"]
        messages[-1]["content"] = f"{old}\n\n{task}" if old else task
        messages[-1]["ts"] = now_ts
        if journal is not None:
            # A merge replaces prior history; replay it as a replacement,
            # rather than inventing a second submitted user message.
            _write_journal_record(session, {"type": "reset_messages", "reason": "user_merge",
                                            "messages": list(messages)})
    else:
        _append_message({"role": "user", "content": task, "ts": now_ts})
    if task is not None:
        _log(session, {"type": "user", "content": task, "ts": now_ts})

    total_tokens = 0
    max_tokens   = DEFAULT_MAX_TOKENS
    # Consecutive context-window rejections this turn.  Each rejection
    # triggers a compaction+retry; a second consecutive rejection means the
    # kept suffix alone no longer fits (pathological single tool result),
    # so we stop rather than loop forever.
    context_overflow_retries = 0
    # Pending model-facing token-awareness injection (countdown + optional
    # checkpoint reminder), suffixed onto the next tool-result message.
    pending_ta: str | None = None

    def _with_pending_ta(text: str) -> str:
        nonlocal pending_ta
        if pending_ta:
            text = f"{text}\n\n{pending_ta}"
            pending_ta = None
        # Hook additionalContext rides the same suffix path (Claude-API
        # semantics: no extra messages, cache-friendly).  Async hooks'
        # informational output is drained at this safe point too.
        async_ctx = _drain_pending_hook_context(session)
        if async_ctx:
            text = f"{text}\n\n{async_ctx}"
        state = session.get("_hook_state") or {}
        queued = state.pop("pending_context", None) or []
        if queued:
            text = f"{text}\n\n" + "\n".join(queued)
        return text

    def _check_cancelled() -> None:
        if cancel is None or not cancel.cancelled:
            return
        if timeout_expired is not None and timeout_expired.is_set():
            raise TimeoutError("run_turn timed out")
        raise KeyboardInterrupt()

    try:
        while True:
            _check_cancelled()
            kwargs: dict[str, Any] = dict(model=model, messages=messages, temperature=0)
            if structured:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            request_started_at = time.monotonic()
            try:
                resp  = _complete(client, session, **kwargs)
            except RateLimitError as exc:
                err = str(exc)
                _emit_and_log_error(session, exc, err,
                                    f"\n{RED}Rate limited: {err}{RESET}",
                                    client=client, request_started_at=request_started_at,
                                    error_kind="rate_limit")
                return _session_result(session)
            except Exception as exc:
                if _is_context_window_error(exc):
                    # The request never reached the model — the prompt exceeds
                    # the provider's token limit (e.g. the compaction trigger
                    # was misconfigured above the true context window, or a
                    # tool result jumped the size between measurements).
                    # Retry is only possible after shrinking the history.
                    err = f"Context window exceeded: {exc}"
                    _emit_and_log_error(session, exc, err,
                                        f"\n{YEL}{err}{RESET}",
                                        client=client,
                                        request_started_at=request_started_at,
                                        error_kind="context_window_exceeded")
                    context_overflow_retries += 1
                    compacted = compact_session(client, model, session)
                    session["compaction_last_prompt_tokens"] = 0
                    messages = session["messages"]
                    if compacted and context_overflow_retries < 3:
                        _emit(session, "context_overflow_retry",
                              retries=context_overflow_retries,
                              fmt=f"{DIM}{MAG}[context overflow] compacted; "
                                  f"retrying the request{RESET}")
                        continue
                    _emit(session, "context_overflow_abort",
                          retries=context_overflow_retries,
                          fmt=f"\n{RED}Context window still exceeded after "
                              f"compaction; aborting turn.{RESET}")
                    return _session_result(session)
                err = f"API error: {exc}"
                _emit_and_log_error(session, exc, err, f"\n{RED}Error: {err}{RESET}",
                                    client=client, request_started_at=request_started_at)
                return _session_result(session)
            _check_cancelled()
            msg   = resp.choices[0].message
            context_overflow_retries = 0

            # Accumulate token usage from the response and surface it to the user.
            usage = getattr(resp, "usage", None)
            if usage:
                session["llm_call_count"] = session.get("llm_call_count", 0) + 1
                pending_ta = _token_awareness_injection(session, usage)
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
                trigger = session.get("compaction_trigger_tokens", DEFAULT_COMPACTION_TRIGGER_TOKENS)
                _emit(session, "usage",
                      prompt=getattr(usage, "prompt_tokens", 0) or 0,
                      completion=getattr(usage, "completion_tokens", 0) or 0,
                      total=getattr(usage, "total_tokens", 0) or 0,
                      cached=getattr(usage, "cached_tokens", 0) or 0,
                      cache_write=getattr(usage, "cache_creation_tokens", 0) or 0,
                      fmt=f"{DIM}{MAG}[tokens] {fmt_usage(usage, compaction_trigger=trigger)}{RESET}")
                _log(session, {"type": "usage",
                               "prompt_tokens":      getattr(usage, "prompt_tokens", 0) or 0,
                               "completion_tokens":  getattr(usage, "completion_tokens", 0) or 0,
                               "total_tokens":       getattr(usage, "total_tokens", 0) or 0,
                               "cached_tokens":      getattr(usage, "cached_tokens", 0) or 0,
                               "cache_creation_tokens": getattr(usage, "cache_creation_tokens", 0) or 0,
                               **({"token_budget_remaining":
                                   max(0, (session.get("token_awareness_budget_tokens", 0) or 0)
                                       - (getattr(usage, "prompt_tokens", 0) or 0))}
                                  if session.get("token_awareness_enabled") else {}),
                               "ts": datetime.datetime.now().isoformat(timespec="seconds")})
            elif session.get("strict_cache_proof", True) and session.get("llm_call_count", 0) >= 1:
                age = _last_message_age_seconds(session)
                if age is not None and age > CACHE_COLD_GAP_SECONDS:
                    # Cold resume: cache has expired; a missing usage block on
                    # the first post-resume call is not a hard violation.
                    notice = (
                        f"{DIM}Prefix cache expired (last message {int(age or 0)}s old); "
                        f"usage metadata unavailable this turn.{RESET}"
                    )
                    _emit(session, "cache_cold", age=int(age or 0), fmt=notice)
                    session["_cache_cold_warned"] = True
                else:
                    # Tokens for this call are already paid; aborting would
                    # only waste them.  Continue with a temporary warning.
                    session["_cache_status"] = "missing"
                    notice = (
                        f"{YEL}⚠ No usage metadata from the server after the first call; "
                        f"continuing without strict cache proof. Warning clears on the "
                        f"next observed cache hit.{RESET}"
                    )
                    _emit(session, "cache_proof_missing", fmt=notice)
            _apply_compaction_policy(client, model, session, usage, phase="mid_turn")

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
                return _session_result(session)

            # ── structured tool_calls ────────────────────────────────────────────
            if structured and msg.tool_calls:
                now_ts = datetime.datetime.now().isoformat(timespec="seconds")
                _append_message({
                    "role": "assistant",
                    "tool_calls": [
                        (_tool_call_history_item(tc)) for tc in msg.tool_calls
                    ],
                    "ts": now_ts,
                })
                for tc in msg.tool_calls:
                    if tc.type == "custom":
                        # Custom tool call: the raw text input is dispatched
                        # as a single `input` kwarg — no JSON decoding.
                        args = {"input": tc.custom_input
                                if tc.custom_input is not None
                                else tc.function.arguments}
                    else:
                        try:
                            args = json.loads(tc.function.arguments)
                            if not isinstance(args, dict):
                                raise ValueError("arguments must be a JSON object")
                        except (json.JSONDecodeError, ValueError) as parse_exc:
                            # Feed the malformed call back to the model so it
                            # can emit corrected JSON on the next iteration
                            # instead of silently dispatching with {} args.
                            bad = tc.function.arguments or ""
                            result = (f"ERROR: malformed tool call arguments for "
                                      f"{tc.function.name!r}: {parse_exc}. "
                                      f"Received: {bad[:200]!r}. "
                                      f"Re-emit the tool call with valid JSON "
                                      f"object arguments.")
                            _write_journal_record(session, {
                                "type": "tool_start", "call_id": tc.id,
                                "name": tc.function.name, "args": {}})
                            _write_journal_record(session, {
                                "type": "tool_end", "call_id": tc.id,
                                "name": tc.function.name, "result": result,
                                "outcome": "error"})
                            _emit(session, "tool_result", name=tc.function.name,
                                  result=result, streamed=False, fmt=fmt_result(result))
                            _log(session, {"type": "tool_error",
                                           "name": tc.function.name,
                                           "result": result,
                                           "ts": datetime.datetime.now().isoformat(timespec="seconds")})
                            _append_message({"role": "tool", "tool_call_id": tc.id,
                                             "content": _with_pending_ta(result),
                                             "ts": datetime.datetime.now().isoformat(timespec="seconds")})
                            continue
                    result = _handle_tool_call(tc.function.name, args, session,
                                               call_id=tc.id)
                    _append_message({"role": "tool", "tool_call_id": tc.id,
                                     "content": _with_pending_ta(result),
                                     "ts": datetime.datetime.now().isoformat(timespec="seconds")})
                continue

            text = msg.content or ""

            # ── inline JSON tool calls ───────────────────────────────────────────
            if not structured:
                now_ts = datetime.datetime.now().isoformat(timespec="seconds")
                _append_message({"role": "assistant", "content": text, "ts": now_ts})
                calls = extract_inline_calls(text)
                if calls:
                    results = []
                    for name, args in calls:
                        result = _handle_tool_call(name, args, session)
                        results.append(f"[{name}] {_with_pending_ta(result)}")
                    _append_message({"role": "user",
                                     "content": "Tool results:\n" + "\n\n".join(results),
                                     "ts": datetime.datetime.now().isoformat(timespec="seconds")})
                    continue

            # ── final answer ─────────────────────────────────────────────────────
            # In structured mode the assistant message wasn't appended above.
            if structured:
                _append_message({"role": "assistant", "content": text,
                                 "ts": datetime.datetime.now().isoformat(timespec="seconds")})
            _log(session, {"type": "assistant", "content": text,
                       "ts": datetime.datetime.now().isoformat(timespec="seconds")})
            already_streamed = session.get("_content_was_streamed", False)
            _emit(session, "final_answer", text=text.strip(),
                  fmt="" if already_streamed else f"\n{GREEN}{BOLD}» {RESET}{text.strip()}\n")
            t = session["usage_totals"]
            cached_part = f"  |  cached {t['cached']:,}" if t["cached"] else ""
            _emit(session, "session_usage", **t,
                  fmt=(f"{DIM}{MAG}[session tokens] prompt {t['prompt']:,}{cached_part}  |  "
                       f"completion {t['completion']:,}{RESET}\n"))
            # ── Stop hooks ─────────────────────────────────────────────
            # decision:block / exit 2 refuses to stop: the reason becomes a
            # new user message and the loop continues.  stop_hook_active
            # guards against infinite continuation (a second block while
            # already true is ignored), exactly Claude/Codex's guard.
            state = session.setdefault("_hook_state", {})
            if not state.get("stop_hook_active"):
                stop = _fire_hooks(
                    session, "Stop",
                    stop_hook_active=False,
                    last_assistant_message=text)
                if stop.block and stop.reason:
                    state["stop_hook_active"] = True
                    _append_message({
                        "role": "user", "content": stop.reason,
                        "ts": datetime.datetime.now().isoformat(timespec="seconds")})
                    continue
            else:
                # Guarded continuation: report the active flag but ignore a
                # second block, exactly Claude/Codex's loop guard.
                _fire_hooks(session, "Stop", stop_hook_active=True,
                            last_assistant_message=text)
            # Build the result before turn-end compaction so final_reply
            # survives even when the policy summarizes the whole history away.
            session_result = _session_result(session)
            _apply_compaction_policy(client, model, session, usage, phase="turn_end")
            return session_result
    except CacheProofError as exc:
        err = str(exc)
        _emit_and_log_error(session, exc, err, f"\n{RED}Error: {err}{RESET}",
                            client=client)
        return _session_result(session)
    finally:
        # Durable: every exit path (final answer, error, interrupt) closes the
        # turn in the journal so replay knows where the turn ended.
        if journal is not None:
            _write_journal_record(session, {"type": "turn_end"})


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


def _load_azure_price_cache() -> "dict[str, Any]" | None:
    """Load cached Azure pricing data if it is less than 7 days old."""
    path = _azure_price_cache_path()
    if not path.exists():
        return None
    try:
        with path.open() as f:
            cache = cast(dict[str, Any], json.load(f))
        cached_at = cache.get("cached_at")
        if cached_at is None:
            return None
        age = datetime.datetime.now() - datetime.datetime.fromisoformat(cached_at)
        if age.days >= 7:
            return None
        return cache
    except Exception:
        return None


def _save_azure_price_cache(items: list[dict[str, Any]]) -> None:
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
        # (is_input, price_per_mtok, is_cache)
        global_entries: list[tuple[bool, float, bool]] = []
        dz_entries: list[tuple[bool, float, bool]] = []

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

            entry: "tuple[bool, float, bool]" = (is_input, price_per_mtok, is_cache)
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


def check_and_display_pricing(schema: "dict[str, Any]") -> None:
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

def validate_schema(schema: "dict[str, Any]") -> None:
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
    usage:       dict[str, Any]
    messages:    list[dict[str, Any]]


def _session_result(session: Session) -> SessionResult:
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


def _session_result_with_reply(session: Session, reply: str) -> SessionResult:
    """SessionResult carrying an explicit final reply (hook-blocked prompt)."""
    return SessionResult(
        session_id  = session["session_id"],
        final_reply = reply,
        usage       = dict(session["usage_totals"]),
        messages    = session["messages"],
    )


def _endpoint_is_openrouter(endpoint: str | None) -> bool:
    """True if *endpoint* points at openrouter.ai."""
    try:
        host = urllib.parse.urlparse(endpoint or "").hostname or ""
    except ValueError:
        return False
    return "openrouter.ai" in host


def _get_key_for_schema(schema: "dict[str, Any]") -> str:
    """Return the API key appropriate for this schema.

    Priority:
      1. keyring_service + keyring_username in the spec → keyring lookup,
         then the env var named by the uppercased username
      2. key_env in the spec → read that env variable
      3. Default, by endpoint:
         - OpenRouter endpoints: OPENROUTER_API_KEY through ensure_api_key()
           (balance check + rotation).
         - Any other endpoint: API_KEY, read as a plain key (env →
           password-get) without the OpenRouter management machinery, then
           OPENROUTER_API_KEY, which wrappers exported as a generic channel
           before API_KEY existed and which is kept for them.

    A configured-but-unresolvable source (1 or 2) raises AuthenticationError
    naming that source — an OpenRouter key is never silently substituted.
    """
    ks = schema.get("keyring_service")
    ku = schema.get("keyring_username")
    if ks and ku:
        import_error: Exception | None = None
        val: str | None = None
        try:
            import keyring as _kr
        except Exception as e:  # optional dependency may be missing
            import_error = e
        else:
            try:
                val = _kr.get_password(ks, ku)
            except Exception:
                val = None
        if val:
            return val
        # Fall back to env var named by keyring_username uppercased
        env_name = ku.upper().replace("-", "_")
        val = os.environ.get(env_name)
        if val:
            return val
        if import_error is not None:
            raise AuthenticationError(
                f"Cannot obtain API key from keyring ({ks}/{ku}): the keyring "
                f"package is unavailable ({import_error}). Install it with "
                f"pip install 'agentknit[keyring]' or set {env_name}."
            )
        raise AuthenticationError(
            f"Cannot obtain API key from keyring ({ks}/{ku}) or "
            f"environment variable {env_name}."
        )

    key_env = schema.get("key_env")
    if key_env:
        val = os.environ.get(key_env)
        if val:
            return val
        raise AuthenticationError(
            f"Cannot obtain API key: environment variable {key_env} is not set."
        )

    endpoint = schema.get("endpoint") or DEFAULT_ENDPOINT
    if _endpoint_is_openrouter(endpoint):
        return get_api_key()

    # Non-OpenRouter endpoint with no configured key source: a plain key
    # (env → password-get), without balance checks or rotation.
    #
    # API_KEY is the generic name and the one to reach for. OPENROUTER_API_KEY
    # is read after it because wrappers have been exporting it as the generic
    # channel since before there was one -- a single provider's name doing
    # duty for every provider. It keeps working; new callers should not learn
    # it.
    from .keys import _get_raw_key
    for env_name in ("API_KEY", "OPENROUTER_API_KEY"):
        val = _get_raw_key(env_name)
        if val:
            return val
    raise AuthenticationError(
        f"No API key source configured for endpoint {endpoint!r}. "
        "Set API_KEY, or 'key_env' or 'keyring_service'+'keyring_username' "
        "in the spec."
    )


def create_client(schema: "dict[str, Any]") -> "openai.OpenAI | SubprocessOpenAI":
    """Create an API client from a loaded agent spec schema.

    Handles subprocess (run://), OpenCode GitHub-Copilot, and standard
    OpenAI-compatible endpoints.  Call :func:`load_specification` first to obtain
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
    kwargs: dict[str, Any] = {}
    if max_rpm is not None:
        kwargs["max_rpm"] = max_rpm
    if binary_path is not None:
        return SubprocessOpenAI(binary_path)
    if auth == "opencode-github-copilot":
        return openai.OpenAI(api_key=_get_opencode_token(), base_url=endpoint,
                             auth_header="X-API-Key", **kwargs)
    return openai.OpenAI(api_key=_get_key_for_schema(schema), base_url=endpoint, **kwargs)


def run_task(
    schema: "dict[str, Any]",
    task: str,
    *,
    non_interactive: bool = False,
    session_id: str | None = None,
    cache_key: str | None = None,
    system_prompt_supplement: str = "",
    max_output_tokens: int | None = None,
    strict_cache_proof: bool = True,
    on_event: "EventCallback | None" = None,
    tool_executor: "ToolExecutor | None" = None,
    compaction_enabled: bool | None = None,
    compaction_trigger_tokens: int | None = None,
    compaction_target_tokens: int | None = None,
    compaction_keep_last_turns: int | None = None,
    compaction_policy: "str | Callable[..., bool] | None" = None,
    compaction_min_chars: int | None = None,
    min_cacheable_tokens: int | None = None,
    durable: bool | None = None,
    session_dir: str | Path | None = None,
    durable_sink: DurableSink | None = None,
    client: "openai.OpenAI | SubprocessOpenAI | None" = None,
    hooks: "str | Path | dict[str, Any] | list[Any] | None" = None,
    hooks_enabled: bool | None = None,
) -> SessionResult:
    """Run a single task against the agent and return a :class:`SessionResult`.

    This is the primary SDK entry point for one-shot programmatic use — no
    argparse, no stdin reading, no REPL loop.

    Example::

        schema = load_specification("qwen/qwen3-8b", "https://openrouter.ai/api/v1")
        result = run_task(schema, "List the files in /tmp")
        print(result.final_reply)
        print(result.usage)

    Pass ``min_cacheable_tokens=N`` (or set it in the schema) when the
    provider has a minimum cacheable prompt size, so small prompts that
    legitimately miss the cache don't trip strict cache-proof mode. See
    :func:`init_session` for details.

    Pass ``durable=False`` (or set ``"durable": false`` in the schema) to
    disable the write-ahead journal and fall back to turn-boundary
    snapshots only.

    Pass ``client=`` to inject a custom or wrapped client (e.g. a sandbox
    client or an instrumented subclass of
    :class:`~agentknit.openai_compat.SubprocessOpenAI`) instead of the one
    :func:`create_client` would build from the schema.  The schema is still
    validated and used for tools, prompts and session state.
    """
    if session_id is not None and session_dir is None:
        schema = _bind_schema_to_resumed_session(schema, session_id)
    validate_schema(schema)
    client = client or create_client(schema)
    session = init_session(
        schema,
        non_interactive=non_interactive,
        resumed_from=session_id,
        system_prompt_supplement=system_prompt_supplement,
        cache_key=cache_key,
        max_output_tokens=max_output_tokens,
        strict_cache_proof=strict_cache_proof,
        on_event=on_event,
        tool_executor=tool_executor,
        compaction_enabled=compaction_enabled,
        compaction_trigger_tokens=compaction_trigger_tokens,
        compaction_target_tokens=compaction_target_tokens,
        compaction_keep_last_turns=compaction_keep_last_turns,
        compaction_policy=compaction_policy,
        compaction_min_chars=compaction_min_chars,
        min_cacheable_tokens=min_cacheable_tokens,
        durable=durable,
        session_dir=session_dir,
        durable_sink=durable_sink,
        hooks=hooks,
        hooks_enabled=hooks_enabled,
    )
    try:
        return run_turn(client, schema["model"], session, task)
    finally:
        _save_messages_snapshot(session)
        _log(session, {"type": "session_end", "session_id": session["session_id"],
                       "reason": "run_task_complete"})
        _fire_session_end(session, "other")
        journal = session.get("_journal")
        if journal is not None:
            journal.close()


def run_agent(
    *,
    task: str,
    model: str,
    endpoint: str,
    tools: list[Tool],
    auth: str | None = None,
    non_interactive: bool = False,
    session_id: str | None = None,
    cache_key: str | None = None,
    system_prompt_supplement: str = "",
    max_output_tokens: int | None = None,
    strict_cache_proof: bool = True,
    on_event: "EventCallback | None" = None,
    tool_executor: "ToolExecutor | None" = None,
    compaction_enabled: bool | None = None,
    compaction_trigger_tokens: int | None = None,
    compaction_target_tokens: int | None = None,
    compaction_keep_last_turns: int | None = None,
    compaction_policy: "str | Callable[..., bool] | None" = None,
    compaction_min_chars: int | None = None,
    min_cacheable_tokens: int | None = None,
    durable: bool | None = None,
    session_dir: str | Path | None = None,
    durable_sink: DurableSink | None = None,
    client: "openai.OpenAI | SubprocessOpenAI | None" = None,
    hooks: "str | Path | dict[str, Any] | list[Any] | None" = None,
    hooks_enabled: bool | None = None,
) -> SessionResult:
    """Run a one-shot agent from direct tool definitions.

    This scripting-oriented convenience API creates the internal agent schema
    and registers each :class:`~agentknit.Tool` callable before delegating to
    :func:`run_task`.  Use :func:`run_task` when a checked-in or probed schema
    is already available.
    """
    tool_schema, tool_dispatch = build_tool_spec(tools)
    register_tools_in_library(tools)
    schema = {
        "model": model,
        "endpoint": endpoint,
        "inferred_tool_schema": tool_schema,
        "tool_dispatch": tool_dispatch,
    }
    if auth is not None:
        schema["auth"] = auth
    return run_task(
        schema, task,
        non_interactive=non_interactive,
        session_id=session_id,
        cache_key=cache_key,
        system_prompt_supplement=system_prompt_supplement,
        max_output_tokens=max_output_tokens,
        strict_cache_proof=strict_cache_proof,
        on_event=on_event,
        tool_executor=tool_executor,
        compaction_enabled=compaction_enabled,
        compaction_trigger_tokens=compaction_trigger_tokens,
        compaction_target_tokens=compaction_target_tokens,
        compaction_keep_last_turns=compaction_keep_last_turns,
        compaction_policy=compaction_policy,
        compaction_min_chars=compaction_min_chars,
        min_cacheable_tokens=min_cacheable_tokens,
        durable=durable,
        session_dir=session_dir,
        durable_sink=durable_sink,
        client=client,
        hooks=hooks,
        hooks_enabled=hooks_enabled,
    )


def run(
    schema: "dict[str, Any]" | None = None,
    task: str | None = None,
    *,
    model: str | None = None,
    endpoint: str | None = None,
    non_interactive: bool = False,
    session_id: str | None = None,
    cache_key: str | None = None,
    system_prompt_supplement: str = "",
    max_output_tokens: int | None = None,
    strict_cache_proof: bool = True,
    on_event: "EventCallback | None" = None,
    tool_executor: "ToolExecutor | None" = None,
    compaction_enabled: bool | None = None,
    compaction_trigger_tokens: int | None = None,
    compaction_target_tokens: int | None = None,
    compaction_keep_last_turns: int | None = None,
    compaction_policy: "str | Callable[..., bool] | None" = None,
    compaction_min_chars: int | None = None,
    min_cacheable_tokens: int | None = None,
    durable: bool | None = None,
    session_dir: str | Path | None = None,
    durable_sink: DurableSink | None = None,
    client: "openai.OpenAI | SubprocessOpenAI | None" = None,
    hooks: "str | Path | dict[str, Any] | list[Any] | None" = None,
    hooks_enabled: bool | None = None,
) -> SessionResult:
    """Backward-compatible helper for :func:`run_task`.

    Accepts either a loaded schema dict or a ``model``/``endpoint`` pair for
    wrapper scripts that want to skip ``agentknit.main()``.
    """
    if schema is None:
        if model is None or endpoint is None:
            raise TypeError("run() requires either `schema` or both `model` and `endpoint`.")
        schema = load_specification(model, endpoint)
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
        tool_executor=tool_executor,
        compaction_enabled=compaction_enabled,
        compaction_trigger_tokens=compaction_trigger_tokens,
        compaction_target_tokens=compaction_target_tokens,
        compaction_keep_last_turns=compaction_keep_last_turns,
        compaction_policy=compaction_policy,
        compaction_min_chars=compaction_min_chars,
        min_cacheable_tokens=min_cacheable_tokens,
        durable=durable,
        session_dir=session_dir,
        durable_sink=durable_sink,
        client=client,
        hooks=hooks,
        hooks_enabled=hooks_enabled,
    )


# Executables that are the agentknit CLI itself: they take <model> as a
# positional argument, so the resume hint must repeat it.  Any other
# executable (a wrapper such as `agent-glm-5.2.py`) pins the model itself
# and must be resumed without it.
_FRAMEWORK_CLI_NAMES = frozenset({"agentknit", "agent-probe"})


def _is_framework_cli(program: str) -> bool:
    """Return True if *program* is the agentknit CLI rather than a wrapper."""
    if not program:
        return False
    if Path(program).name in _FRAMEWORK_CLI_NAMES:
        return True
    try:
        return Path(program).resolve() == Path(__file__).resolve()
    except (OSError, ValueError):
        return False


def _build_resume_cmd(
    model: str,
    session_id: str,
    default_program: str | None = None,
    include_model: bool | None = None,
) -> str:
    """Build the ``Resume: ...`` hint printed when a session ends.

    Resolution order:

    1. ``AGENTKNIT_RESUME_COMMAND`` — wrappers that need full control set it;
       the value is used verbatim, with just ``--session`` appended.
    2. The running executable (``sys.argv[0]``, or *default_program*).
       The ``<model>`` argument is appended only when that executable is
       the agentknit CLI itself; wrapper executables embed the model, so
       their resume command is ``<executable> --session <id>``.

    *include_model* forces the decision (``True``/``False``) instead of
    deriving it from the program name.
    """
    program = os.environ.get("AGENTKNIT_RESUME_COMMAND")
    if program:
        return f"{program} --session {session_id}"
    if default_program is None:
        default_program = sys.argv[0]
    if include_model is None:
        include_model = _is_framework_cli(default_program)
    model_part = f"{model} " if include_model else ""
    return f"{default_program} {model_part}--session {session_id}"


def _repl_setup(
    schema: "dict[str, Any]",
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
    compaction_policy: "str | Callable[..., bool] | None" = None,
    compaction_min_chars: int | None = None,
    min_cacheable_tokens: int | None = None,
    durable: bool | None = None,
    session_dir: str | Path | None = None,
    durable_sink: DurableSink | None = None,
    client: "openai.OpenAI | SubprocessOpenAI | None" = None,
) -> tuple[Any, ...]:
    """Common REPL setup: validate, create client, init session, return (client, session, model, hist_file)."""
    if session_id is not None and session_dir is None:
        schema = _bind_schema_to_resumed_session(schema, session_id)
    validate_schema(schema)
    client = client or create_client(schema)
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
        compaction_policy=compaction_policy,
        compaction_min_chars=compaction_min_chars,
        min_cacheable_tokens=min_cacheable_tokens,
        durable=durable,
        session_dir=session_dir,
        durable_sink=durable_sink,
    )
    model = schema["model"]

    tool_specs = schema.get("inferred_tool_schema") or schema.get("tool_specs") or []
    if tool_specs:
        tool_names = [t.get("function", {}).get("name", "?") for t in tool_specs]
        print(f"Tools:    {', '.join(tool_names)}")
    else:
        print("Tools:    (none)")

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


def _fire_session_end(session: Session, reason: str) -> None:
    """Fire SessionEnd hooks (advisory, synchronous, short budget)."""
    try:
        _fire_hooks(session, "SessionEnd", matcher_values=[reason], reason=reason)
    except Exception:
        pass


def _repl_teardown(session: Session, hist_file: Path, resume_cmd: str) -> None:
    """Common REPL teardown: save history, snapshot, log."""
    try:
        readline.write_history_file(hist_file)
    except Exception:
        pass
    _save_messages_snapshot(session)
    _log(session, {"type": "session_end", "session_id": session["session_id"],
                   "reason": "repl_exit"})
    _fire_session_end(session, "prompt_input_exit")
    journal = session.get("_journal")
    if journal is not None:
        journal.close()
    print(f"\n{DIM}Resume: {resume_cmd}{RESET}")


def _repl_loop_body(
    t: str,
    client: openai.OpenAI | SubprocessOpenAI,
    session: Session,
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
    retried = False

    def _retry_turn() -> None:
        nonlocal retried
        retried = True
        if use_async_input:
            _async_repl_turn(None, client, session, current_model)
        else:
            _sync_repl_turn(None, client, session, current_model)

    if _slash_registry.dispatch(t, session, client, current_model, on_continue=_retry_turn):
        if not retried:
            _save_messages_snapshot(session)
        return

    if use_async_input:
        _async_repl_turn(t, client, session, current_model)
    else:
        _sync_repl_turn(t, client, session, current_model)


def _sync_repl_turn(
    t: str | None,
    client: openai.OpenAI | SubprocessOpenAI,
    session: Session,
    model: str,
) -> None:
    """Run a single turn synchronously — no background reader thread."""
    try:
        run_turn(client, model, session, t)
    except KeyboardInterrupt:
        print(f"\n{DIM}[interrupted]{RESET}")
    except Exception as exc:
        _emit_and_log_error(session, exc, str(exc), f"\n{RED}Error: {exc}{RESET}",
                            client=client)
    _save_messages_snapshot(session)


def _async_repl_turn(
    t: str | None,
    client: openai.OpenAI | SubprocessOpenAI,
    session: Session,
    model: str,
) -> None:
    """Run a turn with a background ``_InputCollector`` queuing keystrokes."""
    _collector = _InputCollector()
    _tool_module._input_collector = _collector
    _pending: list[str | None] = [t]
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
                _emit_and_log_error(session, exc, str(exc), f"\n{RED}Error: {exc}{RESET}",
                                    client=client)
            finally:
                _collector.stop()
            _save_messages_snapshot(session)
            if _interrupted:
                break
            for _qi in _collector.drain():
                _qs = _qi.strip()
                if not _qs or _qs.lower() in ("exit", "quit", "q"):
                    continue

                _continued = False

                def _on_continue() -> None:
                    nonlocal _continued
                    _continued = True
                    _pending.append(None)

                if not _slash_registry.dispatch(_qs, session, client, model,
                                                on_continue=_on_continue):
                    _pending.append(_qi)
                elif not _continued:
                    _save_messages_snapshot(session)
    finally:
        _tool_module._input_collector = None


def run_repl(
    schema: "dict[str, Any]",
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
    compaction_policy: "str | Callable[..., bool] | None" = None,
    compaction_min_chars: int | None = None,
    min_cacheable_tokens: int | None = None,
    durable: bool | None = None,
    session_dir: str | Path | None = None,
    durable_sink: DurableSink | None = None,
    client: "openai.OpenAI | SubprocessOpenAI | None" = None,
) -> None:
    """Start an interactive REPL session against the agent (sync, no background thread).

    Reads tasks line-by-line from stdin and runs :func:`run_turn` for each.
    The session snapshot is saved after every turn so it can be resumed with
    ``--session <session_id>``.

    Slash commands (``/c``, ``/clear``, ``/model``, ``/usage``, ``/help``) are
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
        compaction_policy=compaction_policy,
        compaction_min_chars=compaction_min_chars,
        min_cacheable_tokens=min_cacheable_tokens,
        durable=durable,
        session_dir=session_dir,
        durable_sink=durable_sink,
        client=client,
    )
    resume_cmd = _build_resume_cmd(model, session["session_id"], sys.argv[0])

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
    schema: "dict[str, Any]",
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
    compaction_policy: "str | Callable[..., bool] | None" = None,
    compaction_min_chars: int | None = None,
    min_cacheable_tokens: int | None = None,
    durable: bool | None = None,
    session_dir: str | Path | None = None,
    durable_sink: DurableSink | None = None,
    client: "openai.OpenAI | SubprocessOpenAI | None" = None,
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
        compaction_policy=compaction_policy,
        compaction_min_chars=compaction_min_chars,
        min_cacheable_tokens=min_cacheable_tokens,
        durable=durable,
        session_dir=session_dir,
        durable_sink=durable_sink,
        client=client,
    )
    resume_cmd = _build_resume_cmd(model, session["session_id"], sys.argv[0])

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
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Endpoint base URL")
    p.add_argument("--spec-path", metavar="PATH", dest="spec_path", default=None,
                   help="Load the agent spec from this JSON file, skipping all "
                        "name-based spec lookup and model probing")
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
    p.add_argument("--context-window", type=int, dest="context_window", default=None,
                   metavar="N",
                   help="Set/override the model's context window in tokens. Applied "
                        "on top of the loaded spec (including the default in-memory "
                        "spec for run:// models), so face launchers can declare the "
                        "window without materializing a temp spec file. Feeds the "
                        "token-awareness budget and context-window reminders.")
    p.add_argument("--no-strict-cache-proof", action="store_true",
                   help="Disable the default fail-closed cache-proof check that "
                        "requires a nonzero cache hit on every LLM call after the first.")
    p.add_argument("--min-cacheable-tokens", type=int, dest="min_cacheable_tokens", default=None,
                   help="Provider's minimum cacheable prompt size in tokens (e.g. 4096 for "
                        "Anthropic Claude Haiku, 1024 for GPT-5.6-class models). Below this "
                        "size, a zero-cache-hit response under strict cache-proof mode is "
                        "treated as expected rather than a failure. Overrides the spec's "
                        "min_cacheable_tokens.")
    p.add_argument("--no-durable", action="store_false", dest="durable", default=None,
                   help="Disable the write-ahead journal; fall back to turn-boundary "
                        "snapshots only. With durability on (default), every message, tool "
                        "call and tool result inside a turn is fsync'd to disk as it "
                        "happens, so a crashed session recovers to the exact point of "
                        "failure instead of losing the whole turn.")
    p.add_argument("--hooks", metavar="PATH", action="append", dest="hooks",
                   default=None,
                   help="Load Claude Code / Codex-compatible lifecycle hooks from this "
                        "hooks.json-shaped file (repeatable). Layers merge additively "
                        "with the spec's behaviour.hooks, <git-root>/.agentknit/hooks.json "
                        "and ~/.agentknit/hooks.json.")
    p.add_argument("--no-hooks", action="store_false", dest="hooks_enabled",
                   default=None,
                   help="Disable all lifecycle hooks for this session, whatever their "
                        "source.")
    return p.parse_intermixed_args()


def main() -> None:
    if sys.stdout.isatty():
        enable_osc8_hyperlinks()
    args   = parse_args()
    try:
        schema = load_specification(args.model, args.endpoint, spec_path=args.spec_path)
        if args.session:
            # A resumed session must continue on the endpoint it was run on,
            # not on whatever --endpoint / the OpenRouter default resolves to.
            schema = _bind_schema_to_resumed_session(schema, args.session)
        if args.context_window is not None:
            if args.context_window <= 0:
                sys.exit(f"{RED}--context-window must be a positive integer{RESET}")
            schema["context_window"] = args.context_window
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
    except RateLimitError as e:
        sys.exit(f"{RED}Rate limited: {e}{RESET}")

    model    = schema["model"]
    behaviour = schema.get("behaviour") or {}
    mode     = behaviour.get("call_delivery_mode", "structured_tool_calls")
    mode_str = f"  |  mode: {mode}" if mode != "structured_tool_calls" else ""

    opts: dict[str, Any] = dict(
        non_interactive          = args.non_interactive,
        resumed_from             = args.session,
        system_prompt_supplement = args.system_prompt_supplement,
        cache_key                = args.cache_key,
        max_output_tokens        = args.max_tokens,
        strict_cache_proof       = not args.no_strict_cache_proof,
        min_cacheable_tokens     = args.min_cacheable_tokens,
        durable                  = args.durable,
        hooks                    = args.hooks,
        hooks_enabled            = args.hooks_enabled,
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
            _fire_session_end(session, "other")
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
                _fire_session_end(session, "other")
                print(f"\n{DIM}Resume: {resume_cmd}{RESET}")
        return

    # Interactive REPL — reuse the already-created client + session.
    repl_opts = {k: v for k, v in opts.items() if k != "resumed_from"}
    repl_opts["session_id"] = opts.get("resumed_from")
    run_repl(schema, **repl_opts)


if __name__ == "__main__":
    main()
