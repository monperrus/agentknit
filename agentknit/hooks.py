"""Claude Code / Codex-compatible lifecycle hooks for agentknit.

This module implements the *common core* of the Claude Code and Codex CLI
hook conventions so that a ``hooks.json`` written for either tool works
unmodified:

* same three-level config shape — event → matcher group → handler list
* same matcher semantics — ``""``/``"*"`` match all, bare
  ``[A-Za-z0-9_ ,|-]`` strings are exact alternatives split on ``|``/``,
  anything else is an unanchored regular expression
* same input — one JSON object on stdin (command hooks) or as the dict
  argument (Python hooks), with the shared fields ``session_id``,
  ``transcript_path``, ``cwd``, ``hook_event_name``, ``model``,
  ``permission_mode`` plus per-event fields
* same output contract — exit 0 silent = no decision, exit 0 + JSON on
  stdout = structured control, exit 2 = blocking (stderr is the reason),
  any other exit code / timeout / invalid output = non-blocking error and
  the operation proceeds (fail open)

Strict symmetry between scripts and Python functions is enforced by
construction: both front-ends (:func:`run_command_hook`,
:func:`run_python_hook`) produce a :class:`RawHookResult`, and a single
:func:`normalize` implements the entire exit-code/JSON contract.  Nothing
downstream of :func:`normalize` can tell which front-end produced a
decision.

The module has no dependency on ``agentknit._core`` (it is imported *by*
it), so it can also be used standalone.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeAlias

__all__ = [
    "DEFAULT_HOOK_TIMEOUT",
    "HOOK_CONTEXT_LIMIT",
    "HOOK_EVENTS",
    "HOOKS_NEVER_FIRE",
    "HookBlock",
    "HookDecision",
    "HookEntry",
    "HookHandler",
    "RawHookResult",
    "canonical_tool_name",
    "cap_model_text",
    "combine_decisions",
    "load_hooks",
    "matcher_matches",
    "normalize",
    "parse_hooks_config",
    "register_hook",
    "run_command_hook",
    "run_hooks",
    "run_python_hook",
    "translate_updated_input",
]

# ── contract constants ────────────────────────────────────────────────────────

DEFAULT_HOOK_TIMEOUT = 60.0

# Events with short budgets (both upstreams: SessionEnd/Interrupt default to
# 1 s and are capped at 3 s).
_EVENT_TIMEOUT_DEFAULTS: dict[str, float] = {"SessionEnd": 1.0, "Interrupt": 1.0}
_EVENT_TIMEOUT_CAPS: dict[str, float] = {"SessionEnd": 3.0, "Interrupt": 3.0}

# Model-visible hook text is capped; overflow spills to disk and the model
# gets a head+tail preview plus the path (Claude's rule).
HOOK_CONTEXT_LIMIT = 10_000
_SPILL_HEAD = 4000
_SPILL_TAIL = 2000

# Events agentknit actually fires.
HOOK_EVENTS = frozenset({
    "SessionStart", "SessionEnd", "UserPromptSubmit",
    "PreToolUse", "PostToolUse", "Stop",
    "PreCompact", "PostCompact", "Interrupt",
})

# Parsed and loadable but never dispatched: agentknit has no approval gate,
# subagents, or notifications today.  A Claude/Codex settings file that
# configures them drops in without errors; the hooks simply never run.
HOOKS_NEVER_FIRE = frozenset({
    "PermissionRequest", "SubagentStart", "SubagentStop", "Notification",
})

_KNOWN_EVENTS = HOOK_EVENTS | HOOKS_NEVER_FIRE

# Per-event plain-stdout handling (identical in both upstreams).
_PLAIN_TEXT_CONTEXT_EVENTS = frozenset({"SessionStart", "SubagentStart", "UserPromptSubmit"})
_PLAIN_TEXT_IGNORED_EVENTS = frozenset({
    "PreToolUse", "PermissionRequest", "PostToolUse", "PreCompact",
    "PostCompact", "SessionEnd", "Interrupt", "Notification",
})
_JSON_REQUIRED_EVENTS = frozenset({"Stop", "SubagentStop"})

# Events where exit code 2 (or a blocking decision) changes the outcome.
_CAN_BLOCK_EVENTS = frozenset({
    "PreToolUse", "UserPromptSubmit", "PostToolUse",
    "Stop", "SubagentStop", "PreCompact",
})

# Events that accept hookSpecificOutput.additionalContext.
_ADDITIONAL_CONTEXT_EVENTS = frozenset({
    "SessionStart", "SubagentStart", "UserPromptSubmit", "PreToolUse",
    "PostToolUse", "Stop", "SubagentStop", "PreCompact", "PostCompact",
})

_VALID_PERMISSION_DECISIONS = ("allow", "deny", "ask")

# native agentknit tool name → Claude-canonical name reported as tool_name.
_TOOL_NAME_ALIASES: dict[str, str] = {
    "exec_shell": "Bash",
    "nohup": "Bash",
    "nohup_wait": "Bash",
    "nohup_query": "Bash",
    "read_file": "Read",
    "write_file": "Write",
    "str_replace": "Edit",
    "update": "Edit",
    "update_file": "Edit",
    "search": "Grep",
    "search_files": "Grep",
    "glob": "Glob",
    "find_files": "Glob",
    "list_dir": "LS",
    "list_directory": "LS",
}

# Claude-style argument names → agentknit-native argument names, applied to
# PreToolUse updatedInput before dispatch.
_ARG_ALIASES: dict[str, str] = {
    "file_path": "path",
    "old_string": "old_str",
    "new_string": "new_str",
}

# Claude Bash-tool metadata agentknit's exec_shell does not consume; dropped
# silently from updatedInput rather than rejected.
_CLAUDE_METADATA_ARGS = frozenset({"description", "timeout", "run_in_background"})


def canonical_tool_name(native_name: str) -> str:
    """Return the Claude-canonical name for an agentknit tool, or the name."""
    return _TOOL_NAME_ALIASES.get(native_name, native_name)


def translate_updated_input(updated: dict[str, Any]) -> dict[str, Any]:
    """Translate a Claude-shaped ``updatedInput`` into agentknit argument names.

    Known aliases are renamed (``file_path`` → ``path``, …), Claude Bash
    metadata keys are dropped, everything else passes through unchanged.
    """
    out: dict[str, Any] = {}
    for key, value in updated.items():
        if key in _CLAUDE_METADATA_ARGS:
            continue
        out[_ARG_ALIASES.get(key, key)] = value
    return out


# ── data structures ───────────────────────────────────────────────────────────

PythonHookFn: TypeAlias = Callable[[dict[str, Any]], Any]


class HookBlock(Exception):
    """Raise from a Python hook to produce the exit-code-2 blocking outcome."""


@dataclass
class RawHookResult:
    """Front-end-neutral result of running one hook handler.

    ``rc`` is ``None`` when the hook could not run at all or timed out, in
    which case ``error`` describes the failure (non-blocking: proceed).
    """

    rc: int | None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None


@dataclass
class HookDecision:
    """Normalized, event-interpreted decision from one hook (or combined)."""

    error: str | None = None
    system_message: str | None = None
    additional_context: str | None = None
    block: bool = False
    reason: str | None = None
    permission_decision: str | None = None       # allow | deny | ask (PreToolUse)
    permission_decision_reason: str | None = None
    updated_input: dict[str, Any] | None = None  # PreToolUse (with allow/ask)
    updated_tool_output: str | None = None       # PostToolUse
    stop: bool = False                           # continue: false
    stop_reason: str | None = None


@dataclass
class HookHandler:
    """One handler inside a matcher group (or registered via the API)."""

    type: str                                    # "command" | "python"
    command: str | None = None
    args: "list[str] | None" = None              # exec form when set (no shell)
    fn: PythonHookFn | None = None
    timeout: float | None = None
    async_: bool = False
    status_message: str | None = None
    additional_context_limit: int | None = None

    def describe(self) -> str:
        if self.type == "python" and self.fn is not None:
            return f"python:{getattr(self.fn, '__name__', repr(self.fn))}"
        return f"{self.command}{' ' + ' '.join(self.args) if self.args else ''}"


@dataclass
class HookEntry:
    """One event/matcher/handler triple from any config layer or the API."""

    event: str
    matcher: str
    handler: HookHandler
    source: str = "config"

    def matches(self, values: "list[str] | tuple[str, ...]") -> bool:
        """True when the matcher hits any of *values* (or needs no values)."""
        if not values:
            return True  # no matcher support for this event: always fires
        return any(matcher_matches(self.matcher, v) for v in values)

    def label(self) -> str:
        return f"{self.event}[{self.matcher or '*'}] {self.handler.describe()}"


def new_hook_state() -> dict[str, Any]:
    """Fresh per-session runtime state consumed by :func:`run_hooks`."""
    return {"turn_id": None, "stop_hook_active": False,
            "pending_context": [], "async_results": []}


# ── matcher engine ────────────────────────────────────────────────────────────

_EXACT_MATCHER_RE = re.compile(r"^[A-Za-z0-9_ ,|\-]*$")


def matcher_matches(matcher: str | None, value: str) -> bool:
    """Evaluate a Claude/Codex matcher against one value.

    ``""``, ``"*"`` or ``None`` match everything.  A matcher containing only
    letters, digits, ``_``, ``-``, spaces, ``,`` and ``|`` is a list of exact
    alternatives.  Anything else is an unanchored regular expression.
    """
    if matcher is None or matcher in ("", "*"):
        return True
    if _EXACT_MATCHER_RE.match(matcher):
        return any(part.strip() == value for part in re.split(r"[|,]", matcher))
    try:
        return re.search(matcher, value) is not None
    except re.error:
        return False


# ── config parsing ────────────────────────────────────────────────────────────

def parse_hooks_config(source: Any) -> tuple[list[HookEntry], list[str]]:
    """Parse one hooks config *source* into entries plus warnings.

    *source* may be:

    * a path (``str`` / ``Path``) to a ``hooks.json``-shaped JSON file
    * a dict with a ``"hooks"`` key (the same JSON shape, inline)
    * a list of any of the above (merged additively)

    Unknown events and unsupported handler types produce a warning and are
    skipped — never an error — so a full Claude/Codex settings file drops in.
    """
    entries: list[HookEntry] = []
    warnings: list[str] = []
    _parse_source(source, entries, warnings)
    return entries, warnings


def _parse_source(source: Any, entries: list[HookEntry], warnings: list[str]) -> None:
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        try:
            data = json.loads(path.read_text())
        except FileNotFoundError:
            warnings.append(f"hooks file not found: {path}")
            return
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"cannot read hooks file {path}: {exc}")
            return
        _parse_config_dict(data, entries, warnings, str(path))
    elif isinstance(source, dict):
        # Accept both the full file shape ({"hooks": {...}}) and the bare
        # event map ({"PreToolUse": [...]}) — specs commonly inline the
        # latter under behaviour.hooks.
        _parse_config_dict(source if "hooks" in source else {"hooks": source},
                           entries, warnings, "inline")
    elif isinstance(source, (list, tuple)):
        for item in source:
            _parse_source(item, entries, warnings)
    else:
        warnings.append(f"ignoring hooks source of type {type(source).__name__}")


def _parse_config_dict(data: Any, entries: list[HookEntry],
                       warnings: list[str], source: str) -> None:
    if not isinstance(data, dict):
        warnings.append(f"{source}: hooks config must be a JSON object")
        return
    hooks = data.get("hooks")
    if not isinstance(hooks, dict):
        warnings.append(f"{source}: missing 'hooks' object")
        return
    for event, groups in hooks.items():
        if event not in _KNOWN_EVENTS:
            warnings.append(f"{source}: ignoring unsupported hook event {event!r}")
            continue
        if not isinstance(groups, list):
            warnings.append(f"{source}: {event}: matcher groups must be a list")
            continue
        for group in groups:
            if not isinstance(group, dict):
                warnings.append(f"{source}: {event}: matcher group must be an object")
                continue
            matcher = group.get("matcher") or ""
            handlers = group.get("hooks") or []
            if not isinstance(handlers, list):
                warnings.append(f"{source}: {event}: 'hooks' must be a list")
                continue
            for h in handlers:
                handler = _parse_handler(h, event, source, warnings)
                if handler is not None:
                    entries.append(HookEntry(event=event, matcher=matcher,
                                             handler=handler, source=source))


def _parse_handler(h: Any, event: str, source: str,
                   warnings: list[str]) -> HookHandler | None:
    if not isinstance(h, dict):
        warnings.append(f"{source}: {event}: handler must be an object")
        return None
    htype = h.get("type") or ("command" if h.get("command") else None)
    if htype in ("mcp_tool", "prompt", "agent"):
        warnings.append(f"{source}: {event}: handler type {htype!r} is not "
                        f"supported yet; skipped")
        return None
    if htype != "command":
        warnings.append(f"{source}: {event}: handler needs type 'command'")
        return None
    command = h.get("command")
    if not isinstance(command, str) or not command:
        warnings.append(f"{source}: {event}: command handler needs a 'command' string")
        return None
    args = h.get("args")
    if args is not None and not isinstance(args, list):
        warnings.append(f"{source}: {event}: 'args' must be a list")
        args = None
    if sys.platform == "win32" and isinstance(h.get("commandWindows"), str):
        command = h["commandWindows"]
    timeout = h.get("timeout")
    if not isinstance(timeout, (int, float)) or timeout < 0:
        timeout = None
    limit = h.get("additionalContextLimit")
    if not isinstance(limit, int) or limit < 0:
        limit = None
    return HookHandler(
        type="command",
        command=command,
        args=[str(a) for a in args] if args is not None else None,
        timeout=float(timeout) if timeout is not None else None,
        async_=bool(h.get("async", False)),
        status_message=h.get("statusMessage") if isinstance(h.get("statusMessage"), str) else None,
        additional_context_limit=limit,
    )


# ── public session-level API ──────────────────────────────────────────────────

def register_hook(session: dict[str, Any], event: str, fn: PythonHookFn | None = None,
                  *, matcher: str = "", timeout: float | None = None,
                  async_: bool = False, command: str | None = None,
                  args: "list[str] | None" = None,
                  source: str = "api") -> HookEntry:
    """Register a Python (or command) hook on a session dict — the API twin
    of one ``hooks.json`` entry.

    Pass *fn* (called as ``fn(payload_dict) -> dict | str | None``) for a
    Python hook, or *command* (+ optional *args*, exec form) for a command
    hook.  A returned dict is the JSON-on-stdout outcome, a returned string
    is plain stdout, ``None``/``{}`` is exit 0 silent, and raising
    :class:`HookBlock` is the exit-2 blocking outcome.
    """
    if event not in _KNOWN_EVENTS:
        raise ValueError(f"unknown hook event {event!r}; known: "
                         f"{', '.join(sorted(_KNOWN_EVENTS))}")
    if fn is None and command is None:
        raise ValueError("either fn (python hook) or command (command hook) is required")
    handler = HookHandler(
        type="python" if fn is not None else "command",
        command=command, args=args, fn=fn,
        timeout=timeout, async_=async_,
    )
    entry = HookEntry(event=event, matcher=matcher or "",
                      handler=handler, source=source)
    session.setdefault("hooks", []).append(entry)
    session.setdefault("_hook_state", new_hook_state())
    return entry


def load_hooks(session: dict[str, Any], source: Any) -> tuple[list[HookEntry], list[str]]:
    """Merge a hooks config *source* into ``session['hooks']``.

    Returns the newly added entries and any config warnings (unknown
    events, unreadable files, unsupported handler types).
    """
    entries, warnings = parse_hooks_config(source)
    if entries:
        session.setdefault("hooks", []).extend(entries)
        session.setdefault("_hook_state", new_hook_state())
    return entries, warnings


# ── the two front-ends ────────────────────────────────────────────────────────

def _force_text(data: Any) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode(errors="replace")
    return str(data)


def _resolve_timeout(handler: HookHandler, event: str) -> float:
    timeout = handler.timeout
    if timeout is None:
        timeout = _EVENT_TIMEOUT_DEFAULTS.get(event, DEFAULT_HOOK_TIMEOUT)
    cap = _EVENT_TIMEOUT_CAPS.get(event)
    if cap is not None:
        timeout = min(timeout, cap)
    return max(0.0, float(timeout))


def run_command_hook(handler: HookHandler, payload: dict[str, Any],
                     cwd: str | Path | None = None,
                     timeout: float | None = None) -> RawHookResult:
    """Run a command hook: JSON payload on stdin, stdout/stderr captured.

    Shell form (no ``args``) runs the command string through the shell with
    the session cwd; exec form (``args`` set) resolves ``command`` on PATH
    and spawns it directly with no shell.
    """
    stdin_text = json.dumps(payload, ensure_ascii=False)
    try:
        if handler.args is not None:
            executable = shutil.which(handler.command or "") or handler.command or ""
            argv: Any = [executable, *handler.args]
            proc = subprocess.run(argv, input=stdin_text, cwd=cwd,
                                  capture_output=True, text=True,
                                  timeout=timeout if timeout else None)
        else:
            proc = subprocess.run(handler.command or "", shell=True,
                                  input=stdin_text, cwd=cwd,
                                  capture_output=True, text=True,
                                  timeout=timeout if timeout else None)
    except subprocess.TimeoutExpired as exc:
        return RawHookResult(None, _force_text(exc.stdout), _force_text(exc.stderr),
                             f"hook timed out after {timeout}s")
    except OSError as exc:
        return RawHookResult(None, "", "", f"failed to run hook: {exc}")
    return RawHookResult(proc.returncode, proc.stdout or "", proc.stderr or "")


def run_python_hook(fn: PythonHookFn, payload: dict[str, Any],
                    timeout: float | None = None) -> RawHookResult:
    """Run a Python hook under the same contract as a command hook.

    Mapping from function behaviour to the command-hook wire format:

    * returns ``None``/``{}``        → exit 0, no stdout
    * returns a ``dict``             → exit 0, that JSON on stdout
    * returns a ``str``              → exit 0, that plain text on stdout
    * raises :class:`HookBlock`      → exit 2, message on stderr
    * raises anything else           → exit 1, traceback text on stderr
    """
    box: dict[str, Any] = {}

    def _target() -> None:
        try:
            out = fn(dict(payload))
        except HookBlock as exc:
            box["rc"], box["stderr"] = 2, str(exc)
        except Exception as exc:  # non-blocking error, like a crashed script
            box["rc"], box["stderr"] = 1, f"{type(exc).__name__}: {exc}"
        else:
            if out is None:
                box["rc"], box["stdout"] = 0, ""
            elif isinstance(out, dict):
                box["rc"] = 0
                box["stdout"] = json.dumps(out, ensure_ascii=False) if out else ""
            elif isinstance(out, str):
                box["rc"], box["stdout"] = 0, out
            else:
                box["rc"], box["stderr"] = 1, (f"hook returned "
                                               f"{type(out).__name__}; expected "
                                               f"dict | str | None")

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout if timeout else None)
    if thread.is_alive():
        return RawHookResult(None, "", "", f"python hook timed out after {timeout}s")
    return RawHookResult(box.get("rc", 1), box.get("stdout", ""),
                         box.get("stderr", ""))


# ── normalization: the single implementation of the output contract ───────────

def normalize(raw: RawHookResult, event: str) -> HookDecision:
    """Interpret one :class:`RawHookResult` for *event*.

    Both front-ends feed this function; it is the only place the exit-code /
    JSON contract lives.  Failures are non-blocking: an error is reported and
    the decision contributes nothing (except that exit 2 still blocks).
    """
    decision = HookDecision()
    if raw.error is not None:
        decision.error = raw.error
        return decision

    text = raw.stdout.strip()
    parsed: dict[str, Any] | None = None
    parse_problem: str | None = None
    if text:
        try:
            candidate = json.loads(text)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(candidate, dict):
                parsed = candidate
            else:
                parse_problem = "stdout is JSON but not an object"

    applied = HookDecision()
    validation_error: str | None = None
    if parsed is not None:
        validation_error = _apply_json(event, parsed, applied)

    if raw.rc == 2:
        if event in _CAN_BLOCK_EVENTS:
            decision.block = True
            reason = (applied.reason or applied.permission_decision_reason
                      or raw.stderr.strip())
            decision.reason = reason or "blocked by hook"
            if event == "PreToolUse":
                decision.permission_decision = "deny"
        elif raw.stderr.strip():
            # Advisory events: stderr is shown to the user only.
            decision.system_message = raw.stderr.strip()
        if validation_error or parse_problem:
            decision.error = validation_error or parse_problem
        return decision

    if validation_error is not None or parse_problem is not None:
        decision.error = validation_error or parse_problem
        return decision
    if parsed is not None:
        return applied

    # No JSON on stdout.
    if raw.rc == 0:
        if not text:
            return decision  # exit 0, silent: no decision
        if event in _PLAIN_TEXT_CONTEXT_EVENTS:
            decision.additional_context = text
            return decision
        if event in _JSON_REQUIRED_EVENTS:
            decision.error = (f"{event} expects JSON on stdout; "
                              f"plain text output is invalid")
        return decision
    decision.error = f"hook exited with code {raw.rc}"
    stderr = raw.stderr.strip()
    if stderr:
        decision.error += f": {stderr}"
    return decision


def _apply_json(event: str, obj: dict[str, Any], d: HookDecision) -> str | None:
    """Apply the parsed JSON fields for *event* onto *d*.

    Returns an error string when the object fails schema validation (wrong
    field for the event, wrong types, mismatched ``hookEventName``, …).
    """
    errors: list[str] = []

    cont = obj.get("continue")
    if cont is not None:
        if not isinstance(cont, bool):
            errors.append("'continue' must be a boolean")
        elif cont is False:
            if event == "PreToolUse":
                errors.append("'continue' is not supported for PreToolUse")
            else:
                d.stop = True
                stop_reason = obj.get("stopReason")
                if stop_reason is not None:
                    if isinstance(stop_reason, str):
                        d.stop_reason = stop_reason
                    else:
                        errors.append("'stopReason' must be a string")

    system_message = obj.get("systemMessage")
    if system_message is not None:
        if isinstance(system_message, str):
            d.system_message = system_message
        else:
            errors.append("'systemMessage' must be a string")

    # 'suppressOutput' is parsed but ignored, as in both upstreams.

    decision = obj.get("decision")
    reason = obj.get("reason")
    if decision is not None:
        if not isinstance(decision, str):
            errors.append("'decision' must be a string")
        elif decision in ("block", "approve"):
            if event not in _CAN_BLOCK_EVENTS:
                errors.append(f"'decision' is not supported for {event}")
            elif decision == "approve":
                # Deprecated Claude value; only meaningful for PreToolUse.
                if event == "PreToolUse":
                    d.permission_decision = "allow"
            else:
                d.block = True
                d.reason = reason if isinstance(reason, str) else "blocked by hook"
                if event == "PreToolUse":
                    d.permission_decision = "deny"
        else:
            errors.append(f"unsupported decision value {decision!r}")

    hso = obj.get("hookSpecificOutput")
    if hso is not None:
        if not isinstance(hso, dict):
            errors.append("'hookSpecificOutput' must be an object")
        else:
            hook_event_name = hso.get("hookEventName")
            if hook_event_name != event:
                errors.append(f"hookSpecificOutput.hookEventName "
                              f"{hook_event_name!r} does not match event {event!r}")
            else:
                _apply_hook_specific(event, hso, d, errors)

    if errors:
        return "; ".join(errors)
    return None


def _apply_hook_specific(event: str, hso: dict[str, Any],
                         d: HookDecision, errors: list[str]) -> None:
    pd = hso.get("permissionDecision")
    if pd is not None:
        if event == "PreToolUse" and pd in _VALID_PERMISSION_DECISIONS:
            d.permission_decision = pd
            pdr = hso.get("permissionDecisionReason")
            if pdr is not None:
                if isinstance(pdr, str):
                    d.permission_decision_reason = pdr
                else:
                    errors.append("'permissionDecisionReason' must be a string")
        else:
            errors.append(f"'permissionDecision' value {pd!r} is not "
                          f"supported for {event}")

    updated_input = hso.get("updatedInput")
    if updated_input is not None:
        if (event == "PreToolUse" and isinstance(updated_input, dict)
                and d.permission_decision in ("allow", "ask")):
            d.updated_input = updated_input
        else:
            errors.append("'updatedInput' is only valid on PreToolUse with "
                          "permissionDecision allow/ask")

    updated_output = hso.get("updatedToolOutput")
    if updated_output is not None:
        if event == "PostToolUse" and isinstance(updated_output, str):
            d.updated_tool_output = updated_output
        else:
            errors.append("'updatedToolOutput' is only valid on PostToolUse")

    additional_context = hso.get("additionalContext")
    if additional_context is not None:
        if event in _ADDITIONAL_CONTEXT_EVENTS and isinstance(additional_context, str):
            d.additional_context = additional_context
        else:
            errors.append(f"'additionalContext' is not supported for {event}")


# ── combination and dispatch ──────────────────────────────────────────────────

def combine_decisions(decisions: "list[HookDecision]") -> HookDecision:
    """Combine the decisions of all matching hooks for one event.

    Precedence rules (common core of both upstreams): any ``deny`` wins over
    ``ask`` which wins over ``allow``; any exit-2-style block counts as deny;
    ``continue: false`` (``stop``) takes precedence over event-specific
    decisions; ``additionalContext``/``systemMessage`` from several hooks are
    all delivered.
    """
    out = HookDecision()
    errors = [d.error for d in decisions if d.error]
    if errors:
        out.error = "; ".join(errors)
    messages = [d.system_message for d in decisions if d.system_message]
    if messages:
        out.system_message = "\n".join(messages)
    contexts = [d.additional_context for d in decisions if d.additional_context]
    if contexts:
        out.additional_context = "\n".join(contexts)

    stops = [d for d in decisions if d.stop]
    if stops:
        out.stop = True
        out.stop_reason = next((d.stop_reason for d in stops if d.stop_reason), None)
        return out  # continue:false wins over any event-specific decision

    permission_decisions = [d.permission_decision for d in decisions
                            if d.permission_decision]
    if permission_decisions:
        for level in ("deny", "ask", "allow"):
            if level in permission_decisions:
                out.permission_decision = level
                winners = [d for d in decisions if d.permission_decision == level]
                out.permission_decision_reason = next(
                    (w.permission_decision_reason for w in winners
                     if w.permission_decision_reason), None)
                out.updated_input = next(
                    (w.updated_input for w in winners if w.updated_input), None)
                break

    blocks = [d for d in decisions if d.block]
    if blocks:
        out.block = True
        out.reason = next((b.reason for b in blocks if b.reason), None) \
            or "blocked by hook"
        if not out.permission_decision and any(
                d.permission_decision == "deny" for d in decisions):
            out.permission_decision = "deny"

    outputs = [d.updated_tool_output for d in decisions
               if d.updated_tool_output is not None]
    if outputs:
        out.updated_tool_output = outputs[0]
    return out


def cap_model_text(text: str, limit: int, spill_dir: str | Path | None) -> str:
    """Cap model-visible hook text, spilling overflow to disk.

    Over ``limit`` characters the full text is written under
    ``<spill_dir>/hook_outputs/`` and replaced by a head+tail preview plus
    the saved path.  Without a usable *spill_dir* a plain truncation is used.
    """
    if limit <= 0 or len(text) <= limit:
        return text
    head, tail = text[:_SPILL_HEAD], text[-_SPILL_TAIL:]
    omitted = len(text) - _SPILL_HEAD - _SPILL_TAIL
    if spill_dir is None:
        return (f"{head}\n\n…[{omitted} chars omitted from hook output]\n\n{tail}")
    try:
        directory = Path(spill_dir) / "hook_outputs"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{uuid.uuid4().hex}.txt"
        path.write_text(text, encoding="utf-8")
    except OSError:
        return f"{head}\n\n…[{omitted} chars omitted from hook output]\n\n{tail}"
    return (f"{head}\n\n…[{omitted} chars omitted; full hook output saved to "
            f"{path}]\n\n{tail}")


def _run_entry(entry: HookEntry, payload: dict[str, Any],
               cwd: str | Path | None, event: str) -> RawHookResult:
    timeout = _resolve_timeout(entry.handler, event)
    if entry.handler.type == "command":
        return run_command_hook(entry.handler, payload, cwd, timeout)
    assert entry.handler.fn is not None
    return run_python_hook(entry.handler.fn, payload, timeout)


def _cap_entry_context(decision: HookDecision, entry: HookEntry,
                       spill_dir: str | Path | None) -> None:
    if decision.additional_context is None:
        return
    limit = (entry.handler.additional_context_limit
             if entry.handler.additional_context_limit is not None
             else HOOK_CONTEXT_LIMIT)
    decision.additional_context = cap_model_text(
        decision.additional_context, limit, spill_dir)


def run_hooks(entries: "list[HookEntry]", event: str, payload: dict[str, Any], *,
              cwd: str | Path | None = None,
              matcher_values: "list[str] | tuple[str, ...] | None" = None,
              state: dict[str, Any] | None = None,
              spill_dir: str | Path | None = None,
              notify: "Callable[[str, dict[str, Any]], None] | None" = None,
              ) -> HookDecision:
    """Dispatch all matching hooks for *event* and combine their decisions.

    Matching handlers run concurrently.  ``async`` handlers run in the
    background: their decisions are discarded and their informational output
    is queued in ``state['async_results']`` for delivery at the next safe
    point (drained by the session integration in ``_core``).
    """
    matching = [e for e in entries
                if e.event == event and e.matches(matcher_values or [])]
    if not matching:
        return HookDecision()

    for entry in matching:
        if entry.handler.async_:
            continue
        if notify is not None:
            notify("hook_status", {
                "text": entry.handler.status_message or entry.label(),
                "source": entry.source,
                "fmt": f"[hook] {entry.handler.status_message or entry.label()}",
            })

    sync_entries = [e for e in matching if not e.handler.async_]
    for entry in matching:
        if entry.handler.async_:
            threading.Thread(
                target=_run_async_entry, args=(entry, payload, cwd, event, state, spill_dir),
                daemon=True).start()

    if not sync_entries:
        return HookDecision()
    decisions: list[HookDecision] = []
    with ThreadPoolExecutor(max_workers=min(8, len(sync_entries))) as executor:
        futures = [(entry, executor.submit(_run_entry, entry, payload, cwd, event))
                   for entry in sync_entries]
        for entry, future in futures:
            decision = normalize(future.result(), event)
            _cap_entry_context(decision, entry, spill_dir)
            decisions.append(decision)
    return combine_decisions(decisions)


def _run_async_entry(entry: HookEntry, payload: dict[str, Any],
                     cwd: str | Path | None, event: str,
                     state: dict[str, Any] | None,
                     spill_dir: str | Path | None) -> None:
    decision = normalize(_run_entry(entry, payload, cwd, event), event)
    _cap_entry_context(decision, entry, spill_dir)
    if decision.additional_context or decision.system_message:
        if state is not None:
            state.setdefault("async_results", []).append({
                "context": decision.additional_context,
                "system_message": decision.system_message,
            })
