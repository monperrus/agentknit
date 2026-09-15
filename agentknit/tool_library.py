"""
Tool implementations for agent_probe.py.

Every callable here is a candidate value for the "python_function" field in
tool_dispatch.  New functions added by the probe's code-generation path land
here too (appended via _register_generated).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import traceback
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


# ── async shell execution ─────────────────────────────────────────────────────
# The implementation (state, registry, nohup tool specs) lives in
# async_toolkit; the names below are re-exported for backward compatibility.

from .async_toolkit import (  # noqa: E402,A
    ASYNC_EXEC_DIR,
    ASYNC_FAST_THRESHOLD_S,
    ASYNC_INLINE_MAX_BYTES,
    NOHUP_TIMEOUT_MIN,
    _AsyncCompletion,
    _AsyncExecEntry,
    _async_add_inline,
    _async_executions,
    _async_exec_lock,
    _async_last_lines,
    _async_try_inline,
    _tool_context,
    async_completion_queue,
    enable_nohup,
    get_async_command_for_output_path,
    nohup_tool_specs,
    t_execute_async,
    t_nohup,
    t_query_exec,
    t_nohup_wait,
)

__all__ = [  # re-exports (mypy --no-implicit-reexport)
    "ASYNC_EXEC_DIR", "ASYNC_FAST_THRESHOLD_S", "ASYNC_INLINE_MAX_BYTES",
    "NOHUP_TIMEOUT_MIN", "_AsyncCompletion", "_AsyncExecEntry",
    "_async_add_inline", "_async_executions", "_async_exec_lock",
    "_async_last_lines", "_async_try_inline", "_tool_context",
    "async_completion_queue", "enable_nohup",
    "get_async_command_for_output_path", "nohup_tool_specs",
    "t_execute_async", "t_nohup", "t_query_exec", "t_nohup_wait",
    "DEFAULT_TOOL_TTL_S", "EXEC_SHELL_MARGIN_S",
]

# Optional reference to the active _InputCollector (set by _core REPL loop).
# When set, t_ask_user_question pauses it before calling input() so the
# background reader thread doesn't steal keystrokes.
class _InputCollectorProtocol(Protocol):
    def pause(self) -> None: ...
    def resume(self) -> None: ...

_input_collector: _InputCollectorProtocol | None = None


# Colour escapes needed for interactive user-facing prompts in t_ask_user*.
_BOLD = "\033[1m"
_YEL = "\033[33m"
_RED = "\033[31m"
_RESET = "\033[0m"
_RL_BOLD  = "\x01\033[1m\x02"
_RL_RESET = "\x01\033[0m\x02"

# Tracks the subprocess currently executing inside a tool, so the SIGINT
# handler in _core.py can SIGKILL it immediately on Ctrl-C.
_active_proc: "subprocess.Popen[str] | None" = None


# ── streamed tool output ──────────────────────────────────────────────────────
# Long-running tools echo their subprocess output live so a human watching the
# terminal sees progress.  A host that speaks a protocol on stdout — MCP, ACP,
# LSP — must send that stream elsewhere, or the echoed bytes land in the middle
# of a frame and corrupt the session.  set_tool_output_stream() moves it.

_tool_output_stream: "IO[str] | None" = None


def set_tool_output_stream(stream: "IO[str] | None") -> None:
    """Send live tool output to *stream* instead of stdout.

    Pass ``None`` to restore the default.  The default is resolved at write
    time, so a caller that reassigns ``sys.stdout`` is still honoured.

    >>> import sys
    >>> set_tool_output_stream(sys.stderr)   # stdout carries a protocol
    """
    global _tool_output_stream
    _tool_output_stream = stream


def get_tool_output_stream() -> "IO[str]":
    """Return the stream live tool output is currently written to."""
    return _tool_output_stream if _tool_output_stream is not None else sys.stdout


def _emit(text: str) -> None:
    """Write streamed tool output, never failing the tool if the sink is gone."""
    try:
        stream = get_tool_output_stream()
        stream.write(text)
        stream.flush()
    except Exception:
        pass


def t_read(path: str, offset: int | None = None, limit: int | None = None) -> tuple[str, dict[str, object]]:
    """Read and return the contents of a file at the specified path.

    Tool spec:
        name: read_file
        description: Read and return the contents of a file at the specified path.
        parameters:
            path:
                type: string
                description: Path to the file.
                required: true
            offset:
                type: integer
                description: Line number to start reading from (1-indexed).
            limit:
                type: integer
                description: Maximum number of lines to read.
    """
    # Rationale for XML envelope with checksum tag:
    #   The outermost XML tag is an 8-hex-char checksum (SHA256 prefix) of the
    #   returned content.  This lets the model self-consistently identify file
    #   versions across turns, and provides forensic traceability — given a
    #   tool-call log, an auditor can verify exactly what content was read,
    #   even if the file has since changed on disk.  When offset/limit is used
    #   the tag carries those attributes so a partial read is self-describing.
    import hashlib
    path = _coerce_str(path, "path")
    try:
        content = Path(os.path.expanduser(path)).read_text()
        lines = content.splitlines(keepends=True)
        # Default to partial read (first 100 lines) for files larger than 100 lines.
        # This is essential for token efficiency — reading an entire large file
        # can consume 10k+ tokens in a single tool call, blowing the budget and
        # pushing out other context.  The model can always request more via
        # offset/limit if it needs the rest.
        if offset is None and limit is None and len(lines) > 100:
            offset = 1
            limit = 100
        if offset is not None or limit is not None:
            start = (offset - 1) if offset is not None else 0
            start = max(0, start)
            if limit is not None:
                end = start + limit
            else:
                end = len(lines)
            content = "".join(lines[start:end])
        checksum = hashlib.sha256(content.encode()).hexdigest()[:8]
        attrs = ""
        if offset is not None:
            attrs += f' offset="{offset}"'
        if limit is not None:
            attrs += f' limit="{limit}"'
        xml = f"<{checksum}{attrs}>{content}</{checksum}>"
        return xml, {"result": xml}
    except Exception as e:
        return _tool_error("read_file", e)

def _coerce_str(value: object, name: str) -> str:
    """Best-effort str for model-supplied args (models sometimes send
    booleans/None where the schema says string)."""
    if isinstance(value, str):
        return value
    return str(value)


def _tool_failure(message: str) -> tuple[str, dict[str, object]]:
    """Uniform failure envelope: the message for the model, ``ok: False`` for code.

    The text a tool returns is written for the model, so failures are just an
    ``ERROR: …`` sentence.  A caller that has to *act* on the outcome — a
    protocol host mapping it to MCP's ``isError``, a hook, a dashboard — should
    read ``meta`` instead of pattern-matching English::

        text, meta = dispatch(name, args, tool_dispatch)
        if not meta.get("ok", True):
            ...
    """
    return message, {"result": message, "ok": False, "error": message}


def _tool_error(tool: str, exc: Exception) -> tuple[str, dict[str, object]]:
    """Uniform internal-error envelope: exception type + message + innermost
    frames, so the model (and the human) can locate tool bugs from the
    result alone instead of a bare ``ERROR: 'bool' object has no attribute
    'splitlines'``."""
    tb = traceback.extract_tb(exc.__traceback__)
    inner = ", ".join(
        f"{os.path.basename(f.filename)}:{f.lineno} in {f.name}" for f in tb[-3:]
    )
    r = f"ERROR: {tool} internal error: {type(exc).__name__}: {exc} ({inner})"
    return _tool_failure(r)


def t_write(path: str, content: str) -> tuple[str, dict[str, object]]:
    """Write (or overwrite) a file at the specified path with the given content.

    Tool spec:
        name: write_file
        description: Write (or overwrite) a file at the specified path with the given content.
        parameters:
            path:
                type: string
                description: Path to the file.
                required: true
            content:
                type: string
                description: Content to write.
                required: true
    """
    path = _coerce_str(path, "path")
    content = _coerce_str(content, "content")
    try:
        p = Path(os.path.expanduser(path))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        added = len(content.splitlines()) if content else 0
        r = f"OK: wrote {len(content)} bytes to {path}"
        return r, {
            "result": r,
            "files": [path],
            "diff_summary": {"path": path, "added": added, "removed": 0},
        }
    except Exception as e:
        return _tool_error("write_file", e)

def _apply_patch_format(patch: str) -> tuple[str, dict[str, object]]:
    """Handle OpenAI-style apply_patch format.

    Expected shape:
        *** Begin Patch
        *** Update File: /path/to/file
        @@
        -old line(s)
        +new line(s)
         context line(s)
        *** End Patch
    """
    lines = patch.splitlines()
    path: str | None = None
    for line in lines:
        if line.startswith("*** Update File:"):
            path = line.split(":", 1)[1].strip()
            break
    if not path:
        return _tool_failure("ERROR: apply_patch: could not find '*** Update File:' in patch")

    # Collect hunk lines after the @@ marker
    in_hunk = False
    old_lines: list[str] = []
    new_lines: list[str] = []
    for line in lines:
        if line.startswith("@@"):
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if line.startswith("*** "):
            break
        if line.startswith("-"):
            old_lines.append(line[1:])
        elif line.startswith("+"):
            new_lines.append(line[1:])
        else:
            # Context line — belongs to both sides
            old_lines.append(line[1:] if line.startswith(" ") else line)
            new_lines.append(line[1:] if line.startswith(" ") else line)

    old = "\n".join(old_lines)
    new = "\n".join(new_lines)
    return t_update(path=path, old=old, new=new)


def t_update(path: str = "", old: str = "", new: str = "", patch: str = "",
             replace_all: bool = False) -> tuple[str, dict[str, object]]:
    """Edit an existing file by replacing a specific substring.

    Tool spec:
        name: str_replace
        description: Edit an existing file by replacing a specific substring.
        parameters:
            path:
                type: string
                description: Path to the file.
                required: true
            old_str:
                type: string
                description: Text to replace.
                required: true
            new_str:
                type: string
                description: Replacement text.
                required: true
            replace_all:
                type: boolean
                description: Replace every occurrence instead of only the first.

    With *replace_all* False (default) only the first occurrence is replaced;
    with it True every occurrence is. Either way the exact byte sequence of
    *old_str* must be present in the file, else the edit is refused.
    """
    path = _coerce_str(path, "path")
    old = _coerce_str(old, "old_str")
    new = _coerce_str(new, "new_str")
    if patch:
        return _apply_patch_format(_coerce_str(patch, "patch"))
    try:
        p = Path(os.path.expanduser(path))
        text = p.read_text()
        if old not in text:
            return _tool_failure(
                f"ERROR: old string not found in {path} "
                f"({len(old)} chars, starts with {repr(old[:80])}). "
                f"Re-read the file and copy the exact bytes."
            )
        n = text.count(old)                      # total matches before the edit
        done = n if replace_all else min(1, n)
        p.write_text(text.replace(old, new) if replace_all else text.replace(old, new, 1))
        # Count lines and UTF-8 characters in the replaced text
        old_lines = (old.count('\n') + (1 if old else 0)) * done
        old_chars = len(old) * done
        new_lines = (new.count('\n') + (1 if new else 0)) * done
        r = f"OK: replaced {done} of {n} occurrence(s)"
        r += (f" ({old_lines} line(s), {old_chars} UTF-8 character(s)) in {path}"
              + ("" if done == n
                 else f"; {n - done} remaining — pass replace_all=True to replace them"))
        return r, {
            "result": r,
            "files": [path],
            "diff_summary": {"path": path, "added": new_lines, "removed": old_lines},
        }
    except Exception as e:
        return _tool_error("str_replace", e)

# Default time-to-live (seconds) for one synchronous tool execution when the
# agent spec does not configure one.  exec_shell's max time is kept a little
# below the TTL (EXEC_SHELL_MARGIN_S), so a timed-out command still leaves a
# margin to stream partial output back before the turn budget is exhausted.
DEFAULT_TOOL_TTL_S = 600
EXEC_SHELL_MARGIN_S = 20


def _exec_shell_timeout_s() -> int:
    """Max wall-clock time for one exec_shell call: TTL minus the margin."""
    try:
        ttl = getattr(_tool_context, "tool_ttl_seconds", None)
    except Exception:
        ttl = None
    if not isinstance(ttl, (int, float)) or ttl <= 0:
        ttl = DEFAULT_TOOL_TTL_S
    return max(1, int(ttl) - EXEC_SHELL_MARGIN_S)


def t_run(command: str) -> tuple[str, dict[str, object]]:
    """Execute a shell command and return its stdout, stderr, and exit code.

    Tool spec:
        name: exec_shell
        description: Execute a shell command and return its stdout, stderr, and exit code.
        parameters:
            command:
                type: string
                description: Shell command to execute.
                required: true
    """
    global _active_proc
    command = _coerce_str(command, "command")
    timeout_s = _exec_shell_timeout_s()
    proc: subprocess.Popen[str] | None = None
    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            executable="/bin/bash",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,
        )
        _active_proc = proc

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        def _drain(stream: Iterable[str], sink: list[str]) -> None:
            for line in stream:
                sink.append(line)
                _emit(line)

        t_out = threading.Thread(target=_drain, args=(proc.stdout, stdout_lines), daemon=True)
        t_err = threading.Thread(target=_drain, args=(proc.stderr, stderr_lines), daemon=True)
        t_out.start()
        t_err.start()

        try:
            proc.wait(timeout=timeout_s)
        except KeyboardInterrupt:
            # Signal handler already SIGKILLed the process; just wait briefly.
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except Exception:
                    pass
            t_out.join(timeout=2)
            t_err.join(timeout=2)
            raise

        t_out.join(timeout=5)
        t_err.join(timeout=5)

        out = "".join(stdout_lines)
        err = "".join(stderr_lines)
        result = json.dumps({
            "stdout": out,
            "stderr": err,
            "returncode": proc.returncode,
        }, separators=(",", ":"))
        return result, {
            "stdout": out, "stderr": err, "returncode": proc.returncode,
            "streamed": True,
            "result": result,
        }
    except subprocess.TimeoutExpired:
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                proc.kill()
        # Give drain threads a moment to capture what was buffered.
        t_out.join(timeout=2)
        t_err.join(timeout=2)
        out = "".join(stdout_lines)
        err = "".join(stderr_lines)
        _exec_async_available = any(
            e.get("python_function") == "t_execute_async"
            for e in getattr(_tool_context, "tool_dispatch", {}).values()
        )
        hint = (
            f"The command did not finish within {timeout_s} seconds. "
            "For long-running commands, use 'nohup <command> &' to run in the "
            "background."
        )
        if _exec_async_available:
            hint += " Or use t_execute_async to start the command asynchronously."
        result = json.dumps({
            "error": f"command timed out after {timeout_s} s",
            "stdout": out,
            "stderr": err,
            "hint": hint,
        }, separators=(",", ":"))
        return result, {"error": result, "result": result, "stdout": out, "stderr": err, "hint": hint}
    except Exception as e:
        result = json.dumps({"error": str(e)}, separators=(",", ":"))
        return result, {"error": result, "result": result}
    finally:
        _active_proc = None

def t_ask_user(question: str) -> tuple[str, dict[str, object]]:
    """Prompt the user interactively and return their answer."""
    print(f"\n{_YEL}{_BOLD}? {question}{_RESET}")
    try:
        answer = input(f"{_RL_BOLD}Your answer:{_RL_RESET} ").strip()
    except (EOFError, KeyboardInterrupt):
        answer = ""
        print()
    r = json.dumps({"answer": answer})
    return r, {"result": r}

def _play_ask_sound() -> None:
    """Fire-and-forget notification sound when the agent needs user input."""
    import platform
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.Popen(
                ["afplay", "/System/Library/Sounds/Glass.aiff"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        elif system == "Linux":
            subprocess.Popen(
                ["paplay", "/usr/share/sounds/freedesktop/stereo/message.oga"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        elif system == "Windows":
            import winsound
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)  # type: ignore[attr-defined]
        else:
            print("\a", end="", flush=True)
    except Exception:
        print("\a", end="", flush=True)


def t_ask_user_question(question: str = '', options: str = '') -> tuple[str, dict[str, object]]:
    """Prompt the user with an optional numbered list of choices."""
    if not question:
        return _tool_failure('ERROR: No question provided')

    parsed_options: list[object] = []
    if options:
        if isinstance(options, list):
            parsed_options = options
        else:
            try:
                parsed_options = json.loads(options)
                if not isinstance(parsed_options, list):
                    parsed_options = [str(parsed_options)]
            except (json.JSONDecodeError, TypeError):
                parsed_options = [opt.strip() for opt in options.split(',') if opt.strip()]

    _play_ask_sound()
    print(f"\n{_YEL}{_BOLD}? {question}{_RESET}")
    if parsed_options:
        for i, opt in enumerate(parsed_options, 1):
            print(f"  {i}. {opt}")

    # Pause the background _InputCollector so it doesn't steal stdin.
    collector = _input_collector
    if collector is not None:
        collector.pause()
    try:
        answer = input(f"{_RL_BOLD}Your answer:{_RL_RESET} ").strip()
    except (EOFError, KeyboardInterrupt):
        return _tool_failure('ERROR: No user input available')
    finally:
        if collector is not None:
            collector.resume()

    if parsed_options and answer.isdigit():
        idx = int(answer)
        if 1 <= idx <= len(parsed_options):
            answer = str(parsed_options[idx - 1])

    r = json.dumps({"answer": answer})
    return r, {'result': answer, 'question': question, 'options': parsed_options}


def t_list_dir(path: str) -> tuple[str, dict[str, object]]:
    """List a directory, one entry per line, prefixed by d (dir) or f (file).

    Tool spec:
        name: list_dir
        description: List the entries of a directory, one per line, prefixed by d (directory) or f (file).
        parameters:
            path:
                type: string
                description: Directory to list.
                required: true
    """
    path = _coerce_str(path, "path")
    try:
        entries = sorted(Path(os.path.expanduser(path)).iterdir(), key=lambda p: (p.is_file(), p.name))
        lines = [("d  " if e.is_dir() else "f  ") + e.name for e in entries]
        result = "\n".join(lines) or "(empty)"
        return result, {"result": result}
    except Exception as e:
        return _tool_error("list_dir", e)

def t_search(path: str = ".", pattern: str = "") -> tuple[str, dict[str, object]]:
    """Grep *path* for *pattern*, returning the matching lines as JSON.

    Tool spec:
        name: search_files
        description: Search file contents for a regular expression and return the matching lines as JSON.
        parameters:
            pattern:
                type: string
                description: Regular expression to search for.
                required: true
            path:
                type: string
                description: File or directory to search in (default '.').
    """
    global _active_proc
    path = _coerce_str(path, "path")
    pattern = _coerce_str(pattern, "pattern")
    proc: subprocess.Popen[str] | None = None
    try:
        proc = subprocess.Popen(
            # -H: without it grep omits the filename when *path* is a single
            # file, and the "file:line:text" parser below then discards every
            # match — searching one file silently returned no results.
            ["grep", "-r", "-H", "-n", "--", pattern, path],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,
        )
        _active_proc = proc

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        def _drain(stream: Iterable[str], sink: list[str]) -> None:
            for line in stream:
                sink.append(line)
                _emit(line)

        t_out = threading.Thread(target=_drain, args=(proc.stdout, stdout_lines), daemon=True)
        t_err = threading.Thread(target=_drain, args=(proc.stderr, stderr_lines), daemon=True)
        t_out.start()
        t_err.start()

        try:
            proc.wait(timeout=30)
        except KeyboardInterrupt:
            # Signal handler already SIGKILLed the process; just wait briefly.
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except Exception:
                    pass
            t_out.join(timeout=2)
            t_err.join(timeout=2)
            raise

        t_out.join(timeout=5)
        t_err.join(timeout=5)

        out = "".join(stdout_lines)
        matches: list[dict[str, object]] = []
        for line in out.splitlines():
            # grep -n output:  file:line:text
            parts = line.split(":", 2)
            if len(parts) >= 3:
                try:
                    matches.append({
                        "file": parts[0],
                        "line": int(parts[1]),
                        "text": parts[2],
                    })
                except ValueError:
                    pass

        result = json.dumps({"matches": matches}, separators=(",", ":"))
        return result, {"result": result, "streamed": True, "matches": matches}
    except subprocess.TimeoutExpired:
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                proc.kill()
        result = json.dumps({"error": "search timed out after 30 s"}, separators=(",", ":"))
        return result, {"result": result, "ok": False, "error": "search timed out after 30 s"}
    except Exception as e:
        result = json.dumps({"error": str(e)}, separators=(",", ":"))
        return result, {"result": result, "ok": False, "error": str(e)}
    finally:
        _active_proc = None

def t_glob(pattern: str) -> tuple[str, dict[str, object]]:
    """Return the paths matching a glob pattern, one per line.

    Tool spec:
        name: glob
        description: Return the paths matching a glob pattern, one per line.
        parameters:
            pattern:
                type: string
                description: Glob pattern, e.g. 'src/**/*.py'.
                required: true
    """
    import glob as _glob
    pattern = _coerce_str(pattern, "pattern")
    try:
        matches = sorted(_glob.glob(pattern, recursive=True))
        result = "\n".join(matches) or "(no matches)"
        return result, {"result": result, "matches": matches}
    except Exception as e:
        return _tool_error("glob", e)


# Functions that interactively ask the user something — excluded in --non-interactive mode.
_ASK_USER_FNS = {"t_ask_user", "t_ask_user_question"}

# Registry: function name (str) → callable.
if TYPE_CHECKING:
    ToolFn = Callable[..., tuple[str, dict[str, object]]]
else:
    ToolFn: Any

TOOL_LIBRARY: "dict[str, ToolFn]" = {
    "t_read":               t_read,
    "t_write":              t_write,
    "t_update":             t_update,
    "t_run":                t_run,
    "t_ask_user":           t_ask_user,
    "t_ask_user_question":  t_ask_user_question,
    "t_list_dir":           t_list_dir,
    "t_search":             t_search,
    "t_glob":               t_glob,
    "t_execute_async":      t_execute_async,
    "t_query_exec":         t_query_exec,
    "t_nohup":              t_nohup,
    "t_nohup_wait":           t_nohup_wait,
}

def enable_rtk_rewrite() -> None:
    """Patch TOOL_LIBRARY so shell commands are rewritten through rtk before execution.

    rtk (https://github.com/rtk-ai/rtk) is a CLI proxy that filters command
    output for 60-90% token savings. This function is a no-op when rtk is not
    in PATH. Call it once before agentknit.main() to opt in; it is off by default.
    """
    import shutil
    if not shutil.which("rtk"):
        return

    import subprocess as _sp

    def _rewrite(command: str) -> str:
        try:
            r = _sp.run(["rtk", "rewrite", command], capture_output=True, text=True, timeout=2)
            if r.returncode in (0, 3) and r.stdout.strip():
                return r.stdout.strip()
        except Exception:
            pass
        return command

    _orig_t_run = TOOL_LIBRARY["t_run"]
    def _rtk_t_run(command: str, **kw: Any) -> tuple[str, dict[str, object]]:
        return _orig_t_run(_rewrite(command), **kw)
    TOOL_LIBRARY["t_run"] = _rtk_t_run

    _orig_t_execute_async = TOOL_LIBRARY["t_execute_async"]
    def _rtk_t_execute_async(command: str, **kw: Any) -> tuple[str, dict[str, object]]:
        return _orig_t_execute_async(_rewrite(command), **kw)
    TOOL_LIBRARY["t_execute_async"] = _rtk_t_execute_async


def _register_generated(fn_name: str, source: str) -> bool:
    """Exec *source* and add the resulting callable to TOOL_LIBRARY.

    Returns True on success, False if compilation/exec fails.
    """
    ns: dict[str, object] = {}
    try:
        exec(compile(source, "<generated>", "exec"), ns)  # noqa: S102
    except Exception as e:
        print(f"{_RED}[codegen] compile error for {fn_name}: {e}{_RESET}", file=sys.stderr)
        return False
    fn = ns.get(fn_name)
    if not callable(fn):
        print(f"{_RED}[codegen] {fn_name} not found after exec{_RESET}", file=sys.stderr)
        return False
    TOOL_LIBRARY[fn_name] = fn
    return True


# --- generated: t_update_file ---
def t_update_file(new_str: str = '', file_path: str = '', old_str: str = '') -> tuple[str, dict[str, object]]:
    result_dict: dict[str, object] = {'result': 'success'}
    if not file_path:
        return _tool_failure("ERROR: File path is required.")
    try:
        with Path(os.path.expanduser(file_path)).open('r') as f:
            content = f.read()
        new_content = content.replace(old_str, new_str)
        with Path(os.path.expanduser(file_path)).open('w') as f:
            f.write(new_content)
        return ("File updated successfully.", result_dict)
    except Exception as e:
        return _tool_failure(f"ERROR: {str(e)}")

TOOL_LIBRARY['t_update_file'] = t_update_file


# --- generated: t_list_directory ---
def t_list_directory(path: str = '') -> tuple[str, dict[str, object]]:
    try:
        p = Path(os.path.expanduser(path))
        if not p.exists() or not p.is_dir():
            return _tool_failure("ERROR: Path does not exist or is not a directory")
        items = [str(item.name) for item in p.iterdir()]
        result_dict: dict[str, object] = {
            'result': 'success',
            'files': items
        }
        return ("Directory listing successful", result_dict)
    except Exception as e:
        return _tool_failure(f"ERROR: {str(e)}")

TOOL_LIBRARY['t_list_directory'] = t_list_directory


# --- generated: t_search_files ---
def t_search_files(command: str = '') -> tuple[str, dict[str, object]]:
    try:
        if not command:
            return _tool_failure("ERROR: command is required")
        
        glob_chars = {'*', '?', '['}
        has_glob = any(c in command for c in glob_chars)
        
        if not has_glob:
            p = Path(os.path.expanduser(command))
            if p.is_dir():
                matches = sorted([str(x) for x in p.iterdir()])
                if matches:
                    return "\n".join(matches), {"result": matches}
                else:
                    return f"Directory '{command}' is empty", {"result": []}
            elif p.exists():
                return str(p), {"result": [str(p)]}
            else:
                return f"No file or directory found: '{command}'", {"result": []}
        
        matches = sorted([str(x) for x in Path('.').glob(command)])
        if not matches:
            return f"No files found for '{command}'", {"result": []}
        
        limit = 100
        if len(matches) > limit:
            human = "\n".join(matches[:limit]) + f"\n... and {len(matches) - limit} more"
        else:
            human = "\n".join(matches)
        
        return human, {"result": matches}
    except Exception as e:
        return _tool_error("t_search_files", e)

TOOL_LIBRARY['t_search_files'] = t_search_files


# --- generated: t_find_files ---
def t_find_files(pattern: str = '', recursive: str = '') -> tuple[str, dict[str, object]]:
    try:
        rec = str(recursive).strip().lower() in ('true', '1', 'yes', 'on')
        if not pattern:
            pattern = '*'
        base = Path.cwd()
        if rec:
            matches = sorted([p for p in base.rglob(pattern) if p.is_file()])
        else:
            matches = sorted([p for p in base.glob(pattern) if p.is_file()])
        files = [str(p) for p in matches]
        if not files:
            human = "No files found."
        else:
            human = f"Found {len(files)} file(s):\n" + "\n".join(files)
        return (human, {"result": files})
    except Exception as e:
        return _tool_error("t_find_files", e)

TOOL_LIBRARY['t_find_files'] = t_find_files
