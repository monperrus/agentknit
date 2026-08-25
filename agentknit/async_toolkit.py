"""Asynchronous background shell tools: definitions and implementations.

Two layers live here:

* Low-level primitives ``t_execute_async`` / ``t_query_exec`` — start a shell
  command in the background (stdout/stderr captured to files, stdin exposed as
  a FIFO) and poll it by ``tool_exec_id``.
* The model-facing async tool trio — async tools always come with three:
  ``nohup`` (start, bounded by ``timeout(1)``), ``nohup_query`` (poll by
  ``tool_exec_id``) and ``wait_for`` (block for *howmuch* × *unit* and report
  everything that finished meanwhile), each with ready-made JSON tool specs.
  :func:`enable_nohup` wires all three into a spec schema in one call:

  >>> from agentknit.async_toolkit import enable_nohup
  >>> schema = agentknit.load_specification(MODEL, ENDPOINT)
  >>> enable_nohup(schema)          # adds nohup + nohup_query + wait_for
"""

from __future__ import annotations

import json
import os
import queue as _queue
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing import BinaryIO, TypedDict
else:
    from typing import BinaryIO, TypedDict


# ── execution state ───────────────────────────────────────────────────────────

# Persistent directory for stdout/stderr capture files (never deleted).
ASYNC_EXEC_DIR = Path.home() / ".cache" / "async_agent_execs"
ASYNC_EXEC_DIR.mkdir(parents=True, exist_ok=True)

# Inline output in the tool response when the command finishes this quickly …
ASYNC_FAST_THRESHOLD_S = 0.100
# … and both stdout and stderr are under this many bytes.
ASYNC_INLINE_MAX_BYTES = 4096

# Default execution bound (minutes) applied by t_nohup via timeout(1).
NOHUP_TIMEOUT_MIN = 10

# exec_id → {"proc": Popen, "stdout_file": str, "stderr_file": str, "start": float}
class _AsyncExecEntry(TypedDict):
    proc: subprocess.Popen[bytes]
    stdout_file: str
    stderr_file: str
    stdin_file: str
    start: float
    started_at: str
    cwd: str
    command: str


_async_executions: dict[str, _AsyncExecEntry] = {}
_async_exec_lock = threading.Lock()

# Completed processes push here so the REPL can trigger a new LLM turn.
# Each entry: {"tool_exec_id", "returncode", "stdout_file", "stderr_file", "duration"}
class _AsyncCompletion(TypedDict):
    tool_exec_id: str
    returncode: int | None
    stdout_file: str
    stderr_file: str
    duration: float
    cwd: str


async_completion_queue: "_queue.Queue[_AsyncCompletion]" = _queue.Queue()

# Thread-local set by _core._handle_tool_call before each dispatch so tools
# can access the current session without being passed the session dict.
# Defined here (not tool_library) so async_toolkit has no import cycle; it is
# re-exported from tool_library for backward compatibility.
_tool_context = threading.local()


def get_async_command_for_output_path(path: str) -> str | None:
    """Return the originating async shell command for a stdout/stderr file."""
    expanded = os.path.expanduser(path)
    with _async_exec_lock:
        for entry in _async_executions.values():
            if expanded in {entry["stdout_file"], entry["stderr_file"]}:
                return entry["command"]
    return None


def _async_try_inline(path: str) -> str | None:
    """Return file text if it fits within ASYNC_INLINE_MAX_BYTES, else None."""
    try:
        p = Path(path)
        if p.stat().st_size > ASYNC_INLINE_MAX_BYTES:
            return None
        return p.read_text(errors="replace")
    except OSError:
        return None


def _async_last_lines(path: str, n: int = 3) -> str:
    """Return the last *n* lines of *path*, or empty string if unreadable."""
    try:
        lines = Path(path).read_text(errors="replace").splitlines()
        return "\n".join(lines[-n:]) if lines else ""
    except OSError:
        return ""


def _async_add_inline(result: dict[str, object], stdout_path: str, stderr_path: str) -> None:
    """Append stdout/stderr content to *result* when both files are small enough."""
    out = _async_try_inline(stdout_path)
    err = _async_try_inline(stderr_path)
    if out is not None:
        result["stdout"] = out
    if err is not None:
        result["stderr"] = err


# ── low-level tools ───────────────────────────────────────────────────────────

def t_execute_async(command: str, when: int = 0) -> tuple[str, dict[str, object]]:
    """Start a shell command asynchronously, capturing stdout/stderr to files.

    *when* (minutes, default 0) delays the start; use it to schedule a command
    for later without a separate planning tool.

    A named FIFO is created at stdin_localfile; write text to it to send input
    to the running process (e.g. via write_file or a shell redirect).

    If the command finishes within ASYNC_FAST_THRESHOLD_S *and* both output
    files are small, the content is inlined so the caller needs no follow-up
    t_query_exec call.
    """
    if when:
        time.sleep(when * 60)
    session_id = getattr(_tool_context, "session_id", None)
    exec_dir = (ASYNC_EXEC_DIR / session_id) if session_id else ASYNC_EXEC_DIR
    exec_dir.mkdir(parents=True, exist_ok=True)
    exec_id = uuid.uuid4().hex[:12]
    stdout_path = str(exec_dir / f"{exec_id}.stdout")
    stderr_path = str(exec_dir / f"{exec_id}.stderr")
    stdin_path  = str(exec_dir / f"{exec_id}.stdin")
    cwd = os.getcwd()

    os.mkfifo(stdin_path)
    stdout_fh = open(stdout_path, "wb", buffering=0)
    stderr_fh = open(stderr_path, "wb", buffering=0)

    # Open the FIFO write-end in a background thread (open() on a FIFO blocks
    # until a reader appears). The read-end is handed to the process.
    stdin_write_fh: "list[BinaryIO]" = []   # populated by the thread once the process opens it

    def _open_fifo_write() -> None:
        fh = open(stdin_path, "wb", buffering=0)
        stdin_write_fh.append(fh)

    fifo_thread = threading.Thread(target=_open_fifo_write, daemon=True)
    fifo_thread.start()

    stdin_read_fh = open(stdin_path, "rb")   # unblocks the writer thread

    t0 = time.monotonic()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    proc = subprocess.Popen(
        command,
        shell=True,
        stdin=stdin_read_fh,
        stdout=stdout_fh,
        stderr=stderr_fh,
        preexec_fn=os.setsid,
    )
    stdin_read_fh.close()   # process has inherited the fd; we don't need it

    try:
        proc.wait(timeout=ASYNC_FAST_THRESHOLD_S)
    except subprocess.TimeoutExpired:
        pass

    stdout_fh.flush()
    stderr_fh.flush()
    returncode = proc.poll()
    fast_done = returncode is not None

    def _close_on_exit() -> None:
        proc.wait()
        stdout_fh.close()
        stderr_fh.close()
        fifo_thread.join(timeout=1)
        for fh in stdin_write_fh:
            try:
                fh.close()
            except OSError:
                pass
        async_completion_queue.put({
            "tool_exec_id": exec_id,
            "returncode":   proc.returncode,
            "stdout_file":  stdout_path,
            "stderr_file":  stderr_path,
            "duration":     round(time.monotonic() - t0, 3),
            "cwd":          cwd,
        })

    if fast_done:
        # Result already inlined; close handles but skip the completion queue push.
        stdout_fh.close()
        stderr_fh.close()
        fifo_thread.join(timeout=1)
        for fh in stdin_write_fh:
            try:
                fh.close()
            except OSError:
                pass
    else:
        threading.Thread(target=_close_on_exit, daemon=True).start()

    duration = round(time.monotonic() - t0, 3)

    with _async_exec_lock:
        _async_executions[exec_id] = {
            "proc": proc,
            "command": command,
            "stdout_file": stdout_path,
            "stderr_file": stderr_path,
            "stdin_file":  stdin_path,
            "start": t0,
            "started_at": started_at,
            "cwd": cwd,
        }

    result: dict[str, object] = {
        "tool_exec_id": exec_id,
        "started_at":       started_at,
        "cwd":              cwd,
        "stdin_localfile":  stdin_path,
        "stdout_localfile": stdout_path,
        "stderr_localfile": stderr_path,
    }
    if fast_done:
        result["completed"] = True
        result["returncode"] = returncode
        result["duration_time"] = duration
        _async_add_inline(result, stdout_path, stderr_path)

    r = json.dumps(result)
    return r, {"result": r}


def t_query_exec(tool_exec_id: str) -> tuple[str, dict[str, object]]:
    """Poll the status of a command started with t_execute_async.

    When completed, returncode is included and stdout/stderr are inlined if
    both are under ASYNC_INLINE_MAX_BYTES.
    """
    with _async_exec_lock:
        entry = _async_executions.get(tool_exec_id)

    if entry is None:
        r = json.dumps({"error": f"unknown tool_exec_id: {tool_exec_id}"})
        return r, {"result": r}

    proc = entry["proc"]
    returncode = proc.poll()
    completed = returncode is not None
    duration = round(time.monotonic() - entry["start"], 3)

    stdout_size = stderr_size = 0
    try:
        stdout_size = Path(entry["stdout_file"]).stat().st_size
    except OSError:
        pass
    try:
        stderr_size = Path(entry["stderr_file"]).stat().st_size
    except OSError:
        pass

    result: dict[str, object] = {
        "completed": completed,
        "returncode": returncode,  # None while running, int when done
        "started_at": entry["started_at"],
        "cwd":         entry.get("cwd", ""),
        "duration_time": duration,
        "stdin_localfile": entry["stdin_file"],
        "stdout_localfile_size": stdout_size,
        "stderr_localfile_localsize": stderr_size,
    }
    if completed:
        _async_add_inline(result, entry["stdout_file"], entry["stderr_file"])

    r = json.dumps(result)
    return r, {"result": r}


# ── nohup / nohup_query / wait_for ───────────────────────────────────────────────────────

def t_nohup(command: str, timeout: int = NOHUP_TIMEOUT_MIN) -> tuple[str, dict[str, object]]:
    """Bound the command with timeout(1) then hand off to t_execute_async."""
    return t_execute_async(f"timeout {int(timeout) * 60} {command}")


# Units understood by t_wait_for.  Wait durations are computed as
# howmuch * WAIT_FOR_UNIT_SECONDS[unit]; unknown units are rejected.
WAIT_FOR_UNIT_SECONDS = {
    "s": 1,
    "m": 60,
    "h": 3600,
    "d": 86400,
}
# Cap so a typo in howmuch cannot block the tool thread for hours on end.
WAIT_FOR_MAX_SECONDS = 3600


def t_wait_for(howmuch: int, unit: str = "s") -> tuple[str, dict[str, object]]:
    """Wait *howmuch* × *unit*, then report every execution finished meanwhile.

    Async tools always come with three: ``nohup`` starts a command in the
    background, ``nohup_query`` polls one by ``tool_exec_id``, and
    ``wait_for`` sleeps so background work can finish instead of busy-polling
    with ``nohup_query``.  Completions queued while waiting (returncode,
    output files, last output lines) are reported inline, so the model gets
    results without an extra round trip.

    Supported units: ``s`` seconds, ``m`` minutes, ``h`` hours, ``d`` days.
    """
    factor = WAIT_FOR_UNIT_SECONDS.get(unit)
    if factor is None:
        expected = "/".join(sorted(WAIT_FOR_UNIT_SECONDS))
        r = json.dumps({"error": f"unknown unit {unit!r}, expected one of {expected}"})
        return r, {"result": r}
    seconds = float(howmuch) * factor
    if seconds <= 0:
        r = json.dumps({"error": "howmuch must be a positive number"})
        return r, {"result": r}
    if seconds > WAIT_FOR_MAX_SECONDS:
        r = json.dumps({"error": f"wait of {seconds:g}s exceeds the {WAIT_FOR_MAX_SECONDS}s cap"})
        return r, {"result": r}

    time.sleep(seconds)

    completions: list[dict[str, object]] = []
    while True:
        try:
            c = async_completion_queue.get_nowait()
        except _queue.Empty:
            break
        entry = _async_executions.get(c["tool_exec_id"])
        out: dict[str, object] = {
            "tool_exec_id":      c["tool_exec_id"],
            "returncode":        c["returncode"],
            "duration_time":     c["duration"],
            "stdout_localfile":  c["stdout_file"],
            "stderr_localfile":  c["stderr_file"],
        }
        if entry is not None:
            out["command"] = entry["command"]
            out["stdout_last_lines"] = _async_last_lines(entry["stdout_file"])
            out["stderr_last_lines"] = _async_last_lines(entry["stderr_file"])
        completions.append(out)

    r = json.dumps({
        "waited_seconds": round(seconds, 3),
        "completed":      completions,
    })
    return r, {"result": r}


def nohup_tool_specs(timeout_min: int = NOHUP_TIMEOUT_MIN) -> list[dict[str, Any]]:
    """JSON tool specs for the nohup / nohup_query / wait_for trio."""
    return [
        {
            "type": "function",
            "function": {
                "name": "nohup",
                "description": (
                    "Start a shell command asynchronously, like nohup(1). Returns "
                    "tool_exec_id and local file paths for stdin (FIFO), stdout, "
                    "and stderr. Write to stdin_localfile to send input to the "
                    "running process. Execution is bounded: the command is killed "
                    f"after `timeout` minutes (default {timeout_min}). If "
                    f"the command finishes within {int(ASYNC_FAST_THRESHOLD_S * 1000)} ms "
                    f"and both outputs are under {ASYNC_INLINE_MAX_BYTES} bytes, "
                    "stdout/stderr are inlined immediately."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to run."},
                        "timeout": {
                            "type": "integer",
                            "description": (
                                "Maximum minutes the command may run before being "
                                f"killed (default {timeout_min})."
                            ),
                        },
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "nohup_query",
                "description": (
                    "Poll a command started with nohup. When completed, includes "
                    "returncode and inlines stdout/stderr if both are under "
                    f"{ASYNC_INLINE_MAX_BYTES} bytes; otherwise reports file sizes."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tool_exec_id": {"type": "string", "description": "The tool_exec_id returned by nohup."},
                    },
                    "required": ["tool_exec_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "wait_for",
                "description": (
                    "Wait howmuch * unit for background commands started with nohup "
                    "to finish, instead of busy-polling with nohup_query. Reports "
                    "every execution that completed while waiting: returncode, "
                    "output file paths, and the last lines of stdout/stderr. Use "
                    "nohup_query afterwards for executions still running."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "howmuch": {"type": "integer", "description": "How long to wait, in the given unit."},
                        "unit": {
                            "type": "string",
                            "enum": sorted(WAIT_FOR_UNIT_SECONDS),
                            "description": (
                                "Time unit for howmuch: s=seconds, m=minutes, h=hours, "
                                f"d=days. Waits above {WAIT_FOR_MAX_SECONDS}s are rejected; "
                                "split long waits into several calls."
                            ),
                        },
                    },
                    "required": ["howmuch"],
                },
            },
        },
    ]


def enable_nohup(schema: dict[str, Any], timeout_min: int = NOHUP_TIMEOUT_MIN) -> dict[str, Any]:
    """Add the nohup / nohup_query / wait_for tools to *schema* in place.

    Appends the tool specs (both ``tool_specs`` and ``inferred_tool_schema``)
    and wires dispatch to the ``t_nohup`` / ``t_query_exec`` / ``t_wait_for``
    functions already registered in TOOL_LIBRARY.  Idempotent: calling it
    twice is a no-op.

    Supports both schema shapes, like the wrappers did inline before:
    ``tools`` (list of TOOL_LIBRARY function names) or a pre-built
    ``tool_dispatch`` dict.
    """
    tool_specs = list(schema.get("tool_specs") or schema.get("inferred_tool_schema") or [])
    names = {((t.get("function") or t).get("name") if isinstance(t, dict) else None)
             for t in tool_specs}
    if "nohup" not in names:
        tool_specs.extend(nohup_tool_specs(timeout_min))
        schema["tool_specs"] = tool_specs
        schema["inferred_tool_schema"] = tool_specs
        if "tools" in schema:
            schema["tools"] = list(schema["tools"]) + ["t_nohup", "t_query_exec", "t_wait_for"]
        else:
            schema.setdefault("tool_dispatch", {})
            schema["tool_dispatch"].update({
                "nohup":       {"python_function": "t_nohup",      "param_map": {}},
                "nohup_query": {"python_function": "t_query_exec", "param_map": {}},
                "wait_for":    {"python_function": "t_wait_for",   "param_map": {}},
            })
    return schema
