"""Asynchronous background shell tools: definitions and implementations.

Two layers live here:

* Low-level primitives ``t_execute_async`` / ``t_query_exec`` — start a shell
  command in the background (stdout/stderr captured to files, stdin exposed as
  a FIFO) and poll it by ``tool_exec_id``.
* The model-facing async tool trio — async tools always come with three:
  ``nohup`` (start, bounded by ``timeout(1)``), ``nohup_query`` (poll by
  ``tool_exec_id``) and ``nohup_wait`` (block on one ``tool_exec_id`` until it
  finishes or *howmuch* × *unit* elapses, reporting CPU/I/O activity when it
  is still running), each with ready-made JSON tool specs.
  :func:`enable_nohup` wires all three into a spec schema in one call:

  >>> from agentknit.async_toolkit import enable_nohup
  >>> schema = agentknit.load_specification(MODEL, ENDPOINT)
  >>> enable_nohup(schema)          # adds nohup + nohup_query + nohup_wait
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

# Cap on wait_before_s (delayed start), same order of magnitude as the
# nohup_wait budget cap: enough to let CI run, not enough to schedule tomorrow.
WAIT_BEFORE_MAX_SECONDS = 3600

# exec_id → {"proc": Popen (None until a wait_before_s delay elapses), …}
class _AsyncExecEntry(TypedDict):
    proc: "subprocess.Popen[bytes] | None"
    stdout_file: str
    stderr_file: str
    stdin_file: str
    start: float
    started_at: str
    cwd: str
    command: str
    io_before: dict[str, int]          # last /proc/<pid>/io counters read for this exec
    scheduled_for: float               # monotonic time at which the command starts
                                        # (== start when there is no wait_before_s delay)
    fast_done: bool                    # finished within ASYNC_FAST_THRESHOLD_S (sync path only)


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

# tool_exec_id returned by the last t_query_exec call. A second query for the
# same still-running execution is answered with a nohup_wait redirect, so the
# model sleeps instead of busy-polling. Reset whenever a new execution starts.
_last_queried_exec_id: str | None = None

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


# ── CPU / I/O activity of a running execution ─────────────────────────────────

# clock ticks per second, used to scale utime/stime from /proc/<pid>/stat.
_CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100

# /proc/<pid>/io fields reported for I/O activity (bytes read/written).
_IO_FIELDS = ("rchar", "wchar", "read_bytes", "write_bytes")


def _process_tree(pid: int) -> list[int]:
    """PIDs of *pid* and all its descendants, alive at snapshot time."""
    pids = [pid]
    # /proc/<pid>/task/<tid>/children lists direct child pids, one per line entry.
    try:
        for tid_dir in Path(f"/proc/{pid}/task").glob("*"):
            try:
                pids.extend(int(x) for x in (tid_dir / "children").read_text().split())
            except (OSError, ValueError):
                pass
    except OSError:
        pass
    return pids


def _read_proc_io(pid: int) -> dict[str, int]:
    """Aggregate /proc/<pid>/io counters, zeroed when unavailable (non-Linux)."""
    io = dict.fromkeys(_IO_FIELDS, 0)
    try:
        for line in Path(f"/proc/{pid}/io").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in io:
                io[key] = int(value.strip() or 0)
    except (OSError, ValueError):
        pass
    return io


def _proc_cpu_seconds(pid: int) -> float:
    """User+system CPU seconds of *pid* from /proc/<pid>/stat (0 if unavailable)."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        # comm may contain spaces: fields start after the last ')'.
        fields = stat[stat.rindex(")") + 2:].split()
        return (int(fields[11]) + int(fields[12])) / _CLK_TCK   # utime + stime
    except (OSError, ValueError, IndexError):
        return 0.0


def _activity_snapshot(entry: _AsyncExecEntry) -> tuple[float, dict[str, int]]:
    """CPU seconds and I/O counters of *entry*'s process tree (0 on non-Linux)."""
    if entry["proc"] is None:
        return 0.0, dict.fromkeys(_IO_FIELDS, 0)
    try:
        pids = _process_tree(entry["proc"].pid)
    except OSError:
        pids = [entry["proc"].pid]
    cpu = sum(_proc_cpu_seconds(pid) for pid in pids)
    io: dict[str, int] = dict.fromkeys(_IO_FIELDS, 0)
    for pid in pids:
        for key, value in _read_proc_io(pid).items():
            io[key] += value
    return cpu, io


def _activity_report(entry: _AsyncExecEntry) -> dict[str, Any]:
    """CPU and I/O consumed by *entry* since the previous call (or since start).

    ``nohup_wait`` calls this once when it has to report a still-running
    execution, so the model can tell an active process from a hung or idle
    one: a flat CPU delta and flat I/O counters mean nothing is happening.
    """
    cpu, io = _activity_snapshot(entry)
    before = entry.get("io_before") or dict.fromkeys(_IO_FIELDS, 0)
    entry["io_before"] = io
    elapsed = max(time.monotonic() - entry["start"], 1e-9)
    return {
        "cpu_seconds": round(cpu, 3),
        "cpu_percent": round(100.0 * cpu / elapsed, 1),
        "io_bytes": {k: io[k] - before[k] for k in _IO_FIELDS},
        "note": ("command not started yet (wait_before_s delay)"
                 if entry["proc"] is None
                 else "bytes read/written since the last nohup_wait report of this execution"),
    }


# ── low-level tools ───────────────────────────────────────────────────────────

def t_execute_async(command: str, wait_before_s: float = 0) -> tuple[str, dict[str, object]]:
    """Start a shell command asynchronously, capturing stdout/stderr to files.

    *wait_before_s* (seconds, default 0) delays the start of the command: the
    execution is registered immediately (so the caller gets its
    ``tool_exec_id`` and file paths right away) and a background thread runs
    the command once the delay has elapsed. Use it to schedule a command for
    later; the delay does not count towards any timeout(1) bound on the
    command.

    A named FIFO is created at stdin_localfile; write text to it to send input
    to the running process (e.g. via write_file or a shell redirect). The FIFO
    is only connected once the command actually starts.

    If the command finishes within ASYNC_FAST_THRESHOLD_S *and* both output
    files are small, the content is inlined so the caller needs no follow-up
    t_query_exec call.
    """
    if wait_before_s < 0:
        r = json.dumps({"error": "wait_before_s must be >= 0"})
        return r, {"result": r}
    if wait_before_s > WAIT_BEFORE_MAX_SECONDS:
        r = json.dumps({"error": f"wait_before_s exceeds the {WAIT_BEFORE_MAX_SECONDS}s cap"})
        return r, {"result": r}
    session_id = getattr(_tool_context, "session_id", None)
    exec_dir = (ASYNC_EXEC_DIR / session_id) if session_id else ASYNC_EXEC_DIR
    exec_dir.mkdir(parents=True, exist_ok=True)
    exec_id = uuid.uuid4().hex[:12]
    stdout_path = str(exec_dir / f"{exec_id}.stdout")
    stderr_path = str(exec_dir / f"{exec_id}.stderr")
    stdin_path  = str(exec_dir / f"{exec_id}.stdin")
    cwd = os.getcwd()

    now = time.monotonic()
    t0 = now + wait_before_s
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() + wait_before_s))

    with _async_exec_lock:
        _async_executions[exec_id] = {
            "proc": None,
            "command": command,
            "stdout_file": stdout_path,
            "stderr_file": stderr_path,
            "stdin_file":  stdin_path,
            "start": t0,
            "started_at": started_at,
            "cwd": cwd,
            "io_before": dict.fromkeys(_IO_FIELDS, 0),
            "scheduled_for": t0,
            "fast_done": False,
        }

    def _spawn() -> None:
        """Sleep out the delay, then run the command (the body of the old synchronous path)."""
        if wait_before_s:
            time.sleep(wait_before_s)

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

        proc = subprocess.Popen(
            command,
            shell=True,
            stdin=stdin_read_fh,
            stdout=stdout_fh,
            stderr=stderr_fh,
            preexec_fn=os.setsid,
        )
        stdin_read_fh.close()   # process has inherited the fd; we don't need it

        with _async_exec_lock:
            _async_executions[exec_id]["proc"] = proc

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

        global _last_queried_exec_id
        _last_queried_exec_id = None

        with _async_exec_lock:
            entry_now = _async_executions[exec_id]
            entry_now["fast_done"] = fast_done

    if wait_before_s:
        threading.Thread(target=_spawn, daemon=True).start()
        scheduled: dict[str, object] = {
            "tool_exec_id": exec_id,
            "scheduled_for": started_at,
            "starts_in_seconds": round(wait_before_s, 3),
            "cwd": cwd,
            "stdin_localfile":  stdin_path,
            "stdout_localfile": stdout_path,
            "stderr_localfile": stderr_path,
            "command": command,
        }
        r = json.dumps(scheduled)
        return r, {"result": r}

    _spawn()
    with _async_exec_lock:
        entry = _async_executions[exec_id]
    proc = entry["proc"]
    assert proc is not None   # synchronous path: _spawn ran to completion

    duration = round(time.monotonic() - t0, 3)

    result: dict[str, object] = {
        "tool_exec_id": exec_id,
        "pid":              proc.pid,
        "started_at":       started_at,
        "cwd":              cwd,
        "stdin_localfile":  stdin_path,
        "stdout_localfile": stdout_path,
        "stderr_localfile": stderr_path,
    }
    if entry.get("fast_done"):
        result["completed"] = True
        result["returncode"] = proc.returncode
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
    if proc is None:
        # Delayed start not elapsed yet (wait_before_s): the command has not run.
        starts_in = round(entry["scheduled_for"] - time.monotonic(), 3)
        scheduled_result: dict[str, object] = {
            "completed": False,
            "scheduled": True,
            "starts_in_seconds": max(starts_in, 0.0),
            "started_at": entry["started_at"],
            "cwd":         entry.get("cwd", ""),
            "command":     entry["command"],
            "stdout_localfile": entry["stdout_file"],
            "stderr_localfile": entry["stderr_file"],
        }
        r = json.dumps(scheduled_result)
        return r, {"result": r}

    returncode = entry["proc"].poll() if entry["proc"] is not None else None
    completed = returncode is not None

    global _last_queried_exec_id
    if not completed and tool_exec_id == _last_queried_exec_id:
        # Consecutive poll of the same still-running execution: busy-waiting
        # wastes turns. Answer with the still-running state and point at
        # nohup_wait, which sleeps and reports this execution as soon as it
        # finishes.
        r = json.dumps({
            "error": (
                "nohup_query was just called for this tool_exec_id and it is "
                "still running"
            ),
            "hint": "nohup_wait(tool_exec_id, howmuch, unit) returns as soon as "
                    "it finishes.",
            "tool_exec_id": tool_exec_id,
        })
        return r, {"result": r}
    _last_queried_exec_id = tool_exec_id
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


# ── nohup / nohup_query / nohup_wait ───────────────────────────────────────────────────────

def t_nohup(command: str, timeout: int = NOHUP_TIMEOUT_MIN,
            wait_before_s: float = 0) -> tuple[str, dict[str, object]]:
    """Bound the command with timeout(1) then hand off to t_execute_async.

    A ``wait_before_s`` delay elapses *before* the command starts and is not
    counted against the bound, which begins when the command runs.
    """
    if wait_before_s < 0:
        r = json.dumps({"error": "wait_before_s must be >= 0"})
        return r, {"result": r}
    if wait_before_s > WAIT_BEFORE_MAX_SECONDS:
        r = json.dumps({"error": f"wait_before_s exceeds the {WAIT_BEFORE_MAX_SECONDS}s cap"})
        return r, {"result": r}
    return t_execute_async(f"timeout {int(timeout) * 60} {command}",
                           wait_before_s=wait_before_s)


# Units understood by t_nohup_wait.  Wait durations are computed as
# howmuch * WAIT_FOR_UNIT_SECONDS[unit]; unknown units are rejected.
WAIT_FOR_UNIT_SECONDS = {
    "s": 1,
    "m": 60,
    "h": 3600,
    "d": 86400,
}
# Cap so a typo in howmuch cannot block the tool thread for hours on end.
WAIT_FOR_MAX_SECONDS = 3600


def t_nohup_wait(tool_exec_id: str, howmuch: int | None = None, unit: str = "s") -> tuple[str, dict[str, object]]:
    """Wait for the execution *tool_exec_id* to finish, at most *howmuch* × *unit*.

    Async tools always come with three: ``nohup`` starts a command in the
    background, ``nohup_query`` polls one by ``tool_exec_id``, and
    ``nohup_wait`` sleeps until **this** execution completes (or the optional
    ``howmuch`` × ``unit`` budget expires) instead of busy-polling with
    ``nohup_query``.  Completions queued while waiting (returncode, output
    files, last output lines) are reported inline, so the model gets results
    without an extra round trip.

    When the budget expires before the execution finishes, the result says so
    (``completed: false``) and reports the CPU and I/O activity of the still
    running process, so the model can tell progress from a hang.

    Supported units: ``s`` seconds, ``m`` minutes, ``h`` hours, ``d`` days.
    """
    # Validate the (optional) wait budget before touching the exec id.
    seconds: float | None = None
    if howmuch is not None:
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

    with _async_exec_lock:
        entry = _async_executions.get(tool_exec_id)
    if entry is None:
        r = json.dumps({"error": f"unknown tool_exec_id: {tool_exec_id}"})
        return r, {"result": r}

    deadline = None if seconds is None else time.monotonic() + seconds
    others: list[_AsyncCompletion] = []
    done = False
    while True:
        proc = entry["proc"]
        if proc is not None and proc.poll() is not None:
            done = True
            break
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            break
        timeout = 0.1 if remaining is None else min(0.1, remaining)
        try:
            c = async_completion_queue.get(timeout=timeout)
        except _queue.Empty:
            continue
        if c["tool_exec_id"] == tool_exec_id:
            done = True
            break
        others.append(c)   # finished meanwhile, but not the one we wait on

    result: dict[str, object] = {"tool_exec_id": tool_exec_id, "completed": done}
    if done:
        proc = entry["proc"]
        assert proc is not None
        result["returncode"] = proc.returncode
        result["duration_time"] = round(time.monotonic() - entry["start"], 3)
        result["stdout_localfile"] = entry["stdout_file"]
        result["stderr_localfile"] = entry["stderr_file"]
        result["command"] = entry["command"]
        result["stdout_last_lines"] = _async_last_lines(entry["stdout_file"])
        result["stderr_last_lines"] = _async_last_lines(entry["stderr_file"])
        _async_add_inline(result, entry["stdout_file"], entry["stderr_file"])
    else:
        if entry["proc"] is None:
            result["scheduled"] = True
            result["starts_in_seconds"] = max(round(entry["scheduled_for"] - time.monotonic(), 3), 0.0)
        result["waited_seconds"] = round(seconds if seconds is not None
                                         else time.monotonic() - entry["start"], 3)
        result["activity"] = _activity_report(entry)
        result["hint"] = (
            "not completed yet; call nohup_wait again for this tool_exec_id "
            "with a fresh howmuch budget."
        )
    if others:
        result["also_completed"] = [
            {
                "tool_exec_id":      c["tool_exec_id"],
                "returncode":        c["returncode"],
                "duration_time":     c["duration"],
                "stdout_localfile":  c["stdout_file"],
                "stderr_localfile":  c["stderr_file"],
            }
            for c in others
        ]

    r = json.dumps(result)
    return r, {"result": r}


def nohup_tool_specs(timeout_min: int = NOHUP_TIMEOUT_MIN) -> list[dict[str, Any]]:
    """JSON tool specs for the nohup / nohup_query / nohup_wait trio."""
    return [
        {
            "type": "function",
            "function": {
                "name": "nohup",
                "description": (
                    "Start a shell command asynchronously, like nohup(1). Returns "
                    "tool_exec_id, the process id (pid), and local file paths for "
                    "stdin (FIFO), stdout, and stderr. Write to stdin_localfile "
                    "to send input to the running process. Execution is bounded: "
                    f"the command is killed after `timeout` minutes (default {timeout_min}). "
                    f"If the command finishes within {int(ASYNC_FAST_THRESHOLD_S * 1000)} ms "
                    f"and both outputs are under {ASYNC_INLINE_MAX_BYTES} bytes, "
                    "stdout/stderr are inlined immediately. To run a command in N "
                    "seconds from now, pass wait_before_s=N: the command starts N "
                    "seconds later and its full `timeout` budget is available "
                    "when it starts."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to run."},
                        "timeout": {
                            "type": "integer",
                            "description": (
                                "Maximum minutes the command may run before being "
                                f"killed (default {timeout_min}). Counts from the "
                                "moment the command starts, not from wait_before_s."
                            ),
                        },
                        "wait_before_s": {
                            "type": "number",
                            "description": (
                                "Seconds to wait before starting the command "
                                f"(default 0). Cap {WAIT_BEFORE_MAX_SECONDS}s. The "
                                "tool returns immediately with tool_exec_id and file "
                                "paths; poll/wait on that id as usual."
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
                    f"{ASYNC_INLINE_MAX_BYTES} bytes; otherwise reports file sizes. "
                    "While a command is still running, nohup_wait(tool_exec_id, "
                    "howmuch, unit) is the efficient way to let it finish."
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
                "name": "nohup_wait",
                "description": (
                    "Wait for a background command started with nohup to finish. "
                    "Takes a mandatory "
                    "tool_exec_id and an optional howmuch × unit wait budget. "
                    "Returns as soon as that execution completes, reporting "
                    "returncode, output file paths and the last lines of "
                    "stdout/stderr. If the budget (if given) expires first, reports "
                    "completed: false together with the CPU and I/O activity of the "
                    "still-running process, so you can tell progress from a hang; "
                    "then call nohup_wait again for the same tool_exec_id. Without "
                    "howmuch, waits indefinitely until it completes."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tool_exec_id": {
                            "type": "string",
                            "description": "The tool_exec_id returned by nohup.",
                        },
                        "howmuch": {
                            "type": "integer",
                            "description": (
                                "Optional wait budget in the given unit. When "
                                "reached with the execution still running, the tool "
                                "returns instead of blocking further."
                            ),
                        },
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
                    "required": ["tool_exec_id"],
                },
            },
        },
    ]


def enable_nohup(schema: dict[str, Any], timeout_min: int = NOHUP_TIMEOUT_MIN) -> dict[str, Any]:
    """Add the nohup / nohup_query / nohup_wait tools to *schema* in place.

    Appends the tool specs (both ``tool_specs`` and ``inferred_tool_schema``)
    and wires dispatch to the ``t_nohup`` / ``t_query_exec`` / ``t_nohup_wait``
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
            schema["tools"] = list(schema["tools"]) + ["t_nohup", "t_query_exec", "t_nohup_wait"]
        else:
            schema.setdefault("tool_dispatch", {})
            schema["tool_dispatch"].update({
                "nohup":       {"python_function": "t_nohup",      "param_map": {}},
                "nohup_query": {"python_function": "t_query_exec", "param_map": {}},
                "nohup_wait":    {"python_function": "t_nohup_wait",   "param_map": {}},
            })
    return schema
