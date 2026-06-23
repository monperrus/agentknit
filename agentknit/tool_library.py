"""
Tool implementations for agent_probe.py.

Every callable here is a candidate value for the "python_function" field in
tool_dispatch.  New functions added by the probe's code-generation path land
here too (appended via _register_generated).
"""

import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path


# ── async shell execution ─────────────────────────────────────────────────────

# Persistent directory for stdout/stderr capture files (never deleted).
ASYNC_EXEC_DIR = Path.home() / ".cache" / "async_agent_execs"
ASYNC_EXEC_DIR.mkdir(parents=True, exist_ok=True)

# Inline output in the tool response when the command finishes this quickly …
ASYNC_FAST_THRESHOLD_S = 0.100
# … and both stdout and stderr are under this many bytes.
ASYNC_INLINE_MAX_BYTES = 4096

# exec_id → {"proc": Popen, "stdout_file": str, "stderr_file": str, "start": float}
_async_executions: dict[str, dict] = {}
_async_exec_lock = threading.Lock()

# Thread-local set by _core._handle_tool_call before each dispatch so tools
# can access the current session without being passed the session dict.
_tool_context = threading.local()


def _async_try_inline(path: str) -> str | None:
    """Return file text if it fits within ASYNC_INLINE_MAX_BYTES, else None."""
    try:
        p = Path(path)
        if p.stat().st_size > ASYNC_INLINE_MAX_BYTES:
            return None
        return p.read_text(errors="replace")
    except OSError:
        return None


def _async_add_inline(result: dict, stdout_path: str, stderr_path: str) -> None:
    """Append stdout/stderr content to *result* when both files are small enough."""
    out = _async_try_inline(stdout_path)
    err = _async_try_inline(stderr_path)
    if out is not None:
        result["stdout"] = out
    if err is not None:
        result["stderr"] = err


def t_execute_async(command: str, when: int = 0) -> tuple[str, dict]:
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

    os.mkfifo(stdin_path)
    stdout_fh = open(stdout_path, "wb")
    stderr_fh = open(stderr_path, "wb")

    # Open the FIFO write-end in a background thread (open() on a FIFO blocks
    # until a reader appears). The read-end is handed to the process.
    stdin_write_fh: list = []   # populated by the thread once the process opens it

    def _open_fifo_write() -> None:
        fh = open(stdin_path, "wb", buffering=0)
        stdin_write_fh.append(fh)

    fifo_thread = threading.Thread(target=_open_fifo_write, daemon=True)
    fifo_thread.start()

    stdin_read_fh = open(stdin_path, "rb")   # unblocks the writer thread

    t0 = time.monotonic()
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

    if fast_done:
        _close_on_exit()
    else:
        threading.Thread(target=_close_on_exit, daemon=True).start()

    duration = round(time.monotonic() - t0, 3)

    with _async_exec_lock:
        _async_executions[exec_id] = {
            "proc": proc,
            "stdout_file": stdout_path,
            "stderr_file": stderr_path,
            "stdin_file":  stdin_path,
            "stdin_write_fh": stdin_write_fh,
            "start": t0,
        }

    result: dict = {
        "tool_exec_id": exec_id,
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


def t_query_exec(tool_exec_id: str) -> tuple[str, dict]:
    """Poll the status of a command started with t_execute_async.

    When completed, returncode is included and stdout/stderr are inlined if
    both are under ASYNC_INLINE_MAX_BYTES.
    """
    with _async_exec_lock:
        entry = _async_executions.get(tool_exec_id)

    if entry is None:
        r = json.dumps({"error": f"unknown tool_exec_id: {tool_exec_id}"})
        return r, {"result": r}

    proc: subprocess.Popen = entry["proc"]
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

    result: dict = {
        "completed": completed,
        "duration_time": duration,
        "stdin_localfile": entry["stdin_file"],
        "stdout_localfile_size": stdout_size,
        "stderr_localfile_localsize": stderr_size,
    }
    if completed:
        result["returncode"] = returncode
        _async_add_inline(result, entry["stdout_file"], entry["stderr_file"])

    r = json.dumps(result)
    return r, {"result": r}


# Colour escapes needed for interactive user-facing prompts in t_ask_user*.
_BOLD = "\033[1m"
_YEL = "\033[33m"
_RED = "\033[31m"
_RESET = "\033[0m"
_RL_BOLD  = "\x01\033[1m\x02"
_RL_RESET = "\x01\033[0m\x02"

# Tracks the subprocess currently executing inside a tool, so the SIGINT
# handler in _core.py can SIGKILL it immediately on Ctrl-C.
_active_proc: "subprocess.Popen | None" = None


def t_read(path: str, offset: int | None = None, limit: int | None = None) -> tuple[str, dict]:
    try:
        content = Path(os.path.expanduser(path)).read_text()
        if offset is not None or limit is not None:
            lines = content.splitlines(keepends=True)
            start = offset if offset is not None else 0
            if start < 0:
                start = max(0, len(lines) + start)
            if limit is not None:
                end = start + limit
            else:
                end = len(lines)
            content = "".join(lines[start:end])
        return content, {"result": content}
    except Exception as e:
        r = f"ERROR: {e}"
        return r, {"result": r}

def t_write(path: str, content: str) -> tuple[str, dict]:
    try:
        p = Path(os.path.expanduser(path))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        r = f"OK: wrote {len(content)} bytes to {path}"
        return r, {"result": r}
    except Exception as e:
        r = f"ERROR: {e}"
        return r, {"result": r}

def _apply_patch_format(patch: str) -> tuple[str, dict]:
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
        r = "ERROR: apply_patch: could not find '*** Update File:' in patch"
        return r, {"result": r}

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


def t_update(path: str = "", old: str = "", new: str = "", patch: str = "") -> tuple[str, dict]:
    if patch:
        return _apply_patch_format(patch)
    try:
        p = Path(os.path.expanduser(path))
        text = p.read_text()
        if old not in text:
            r = (f"ERROR: old string not found in {path} "
                 f"({len(old)} chars, starts with {repr(old[:80])}). "
                 f"Re-read the file and copy the exact bytes.")
            return r, {"result": r}
        n = text.count(old)
        p.write_text(text.replace(old, new))
        # Count lines and UTF-8 characters in the replaced text
        old_lines = old.count('\n') + (1 if old else 0)
        old_chars = len(old)
        r = f"OK: replaced {n} occurrence(s) ({old_lines} line(s), {old_chars} UTF-8 character(s)) in {path}"
        return r, {"result": r}
    except Exception as e:
        r = f"ERROR: {e}"
        return r, {"result": r}

def t_run(command: str) -> tuple[str, dict]:
    global _active_proc
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

        def _drain(stream, sink):
            for line in stream:
                sink.append(line)
                print(line, end="", flush=True)

        t_out = threading.Thread(target=_drain, args=(proc.stdout, stdout_lines), daemon=True)
        t_err = threading.Thread(target=_drain, args=(proc.stderr, stderr_lines), daemon=True)
        t_out.start()
        t_err.start()

        try:
            proc.wait(timeout=60)
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
        combined = out
        if err:
            combined += ("\n" if combined else "") + err
        if proc.returncode != 0:
            combined += f"\n[exit {proc.returncode}]"
        return combined or "(no output)", {
            "stdout": out, "stderr": err, "returncode": proc.returncode,
            "streamed": True,
            "result": combined or "(no output)",
        }
    except subprocess.TimeoutExpired:
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                proc.kill()
        r = "ERROR: command timed out after 60 s"
        return r, {"error": r, "result": r}
    except Exception as e:
        r = f"ERROR: {e}"
        return r, {"error": r, "result": r}
    finally:
        _active_proc = None

def t_ask_user(question: str) -> tuple[str, dict]:
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
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        else:
            print("\a", end="", flush=True)
    except Exception:
        print("\a", end="", flush=True)


def t_ask_user_question(question: str = '', options: str = '') -> tuple[str, dict]:
    """Prompt the user with an optional numbered list of choices."""
    if not question:
        return 'ERROR: No question provided', {'result': 'error'}

    parsed_options: list = []
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

    try:
        answer = input(f"{_RL_BOLD}Your answer:{_RL_RESET} ").strip()
    except (EOFError, KeyboardInterrupt):
        return 'ERROR: No user input available', {'result': 'error'}

    if parsed_options and answer.isdigit():
        idx = int(answer)
        if 1 <= idx <= len(parsed_options):
            answer = str(parsed_options[idx - 1])

    r = json.dumps({"answer": answer})
    return r, {'result': answer, 'question': question, 'options': parsed_options}


def t_list_dir(path: str) -> tuple[str, dict]:
    try:
        entries = sorted(Path(os.path.expanduser(path)).iterdir(), key=lambda p: (p.is_file(), p.name))
        lines = [("d  " if e.is_dir() else "f  ") + e.name for e in entries]
        result = "\n".join(lines) or "(empty)"
        return result, {"result": result}
    except Exception as e:
        r = f"ERROR: {e}"
        return r, {"result": r}

def t_search(path: str = ".", pattern: str = "") -> tuple[str, dict]:
    global _active_proc
    proc: subprocess.Popen[str] | None = None
    try:
        proc = subprocess.Popen(
            ["grep", "-r", "-n", "--", pattern, path],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,
        )
        _active_proc = proc

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        def _drain(stream, sink):
            for line in stream:
                sink.append(line)
                print(line, end="", flush=True)

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

        out = "".join(stdout_lines) or "(no matches)"
        return out, {"result": out, "streamed": True}
    except subprocess.TimeoutExpired:
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                proc.kill()
        r = "ERROR: search timed out after 30 s"
        return r, {"result": r}
    except Exception as e:
        r = f"ERROR: {e}"
        return r, {"result": r}
    finally:
        _active_proc = None

def t_glob(pattern: str) -> tuple[str, dict]:
    import glob as _glob
    try:
        matches = sorted(_glob.glob(pattern, recursive=True))
        result = "\n".join(matches) or "(no matches)"
        return result, {"result": result, "matches": matches}
    except Exception as e:
        r = f"ERROR: {e}"
        return r, {"result": r}


# Functions that interactively ask the user something — excluded in --non-interactive mode.
_ASK_USER_FNS = {"t_ask_user", "t_ask_user_question"}

# Registry: function name (str) → callable.
TOOL_LIBRARY: dict[str, callable] = {
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
}

def _register_generated(fn_name: str, source: str) -> bool:
    """Exec *source* and add the resulting callable to TOOL_LIBRARY.

    Returns True on success, False if compilation/exec fails.
    """
    ns: dict = {}
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
def t_update_file(new_str: str = '', file_path: str = '', old_str: str = '') -> tuple[str, dict]:
    result_dict = {'result': 'success'}
    if not file_path:
        return ("ERROR: File path is required.", {'result': 'error'})
    try:
        with Path(os.path.expanduser(file_path)).open('r') as f:
            content = f.read()
        new_content = content.replace(old_str, new_str)
        with Path(os.path.expanduser(file_path)).open('w') as f:
            f.write(new_content)
        return ("File updated successfully.", result_dict)
    except Exception as e:
        return (f"ERROR: {str(e)}", {'result': 'error'})

TOOL_LIBRARY['t_update_file'] = t_update_file


# --- generated: t_list_directory ---
def t_list_directory(path: str = '') -> tuple[str, dict]:
    try:
        p = Path(os.path.expanduser(path))
        if not p.exists() or not p.is_dir():
            return ("ERROR: Path does not exist or is not a directory", {'result': 'error'})
        items = [str(item.name) for item in p.iterdir()]
        result_dict = {
            'result': 'success',
            'files': items
        }
        return ("Directory listing successful", result_dict)
    except Exception as e:
        return (f"ERROR: {str(e)}", {'result': 'error'})

TOOL_LIBRARY['t_list_directory'] = t_list_directory


# --- generated: t_search_files ---
def t_search_files(command: str = '') -> tuple[str, dict]:
    try:
        if not command:
            return "ERROR: command is required", {"result": "ERROR: command is required"}
        
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
        err = f"ERROR: {e}"
        return err, {"result": err}

TOOL_LIBRARY['t_search_files'] = t_search_files


# --- generated: t_find_files ---
def t_find_files(pattern: str = '', recursive: str = '') -> tuple[str, dict]:
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
        err = f"ERROR: {e}"
        return (err, {"result": err})

TOOL_LIBRARY['t_find_files'] = t_find_files
