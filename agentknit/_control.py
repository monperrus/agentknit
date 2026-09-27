"""Unix-domain-socket control plane for an agentknit session.

Lets another process inject a message into a running session's inbox, or
query whether it is busy, without tmux keystrokes or re-parsing the journal
to infer state.

Wire format: newline-delimited JSON, one request per line, one reply per
line.

* ``{"cmd": "send", "message": "..."}`` -> pushes *message* onto the
  session's inbox; replies ``{"ok": true}``.
* ``{"cmd": "status"}`` -> replies ``{"ok": true, "busy": bool,
  "pending": int, "session_id": str}`` from in-memory state, no file I/O.
* Anything else (bad JSON, unknown ``cmd``) -> ``{"ok": false,
  "error": "..."}``; the connection stays open — one bad line must not kill
  the socket.

Local, unix-socket only: no network exposure, no auth beyond filesystem
permissions (owner-only, ``0600``, mirroring ``~/secrets/*``).
"""

from __future__ import annotations

import json
import socket
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._core import Session


def _control_socket_path(model: str, session_id: str,
                          session_dir: "str | Path | None" = None) -> Path:
    """Mirror ``_journal_path``: ``session_dir/control.sock`` when given,
    else ``LOG_BASE/safe_model_name(model)/{session_id}_control.sock``.

    Deferred import of ``._core`` — this module must stay free of a
    module-level dependency on ``_core`` (which imports :class:`ControlServer`
    from here), so ``LOG_BASE``/``safe_model_name`` are pulled in only when
    this function actually runs.
    """
    if session_dir is not None:
        return Path(session_dir) / "control.sock"
    from ._core import LOG_BASE, safe_model_name
    return LOG_BASE / safe_model_name(model) / f"{session_id}_control.sock"


class ControlServer:
    """Binds *socket_path* and serves the control protocol for *session*.

    One daemon accept-loop thread, one daemon thread per connection.  Reads
    ``session["_busy"]``, ``session["_control_inbox"]`` and
    ``session["session_id"]`` live on every request — no state is cached, so
    replies always reflect the session's current condition.
    """

    def __init__(self, socket_path: "str | Path", session: "Session | dict[str, Any]") -> None:
        self._path = Path(socket_path)
        self._session: "dict[str, Any]" = session  # type: ignore[assignment]
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            self._path.unlink()
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self._sock.bind(str(self._path))
            self._path.chmod(0o600)
            self._sock.listen(8)
        except OSError:
            self._sock.close()
            raise
        self._closed = threading.Event()
        self._accept_thread = threading.Thread(
            target=self._accept_loop, daemon=True,
            name=f"control-accept-{self._path.stem}")
        self._accept_thread.start()

    def _accept_loop(self) -> None:
        while not self._closed.is_set():
            try:
                conn, _addr = self._sock.accept()
            except OSError:
                return
            threading.Thread(target=self._serve_conn, args=(conn,),
                             daemon=True, name="control-conn").start()

    def _serve_conn(self, conn: socket.socket) -> None:
        with conn:
            buf = b""
            while not self._closed.is_set():
                try:
                    chunk = conn.recv(65536)
                except OSError:
                    return
                if not chunk:
                    return
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    reply = self._handle_line(line)
                    try:
                        conn.sendall(json.dumps(reply).encode() + b"\n")
                    except OSError:
                        return

    def _handle_line(self, line: bytes) -> "dict[str, Any]":
        try:
            req = json.loads(line.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            return {"ok": False, "error": f"invalid JSON: {exc}"}
        if not isinstance(req, dict):
            return {"ok": False, "error": "request must be a JSON object"}
        cmd = req.get("cmd")
        if cmd == "send":
            message = req.get("message")
            if not isinstance(message, str):
                return {"ok": False, "error": "'message' must be a string"}
            inbox = self._session.get("_control_inbox")
            if inbox is not None:
                inbox.put(message)
            return {"ok": True}
        if cmd == "status":
            inbox = self._session.get("_control_inbox")
            return {
                "ok": True,
                "busy": bool(self._session.get("_busy", False)),
                "pending": inbox.qsize() if inbox is not None else 0,
                "session_id": self._session.get("session_id"),
            }
        return {"ok": False, "error": f"unknown cmd: {cmd!r}"}

    def close(self) -> None:
        """Stop the accept loop and unlink the socket file.

        A dead session must not leave a stale path a client could connect
        to (and hang on, since nothing would ever accept()).
        """
        if self._closed.is_set():
            return
        self._closed.set()
        try:
            self._sock.close()
        except OSError:
            pass
        try:
            self._path.unlink()
        except FileNotFoundError:
            pass


def _control_request(socket_path: "str | Path", request: "dict[str, Any]", *,
                      timeout: float) -> "dict[str, Any]":
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(str(socket_path))
        sock.sendall(json.dumps(request).encode() + b"\n")
        buf = b""
        while b"\n" not in buf:
            chunk = sock.recv(65536)
            if not chunk:
                break
            buf += chunk
        line = buf.split(b"\n", 1)[0]
        result: "dict[str, Any]" = json.loads(line.decode("utf-8"))
        return result
    finally:
        sock.close()


def send_control_message(socket_path: "str | Path", message: str, *,
                          timeout: float = 5.0) -> "dict[str, Any]":
    """Connect to *socket_path*, send *message*, and return the reply.

    The client half of the control protocol — for a future watchdog,
    ``census.py``, tests, or any other external caller that wants to inject
    a task into a running session without tmux keystrokes.
    """
    return _control_request(socket_path, {"cmd": "send", "message": message},
                            timeout=timeout)
