"""Control socket: inject a message into a running session, or query status.

See ``agentknit/_control.py`` and ``plan-control-socket.md``.
"""

from __future__ import annotations

import json
import socket

import agentknit
from agentknit import default_tool_spec, init_session, poll_control_inbox, send_control_message


def _session(tmp_path, **overrides):
    specs, tool_dispatch = default_tool_spec()
    return init_session(
        {"model": "m", "endpoint": "https://api.example.test/v1",
         "tool_specs": specs, "tool_dispatch": tool_dispatch},
        non_interactive=True, strict_cache_proof=False,
        session_dir=tmp_path / "sess", control_socket=True,
        **overrides,
    )


def _status(socket_path) -> dict:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(5.0)
    try:
        sock.connect(str(socket_path))
        sock.sendall(json.dumps({"cmd": "status"}).encode() + b"\n")
        return json.loads(sock.recv(65536).decode())
    finally:
        sock.close()


def test_session_dir_socket_path(tmp_path) -> None:
    session = _session(tmp_path)
    assert session["control_socket_enabled"] is True
    assert session["control_socket_path"] == tmp_path / "sess" / "control.sock"
    assert session["control_socket_path"].exists()


def test_send_while_idle_is_used_as_next_turn_task(tmp_path) -> None:
    session = _session(tmp_path)
    reply = send_control_message(session["control_socket_path"], "say hello")
    assert reply == {"ok": True}
    assert poll_control_inbox(session) == "say hello"
    assert poll_control_inbox(session) is None


def test_send_while_busy_is_queued_and_consumed_after(tmp_path) -> None:
    session = _session(tmp_path)
    session["_busy"] = True
    send_control_message(session["control_socket_path"], "queued task")
    # Not drained while busy — nobody polls mid-turn.
    st = _status(session["control_socket_path"])
    assert st == {"ok": True, "busy": True, "pending": 1, "session_id": session["session_id"]}
    session["_busy"] = False
    assert poll_control_inbox(session) == "queued task"


def test_status_idle_vs_busy(tmp_path) -> None:
    session = _session(tmp_path)
    assert _status(session["control_socket_path"]) == {
        "ok": True, "busy": False, "pending": 0, "session_id": session["session_id"]}
    session["_busy"] = True
    assert _status(session["control_socket_path"])["busy"] is True


def test_two_sequential_sends_delivered_in_order(tmp_path) -> None:
    session = _session(tmp_path)
    send_control_message(session["control_socket_path"], "first")
    send_control_message(session["control_socket_path"], "second")
    assert poll_control_inbox(session) == "first"
    assert poll_control_inbox(session) == "second"
    assert poll_control_inbox(session) is None


def test_close_unlinks_socket_file(tmp_path) -> None:
    session = _session(tmp_path)
    path = session["control_socket_path"]
    assert path.exists()
    session["_control_server"].close()
    assert not path.exists()


def test_bad_json_line_does_not_kill_the_socket(tmp_path) -> None:
    session = _session(tmp_path)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(5.0)
    try:
        sock.connect(str(session["control_socket_path"]))
        sock.sendall(b"not json\n")
        reply = json.loads(sock.recv(65536).decode())
        assert reply["ok"] is False
    finally:
        sock.close()
    # The socket (and the accept loop) survived — a fresh connection still works.
    assert send_control_message(session["control_socket_path"], "still alive") == {"ok": True}
    assert poll_control_inbox(session) == "still alive"


def test_poll_control_inbox_disabled_returns_none(tmp_path) -> None:
    specs, tool_dispatch = default_tool_spec()
    session = init_session(
        {"model": "m", "endpoint": "https://api.example.test/v1",
         "tool_specs": specs, "tool_dispatch": tool_dispatch},
        non_interactive=True, strict_cache_proof=False, session_dir=tmp_path / "sess",
    )
    assert session.get("control_socket_enabled") is None
    assert poll_control_inbox(session) is None


def test_unknown_cmd_returns_error(tmp_path) -> None:
    session = _session(tmp_path)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(5.0)
    try:
        sock.connect(str(session["control_socket_path"]))
        sock.sendall(json.dumps({"cmd": "nope"}).encode() + b"\n")
        reply = json.loads(sock.recv(65536).decode())
        assert reply["ok"] is False
    finally:
        sock.close()


def test_control_socket_path_helper_mirrors_journal_path(tmp_path) -> None:
    from agentknit._control import _control_socket_path

    assert _control_socket_path("m", "sid", tmp_path) == tmp_path / "control.sock"
    default_path = _control_socket_path("owner_model", "sid123")
    assert str(default_path).endswith("owner_model/sid123_control.sock")
    assert default_path == agentknit.LOG_BASE / "owner_model" / "sid123_control.sock"
