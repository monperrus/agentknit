"""A session that never logs anything but lifecycle records leaves no transcript.

Sessions opened and closed without a task used to leave a one-line
``session_start`` journal behind each time: tens of thousands of empty files.
"""

from __future__ import annotations

import contextlib
import io
import json

import agentknit._core as _core
from agentknit import load_specification

MODEL = "test/model"
ENDPOINT = "https://api.test/v1"


def _session():
    schema = load_specification(MODEL, ENDPOINT)
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        return _core.init_session(schema, non_interactive=True, durable=False)


def _types(session) -> list[str]:
    return [json.loads(line)["type"]
            for line in session["log_path"].read_text().splitlines()]


def test_session_without_content_writes_no_transcript():
    session = _session()
    _core._log(session, {"type": "session_end", "session_id": session["session_id"]})
    assert not session["log_path"].exists()


def test_held_back_lifecycle_records_are_written_first_and_in_order():
    session = _session()
    _core._log(session, {"type": "user", "content": "do it"})
    _core._log(session, {"type": "session_end", "session_id": session["session_id"]})
    assert _types(session) == ["session_start", "user", "session_end"]


def test_any_record_of_substance_flushes_the_transcript():
    session = _session()
    _core._log(session, {"type": "tool_call", "name": "Bash", "args": {}})
    assert _types(session) == ["session_start", "tool_call"]


def test_sessions_built_without_the_buffer_write_through(tmp_path):
    session = {"log_path": tmp_path / "events.jsonl"}
    _core._log(session, {"type": "session_start"})  # type: ignore[arg-type]
    assert session["log_path"].exists()
