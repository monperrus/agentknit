"""Tests for agentknit.async_toolkit — nohup/nohup_query/wait_for tools and specs."""

from __future__ import annotations

import json
import queue as _queue
import time

from agentknit.async_toolkit import (
    NOHUP_TIMEOUT_MIN,
    WAIT_FOR_MAX_SECONDS,
    WAIT_FOR_UNIT_SECONDS,
    async_completion_queue,
    enable_nohup,
    nohup_tool_specs,
    t_execute_async,
    t_nohup,
    t_query_exec,
    t_wait_for,
)
from agentknit.tool_library import TOOL_LIBRARY


def _drain(exec_id: str, timeout: float = 5.0) -> dict:
    """Poll t_query_exec until completed."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result, _ = t_query_exec(exec_id)
        d = json.loads(result)
        if d.get("completed"):
            return d
        time.sleep(0.05)
    raise AssertionError(f"exec {exec_id} did not complete")


def test_t_nohup_registered_in_tool_library() -> None:
    """The async trio is part of TOOL_LIBRARY, next to t_execute_async."""
    assert TOOL_LIBRARY["t_nohup"] is t_nohup
    assert TOOL_LIBRARY["t_execute_async"] is t_execute_async
    assert TOOL_LIBRARY["t_query_exec"] is t_query_exec
    assert TOOL_LIBRARY["t_wait_for"] is t_wait_for


def test_t_nohup_runs_and_reports_output() -> None:
    """t_nohup executes the command and query returns stdout + returncode."""
    result, _ = t_nohup("echo async-toolkit-test")
    d = json.loads(result)
    final = _drain(d["tool_exec_id"])
    assert final["returncode"] == 0
    assert final["stdout"].strip() == "async-toolkit-test"


def test_t_nohup_bounds_with_timeout(monkeypatch) -> None:
    """t_nohup prefixes the command with timeout(1), minutes → seconds."""
    seen: list[str] = []

    def _fake_execute(command: str, when: int = 0):
        seen.append(command)
        return json.dumps({"tool_exec_id": "x"}), {"result": "x"}

    monkeypatch.setattr("agentknit.async_toolkit.t_execute_async", _fake_execute)
    t_nohup("sleep 30")
    t_nohup("sleep 30", timeout=3)
    assert seen == ["timeout 600 sleep 30", "timeout 180 sleep 30"]


def test_nohup_tool_specs_shape() -> None:
    """Specs describe the nohup / nohup_query / wait_for trio with params."""
    specs = nohup_tool_specs()
    names = [s["function"]["name"] for s in specs]
    assert names == ["nohup", "nohup_query", "wait_for"]
    assert specs[0]["function"]["parameters"]["required"] == ["command"]
    assert specs[1]["function"]["parameters"]["required"] == ["tool_exec_id"]
    assert specs[2]["function"]["parameters"]["required"] == ["howmuch"]
    assert specs[2]["function"]["parameters"]["properties"]["unit"]["enum"] == ["d", "h", "m", "s"]
    assert str(NOHUP_TIMEOUT_MIN) in specs[0]["function"]["description"]


def test_enable_nohup_dispatch_schema() -> None:
    """enable_nohup adds specs + dispatch entries to a plain schema."""
    schema: dict = {"model": "m", "inferred_tool_schema": []}
    enable_nohup(schema)
    names = [t["function"]["name"] for t in schema["inferred_tool_schema"]]
    assert "nohup" in names and "nohup_query" in names and "wait_for" in names
    assert schema["tool_dispatch"]["nohup"]["python_function"] == "t_nohup"
    assert schema["tool_dispatch"]["nohup_query"]["python_function"] == "t_query_exec"
    assert schema["tool_dispatch"]["wait_for"]["python_function"] == "t_wait_for"


def test_enable_nohup_tools_list_schema() -> None:
    """When 'tools' is present, enable_nohup extends it with library names."""
    schema: dict = {
        "inferred_tool_schema": [
            {
                "type": "function",
                "function": {
                    "name": "exec_shell",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ],
        "tools": ["t_run"],
    }
    enable_nohup(schema)
    assert schema["tools"][-3:] == ["t_nohup", "t_query_exec", "t_wait_for"]
    assert len(schema["tools"]) == len(schema["inferred_tool_schema"])


def test_enable_nohup_idempotent() -> None:
    """Calling enable_nohup twice does not duplicate specs."""
    schema: dict = {"inferred_tool_schema": []}
    enable_nohup(schema)
    enable_nohup(schema)
    names = [t["function"]["name"] for t in schema["inferred_tool_schema"]]
    assert names.count("nohup") == 1
    assert names.count("nohup_query") == 1
    assert names.count("wait_for") == 1


# ── wait_for ─────────────────────────────────────────────────────────────


def _drain_queue() -> None:
    """Empty async_completion_queue so tests are independent of each other."""
    while True:
        try:
            async_completion_queue.get_nowait()
        except _queue.Empty:
            return


def test_wait_for_reports_completions() -> None:
    """wait_for sleeps, then reports background execs finished meanwhile."""
    _drain_queue()
    result, _ = t_nohup("echo wait-for-test && sleep 1")
    d = json.loads(result)
    t0 = time.monotonic()
    out = json.loads(t_wait_for(2, "s")[0])
    elapsed = time.monotonic() - t0
    assert out["waited_seconds"] == 2
    assert 1.9 <= elapsed < 3
    [done] = out["completed"]
    assert done["tool_exec_id"] == d["tool_exec_id"]
    assert done["returncode"] == 0
    assert done["command"].endswith("echo wait-for-test && sleep 1")
    assert done["stdout_last_lines"] == "wait-for-test"
    assert done["stderr_last_lines"] == ""


def test_wait_for_no_completions(monkeypatch) -> None:
    """An empty completion queue yields an empty list, and the queue is drained."""
    _drain_queue()
    monkeypatch.setattr("agentknit.async_toolkit.time.sleep", lambda s: None)
    out = json.loads(t_wait_for(5, "m")[0])
    assert out == {"waited_seconds": 300.0, "completed": []}


def test_wait_for_units(monkeypatch) -> None:
    """Duration is computed as howmuch × unit; sleep is called with seconds."""
    slept: list[float] = []
    monkeypatch.setattr("agentknit.async_toolkit.time.sleep", slept.append)
    _drain_queue()
    for howmuch, unit in [(1, "s"), (90, "s"), (2, "m"), (1, "h")]:
        out = json.loads(t_wait_for(howmuch, unit)[0])
        assert out["waited_seconds"] == howmuch * WAIT_FOR_UNIT_SECONDS[unit]
        assert slept[-1] == howmuch * WAIT_FOR_UNIT_SECONDS[unit]
    t0 = time.monotonic()
    json.loads(t_wait_for(0.01, "s")[0])
    assert time.monotonic() - t0 < 1


def test_wait_for_rejects_bad_input() -> None:
    """Unknown units, non-positive amounts and over-cap waits error out."""
    assert "unknown unit" in json.loads(t_wait_for(1, "x")[0])["error"]
    assert "positive" in json.loads(t_wait_for(0, "s")[0])["error"]
    assert "positive" in json.loads(t_wait_for(-5, "m")[0])["error"]
    cap = json.loads(t_wait_for(WAIT_FOR_MAX_SECONDS + 1, "s")[0])["error"]
    assert "exceeds" in cap
    assert "exceeds" in json.loads(t_wait_for(2, "h")[0])["error"]
    assert "exceeds" in json.loads(t_wait_for(1, "d")[0])["error"]
