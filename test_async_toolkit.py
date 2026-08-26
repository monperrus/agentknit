"""Tests for agentknit.async_toolkit — nohup/nohup_query/wait_for tools and specs."""

from __future__ import annotations

import json
import queue as _queue
import time

from agentknit.async_toolkit import (
    NOHUP_TIMEOUT_MIN,
    WAIT_FOR_MAX_SECONDS,
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
    assert specs[2]["function"]["parameters"]["required"] == ["tool_exec_id"]
    assert specs[2]["function"]["parameters"]["properties"]["unit"]["enum"] == ["d", "h", "m", "s"]
    assert "howmuch" in specs[2]["function"]["parameters"]["properties"]
    assert "activity" in specs[2]["function"]["description"]
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
    """wait_for sleeps until the exec finishes, then reports it inline."""
    _drain_queue()
    result, _ = t_nohup("echo wait-for-test && sleep 1")
    d = json.loads(result)
    t0 = time.monotonic()
    out = json.loads(t_wait_for(d["tool_exec_id"], 5, "s")[0])
    elapsed = time.monotonic() - t0
    assert out["tool_exec_id"] == d["tool_exec_id"]
    assert out["completed"] is True
    assert out["returncode"] == 0
    assert 0.9 <= elapsed < 5
    assert out["command"].endswith("echo wait-for-test && sleep 1")
    assert out["stdout_last_lines"] == "wait-for-test"
    assert out["stderr_last_lines"] == ""
    assert out["stdout"].strip() == "wait-for-test"


def test_wait_for_returns_immediately_when_already_done() -> None:
    """A finished exec completes the wait without sleeping."""
    _drain_queue()
    d = json.loads(t_nohup("true")[0])
    _drain(d["tool_exec_id"])
    t0 = time.monotonic()
    out = json.loads(t_wait_for(d["tool_exec_id"], 60, "s")[0])
    assert out["completed"] is True
    assert time.monotonic() - t0 < 5


def test_wait_for_reports_not_completed_with_activity() -> None:
    """When the budget expires first, wait_for says so plus CPU/IO activity."""
    _drain_queue()
    d = json.loads(t_nohup("sleep 30")[0])
    t0 = time.monotonic()
    out = json.loads(t_wait_for(d["tool_exec_id"], 1, "s")[0])
    elapsed = time.monotonic() - t0
    assert out["tool_exec_id"] == d["tool_exec_id"]
    assert out["completed"] is False
    assert 0.9 <= elapsed < 3
    assert out["waited_seconds"] == 1
    assert "not completed" in out["hint"]
    activity = out["activity"]
    assert set(activity["io_bytes"]) == {"rchar", "wchar", "read_bytes", "write_bytes"}
    assert all(isinstance(v, int) for v in activity["io_bytes"].values())
    assert activity["cpu_seconds"] >= 0
    assert activity["cpu_percent"] >= 0
    # drain the leftover process so it does not outlive the test session
    assert _drain(d["tool_exec_id"], timeout=45).get("returncode") is not None


def test_wait_for_activity_deltas_between_calls() -> None:
    """io_bytes measures progress since the previous wait_for report."""
    _drain_queue()
    d = json.loads(t_nohup("sleep 2 && dd if=/dev/zero of=/dev/null bs=1k count=2048 2>/dev/null")[0])
    out1 = json.loads(t_wait_for(d["tool_exec_id"], 0.5, "s")[0])
    assert out1["completed"] is False
    out2 = json.loads(t_wait_for(d["tool_exec_id"], 3, "s")[0])
    if out2["completed"]:
        # dd finished inside the second budget: nothing more to assert.
        assert out2["returncode"] == 0
        return
    assert out2["activity"]["io_bytes"]["rchar"] > 0   # dd read its input meanwhile


def test_wait_for_reports_other_completions_as_side_info() -> None:
    """Execs finishing meanwhile are reported, without ending the wait."""
    _drain_queue()
    a = json.loads(t_nohup("sleep 2")[0])
    b = json.loads(t_nohup("sleep 1")[0])
    out = json.loads(t_wait_for(a["tool_exec_id"], 10, "s")[0])
    assert out["tool_exec_id"] == a["tool_exec_id"]
    assert out["completed"] is True
    assert out["returncode"] == 0
    [other] = out["also_completed"]
    assert other["tool_exec_id"] == b["tool_exec_id"]
    assert other["returncode"] == 0


def test_wait_for_unknown_id() -> None:
    """Unknown ids error out before any sleeping happens."""
    t0 = time.monotonic()
    out = json.loads(t_wait_for("nope", 60, "s")[0])
    assert "unknown tool_exec_id" in out["error"]
    assert time.monotonic() - t0 < 1


def test_wait_for_no_howmuch() -> None:
    """Without howmuch there is no budget: wait ends on completion."""
    _drain_queue()
    d = json.loads(t_nohup("sleep 1")[0])
    t0 = time.monotonic()
    out = json.loads(t_wait_for(d["tool_exec_id"])[0])
    assert out["completed"] is True
    assert out["returncode"] == 0
    assert 0.9 <= time.monotonic() - t0 < 5
    assert "waited_seconds" not in out


def test_wait_for_units(monkeypatch) -> None:
    """howmuch × unit is the budget; a fake completion queue short-circuits it."""
    _drain_queue()
    for howmuch, unit in [(1, "s"), (90, "s"), (2, "m"), (1, "h")]:
        d = json.loads(t_nohup("sleep 30")[0])
        # Simulate instant completion so no real time is spent sleeping.
        async_completion_queue.put({
            "tool_exec_id": d["tool_exec_id"], "returncode": 0,
            "stdout_file": "", "stderr_file": "", "duration": 0.0, "cwd": "",
        })
        async_completion_queue.put({
            "tool_exec_id": "filler-" + d["tool_exec_id"], "returncode": 0,
            "stdout_file": "", "stderr_file": "", "duration": 0.0, "cwd": "",
        })
        monkeypatch.setattr("agentknit.async_toolkit.async_completion_queue.get_nowait",
                            lambda: (_ for _ in ()).throw(_queue.Empty))
        out = json.loads(t_wait_for(d["tool_exec_id"], howmuch, unit)[0])
        monkeypatch.undo()
        assert out["completed"] is True
    monkeypatch.setattr("agentknit.async_toolkit.time.sleep", lambda s: None)   # unused now
    monkeypatch.undo()


def test_wait_for_rejects_bad_input() -> None:
    """Unknown units, non-positive amounts and over-cap waits error out."""
    _drain_queue()
    d = json.loads(t_nohup("sleep 30")[0])
    assert "unknown unit" in json.loads(t_wait_for(d["tool_exec_id"], 1, "x")[0])["error"]
    assert "positive" in json.loads(t_wait_for(d["tool_exec_id"], 0, "s")[0])["error"]
    assert "positive" in json.loads(t_wait_for(d["tool_exec_id"], -5, "m")[0])["error"]
    cap = json.loads(t_wait_for(d["tool_exec_id"], WAIT_FOR_MAX_SECONDS + 1, "s")[0])["error"]
    assert "exceeds" in cap
    assert "exceeds" in json.loads(t_wait_for(d["tool_exec_id"], 2, "h")[0])["error"]
    assert "exceeds" in json.loads(t_wait_for(d["tool_exec_id"], 1, "d")[0])["error"]
    assert _drain(d["tool_exec_id"], timeout=45).get("returncode") is not None


# ── nohup_query consecutive-poll denial ──────────────────────────────────


def test_t_query_exec_denies_consecutive_same_id_polls() -> None:
    """A second poll of the same still-running exec is denied with a redirect."""
    result, _ = t_nohup("sleep 3")
    exec_id = json.loads(result)["tool_exec_id"]
    first = json.loads(t_query_exec(exec_id)[0])
    assert first["completed"] is False
    denied = json.loads(t_query_exec(exec_id)[0])
    assert "denied" in denied["error"]
    assert denied["tool_exec_id"] == exec_id
    assert "wait_for" in denied["hint"]
    assert "tool_exec_id" in denied["hint"]


def test_t_query_exec_denial_lifted_after_completion() -> None:
    """Once the exec completes, the same tool_exec_id polls normally again."""
    result, _ = t_nohup("sleep 1")
    exec_id = json.loads(result)["tool_exec_id"]
    assert json.loads(t_query_exec(exec_id)[0])["completed"] is False
    _drain(exec_id, timeout=10.0)
    final = json.loads(t_query_exec(exec_id)[0])
    assert final["completed"] is True
    assert final["returncode"] == 0


def test_t_query_exec_denial_only_for_consecutive_polls() -> None:
    """Polling another exec in between clears the denial for the first one."""
    a, _ = t_nohup("sleep 3")
    b, _ = t_nohup("sleep 3")
    id_a, id_b = json.loads(a)["tool_exec_id"], json.loads(b)["tool_exec_id"]
    assert json.loads(t_query_exec(id_a)[0])["completed"] is False
    assert json.loads(t_query_exec(id_b)[0])["completed"] is False
    # id_b was queried last, so polling id_a again is a fresh poll, not a repeat.
    again = json.loads(t_query_exec(id_a)[0])
    assert "error" not in again
    assert again["completed"] is False


def test_t_query_exec_denial_reset_by_new_execution() -> None:
    """Starting a new execution resets the tracker, so old ids poll again."""
    result, _ = t_nohup("sleep 3")
    exec_id = json.loads(result)["tool_exec_id"]
    assert "completed" in json.loads(t_query_exec(exec_id)[0])
    t_nohup("true")
    again = json.loads(t_query_exec(exec_id)[0])
    assert "error" not in again


def test_t_query_exec_unknown_id() -> None:
    """Unknown ids keep their original error."""
    out = json.loads(t_query_exec("nope")[0])
    assert "unknown tool_exec_id" in out["error"]

