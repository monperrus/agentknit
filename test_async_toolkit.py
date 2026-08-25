"""Tests for agentknit.async_toolkit — nohup/nohup_query tools and specs."""

from __future__ import annotations

import json
import time

from agentknit.async_toolkit import (
    NOHUP_TIMEOUT_MIN,
    enable_nohup,
    nohup_tool_specs,
    t_execute_async,
    t_nohup,
    t_query_exec,
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
    """t_nohup is part of TOOL_LIBRARY, next to t_execute_async/t_query_exec."""
    assert TOOL_LIBRARY["t_nohup"] is t_nohup
    assert TOOL_LIBRARY["t_execute_async"] is t_execute_async
    assert TOOL_LIBRARY["t_query_exec"] is t_query_exec


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
    """Specs describe nohup and nohup_query with required params."""
    specs = nohup_tool_specs()
    names = [s["function"]["name"] for s in specs]
    assert names == ["nohup", "nohup_query"]
    assert specs[0]["function"]["parameters"]["required"] == ["command"]
    assert specs[1]["function"]["parameters"]["required"] == ["tool_exec_id"]
    assert str(NOHUP_TIMEOUT_MIN) in specs[0]["function"]["description"]


def test_enable_nohup_dispatch_schema() -> None:
    """enable_nohup adds specs + dispatch entries to a plain schema."""
    schema: dict = {"model": "m", "inferred_tool_schema": []}
    enable_nohup(schema)
    names = [t["function"]["name"] for t in schema["inferred_tool_schema"]]
    assert "nohup" in names and "nohup_query" in names
    assert schema["tool_dispatch"]["nohup"]["python_function"] == "t_nohup"
    assert schema["tool_dispatch"]["nohup_query"]["python_function"] == "t_query_exec"


def test_enable_nohup_tools_list_schema() -> None:
    """When 'tools' is present, enable_nohup extends it with library names."""
    schema: dict = {
        "inferred_tool_schema": [
            {"type": "function", "function": {"name": "exec_shell",
                                              "parameters": {"type": "object", "properties": {}}}},
        ],
        "tools": ["t_run"],
    }
    enable_nohup(schema)
    assert schema["tools"][-2:] == ["t_nohup", "t_query_exec"]
    assert len(schema["tools"]) == len(schema["inferred_tool_schema"])


def test_enable_nohup_idempotent() -> None:
    """Calling enable_nohup twice does not duplicate specs."""
    schema: dict = {"inferred_tool_schema": []}
    enable_nohup(schema)
    enable_nohup(schema)
    names = [t["function"]["name"] for t in schema["inferred_tool_schema"]]
    assert names.count("nohup") == 1
    assert names.count("nohup_query") == 1
