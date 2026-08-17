"""Tests for the direct-tool ``run_agent`` convenience API (issue #6)."""

from __future__ import annotations

import agentknit
import agentknit._core as core
from agentknit.tool_library import TOOL_LIBRARY


def t_echo(value: str) -> tuple[str, dict]:
    return value, {"result": value}


def test_run_agent_builds_schema_registers_tools_and_delegates(monkeypatch) -> None:
    """run_agent is a thin adapter around the existing run_task API."""
    captured: dict = {}
    sentinel = object()

    def fake_run_task(schema, task, **kwargs):
        captured.update(schema=schema, task=task, kwargs=kwargs)
        return sentinel

    monkeypatch.setattr(core, "run_task", fake_run_task)
    monkeypatch.delitem(TOOL_LIBRARY, "t_echo", raising=False)

    result = agentknit.run_agent(
        task="Say hello",
        model="test/model",
        endpoint="https://api.test/v1",
        auth="opencode-github-copilot",
        tools=[agentknit.Tool("echo", "Echo a value", t_echo)],
        strict_cache_proof=False,
    )

    assert result is sentinel
    assert TOOL_LIBRARY["t_echo"] is t_echo
    assert captured["task"] == "Say hello"
    assert captured["schema"] == {
        "model": "test/model",
        "endpoint": "https://api.test/v1",
        "auth": "opencode-github-copilot",
        "inferred_tool_schema": [{
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Echo a value",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            },
        }],
        "tool_dispatch": {
            "echo": {"python_function": "t_echo", "param_map": {"value": "value"}},
        },
    }
    assert captured["kwargs"]["strict_cache_proof"] is False


def test_run_agent_omits_auth_when_not_given(monkeypatch) -> None:
    captured: dict = {}

    def fake_run_task(schema, task, **kwargs):
        captured["schema"] = schema

    monkeypatch.setattr(core, "run_task", fake_run_task)
    agentknit.run_agent(
        task="Say hello", model="test/model", endpoint="https://api.test/v1",
        tools=[agentknit.Tool("echo", "Echo a value", t_echo)],
    )

    assert "auth" not in captured["schema"]
