from __future__ import annotations

import pytest

from agentknit._core import _normalize_schema, validate_schema
from agentknit.exceptions import AgentSpecInvalidError


def test_normalize_schema_builds_dispatch_from_tools() -> None:
    schema = {
        "model": "test-model",
        "endpoint": "https://example.com",
        "tool_specs": [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "str_replace",
                    "description": "Replace text.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "old_str": {"type": "string"},
                            "new_str": {"type": "string"},
                        },
                        "required": ["path", "old_str", "new_str"],
                    },
                },
            },
        ],
        "tools": ["t_read", "t_update"],
        "behaviour": {"call_delivery_mode": "structured_tool_calls"},
    }

    normalized = _normalize_schema(schema)

    assert normalized["inferred_tool_schema"] == schema["tool_specs"]
    assert normalized["tool_dispatch"]["read_file"] == {
        "python_function": "t_read",
        "param_map": {},
    }
    assert normalized["tool_dispatch"]["str_replace"] == {
        "python_function": "t_update",
        "param_map": {"old_str": "old", "new_str": "new"},
    }


def test_validate_schema_accepts_tool_specs_name() -> None:
    schema = {
        "model": "test-model",
        "endpoint": "https://example.com",
        "tool_specs": [
            {
                "type": "function",
                "function": {
                    "name": "exec_shell",
                    "description": "Run a shell command.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                        },
                        "required": ["command"],
                    },
                },
            }
        ],
        "tools": ["t_run"],
    }

    validate_schema(schema)


def test_normalize_schema_rejects_unknown_tool_function() -> None:
    schema = {
        "model": "test-model",
        "tool_specs": [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                },
            }
        ],
        "tools": ["t_missing"],
    }

    with pytest.raises(AgentSpecInvalidError, match="Unknown tool function"):
        _normalize_schema(schema)


def test_normalize_schema_rejects_length_mismatch() -> None:
    schema = {
        "model": "test-model",
        "tool_specs": [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                },
            }
        ],
        "tools": ["t_read", "t_write"],
    }

    with pytest.raises(AgentSpecInvalidError, match="'tool_specs' has 1 entries but 'tools' has 2"):
        _normalize_schema(schema)


def test_default_dispatch_accepts_legacy_shell_tool_name() -> None:
    """Pre-rename specs named the shell tool 'execute_shell_command'."""
    schema = {
        "model": "test-model",
        "tool_specs": [
            {
                "type": "function",
                "function": {
                    "name": "execute_shell_command",
                    "description": "Run a shell command.",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ],
    }

    normalized = _normalize_schema(schema)

    assert normalized["tool_dispatch"]["execute_shell_command"] == {
        "python_function": "t_run",
        "param_map": {},
    }


def test_legacy_alias_is_dispatch_only_not_advertised() -> None:
    """A retired name must never be offered alongside its successor.

    Advertising both showed the model two identical shell tools
    ('exec_shell' and 'execute_shell_command'), wasting prompt tokens and
    leaving the choice between them ambiguous.
    """
    import agentknit

    schema = agentknit.load_specification("test-model", "http://example.invalid/v1")
    session = agentknit.init_session(schema)

    advertised = [(t.get("function") or t)["name"] for t in session["tools"]]
    assert "exec_shell" in advertised
    assert "execute_shell_command" not in advertised
    assert advertised == sorted(set(advertised), key=advertised.index)  # no duplicates

    # …but a model or restored session emitting the retired name still runs.
    assert session["tool_dispatch"]["execute_shell_command"] == {
        "python_function": "t_run",
        "param_map": {},
    }


def test_spec_declared_aliases_are_still_advertised() -> None:
    """Aliases in the spec are the caller's explicit choice, so keep offering them."""
    import agentknit

    schema = agentknit.load_specification("test-model", "http://example.invalid/v1")
    schema["aliases"] = {"run_command": "exec_shell"}
    session = agentknit.init_session(schema)

    advertised = [(t.get("function") or t)["name"] for t in session["tools"]]
    assert "run_command" in advertised
    assert session["tool_dispatch"]["run_command"]["python_function"] == "t_run"


def test_legacy_named_spec_keeps_its_own_tool() -> None:
    """A pre-rename spec naming the retired tool keeps advertising exactly that."""
    import agentknit

    schema = {
        "model": "test-model",
        "endpoint": "http://example.invalid/v1",
        "status": "default",
        "tool_specs": [
            {
                "type": "function",
                "function": {
                    "name": "execute_shell_command",
                    "description": "Run a shell command.",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ],
        "behaviour": {"call_delivery_mode": "structured_tool_calls"},
    }
    session = agentknit.init_session(schema)

    advertised = [(t.get("function") or t)["name"] for t in session["tools"]]
    assert advertised == ["execute_shell_command"]
    assert session["tool_dispatch"]["execute_shell_command"]["python_function"] == "t_run"
