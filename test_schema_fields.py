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
                    "name": "execute_shell_command",
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
