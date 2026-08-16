"""Tests for first-class custom (freeform/grammar) tools (issue #21).

``Tool(custom_format=...)`` declares an OpenAI custom tool: no JSON Schema is
synthesized, and the raw text argument is dispatched as a single ``input``
string kwarg.
"""

from __future__ import annotations

import json

from agentknit import Tool, build_tool_spec, register_tools_in_library
from agentknit._core import (
    _derive_param_map,
    _tool_name_from_spec,
    _tool_param_names,
    schema_props,
    inline_system_prompt,
    dispatch,
)
from agentknit.tool_library import TOOL_LIBRARY

LARK_GRAMMAR = 'start: WORD+\nWORD: /[a-z]+/'


def t_apply_patch(input: str) -> tuple[str, dict]:  # noqa: A002
    return f"patch applied: {input}", {"result": f"patch applied: {input}"}


def test_custom_tool_emits_custom_shape() -> None:
    """custom_format emits {"type": "custom", ...} with no parameters key."""
    tool = Tool(
        name="apply_patch",
        description="Apply a patch that adds, updates, moves or deletes files.",
        fn=t_apply_patch,
        custom_format={"type": "grammar", "syntax": "lark",
                       "definition": LARK_GRAMMAR},
    )
    schema, dispatch = build_tool_spec([tool])
    assert schema == [{
        "type": "custom",
        "name": "apply_patch",
        "description": "Apply a patch that adds, updates, moves or deletes files.",
        "format": {"type": "grammar", "syntax": "lark",
                   "definition": LARK_GRAMMAR},
    }]
    assert "parameters" not in schema[0]
    assert "function" not in schema[0]
    assert dispatch["apply_patch"]["python_function"] == "t_apply_patch"
    assert dispatch["apply_patch"]["param_map"] == {"input": "input"}


def test_custom_tool_nested_grammar_shape() -> None:
    """The chat-completions nesting {"type": "grammar", "grammar": {...}} is
    re-emitted verbatim."""
    nested = {"type": "grammar",
              "grammar": {"syntax": "lark", "definition": LARK_GRAMMAR}}
    tool = Tool("exec", "Run a shell command.", t_apply_patch,
                custom_format=nested)
    schema, _ = build_tool_spec([tool])
    assert schema[0]["format"] is nested


def test_function_tool_unchanged_by_default() -> None:
    """Without custom_format the function shape is emitted exactly as before."""
    def t_read(path: str) -> tuple[str, dict]:
        return path, {"result": path}

    schema, dispatch = build_tool_spec([Tool("read_file", "Read", t_read)])
    assert schema[0]["type"] == "function"
    assert schema[0]["function"]["name"] == "read_file"
    assert schema[0]["function"]["parameters"]["properties"]["path"]["type"] == "string"
    assert dispatch["read_file"]["param_map"] == {"path": "path"}


def test_custom_tool_dispatch_delivers_input_kwarg() -> None:
    """dispatch() routes {"input": raw} to fn's `input` parameter."""
    register_tools_in_library([Tool("apply_patch", "patch", t_apply_patch,
                                    custom_format={"type": "grammar"})])
    result, meta = dispatch("apply_patch", {"input": "*** Begin Patch"},
                            {"apply_patch": {"python_function": "t_apply_patch",
                                             "param_map": {"input": "input"}}})
    assert result == "patch applied: *** Begin Patch"
    assert meta["result"] == result


def test_core_helpers_understand_custom_specs() -> None:
    """_tool_name_from_spec / _tool_param_names / schema_props handle custom."""
    spec = {"type": "custom", "name": "apply_patch", "description": "d",
            "format": {"type": "grammar", "syntax": "lark",
                       "definition": LARK_GRAMMAR}}
    assert _tool_name_from_spec(spec) == "apply_patch"
    assert _tool_param_names(spec) == []
    assert schema_props(spec) == {"input": {"type": "string"}}


def test_derive_param_map_for_custom_spec() -> None:
    """_derive_param_map returns {"input": "input"} for custom specs."""
    TOOL_LIBRARY["t_apply_patch_test"] = t_apply_patch
    spec = {"type": "custom", "name": "apply_patch"}
    assert _derive_param_map(spec, "t_apply_patch_test") == {"input": "input"}


def test_inline_system_prompt_includes_custom_tools() -> None:
    """Inline (unstructured) mode advertises custom tools with input=<text>."""
    prompt = inline_system_prompt([
        {"type": "custom", "name": "exec", "description": "d",
         "format": {"type": "grammar"}},
    ])
    assert json.dumps({"name": "exec", "arguments": {"input": "<text>"}}) in prompt


def test_custom_tool_end_to_end_through_dispatch() -> None:
    """A grammar tool built with Tool() runs end to end through dispatch."""
    tools = [Tool(
        "apply_patch",
        "Apply a patch that adds, updates, moves or deletes files.",
        t_apply_patch,
        custom_format={"type": "grammar", "syntax": "lark",
                       "definition": LARK_GRAMMAR},
    )]
    register_tools_in_library(tools)
    schema, dispatch_map = build_tool_spec(tools)

    # The wire shape the endpoint receives.
    assert schema[0]["type"] == "custom"
    # The model returns a custom_tool_call → _run_turn dispatches {"input": raw}.
    result, _ = dispatch("apply_patch", {"input": "add file"},
                         dispatch_map)
    assert result == "patch applied: add file"
