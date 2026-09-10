"""Tests for agentknit.tool — Tool dataclass, build_tool_spec, register_tools_in_library."""

from __future__ import annotations

import pytest
from agentknit.tool import Tool, build_tool_spec, register_tools_in_library, _infer_parameters
from agentknit.tool_library import TOOL_LIBRARY
from agentknit._core import FatalToolDispatchError


# ── sample tool implementations ───────────────────────────────────────────────

def t_read(path: str) -> tuple[str, dict]:
    return f"content of {path}", {"result": f"content of {path}"}


def t_write(path: str, content: str) -> tuple[str, dict]:
    return f"wrote to {path}", {"result": f"wrote to {path}"}


def t_update(path: str, old: str, new: str) -> tuple[str, dict]:
    return f"updated {path}", {"result": f"updated {path}"}


def t_run(command: str) -> tuple[str, dict]:
    return f"ran {command}", {"result": f"ran {command}"}


def t_add(a: int, b: int) -> tuple[str, dict]:
    s = a + b
    return str(s), {"result": s}


# ── Tool dataclass ────────────────────────────────────────────────────────────

def test_tool_defaults():
    """Tool stores name, description, fn; parameters/param_map default to None."""
    tool = Tool("read_file", "Read a file", t_read)
    assert tool.name == "read_file"
    assert tool.description == "Read a file"
    assert tool.fn is t_read
    assert tool.parameters is None
    assert tool.param_map is None


def test_tool_explicit_parameters():
    """Tool accepts explicit parameters dict."""
    params = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Path to the file."}},
        "required": ["path"],
    }
    tool = Tool("read_file", "Read a file", t_read, parameters=params)
    assert tool.parameters == params
    assert tool.resolved_parameters == params


def test_tool_explicit_param_map():
    """Tool accepts explicit param_map."""
    pmap = {"file_path": "path"}
    tool = Tool("read_file", "Read a file", t_read, param_map=pmap)
    assert tool.param_map == pmap
    assert tool.resolved_param_map == pmap


def test_tool_python_function_name():
    """python_function_name returns fn.__name__."""
    tool = Tool("read_file", "Read a file", t_read)
    assert tool.python_function_name == "t_read"


# ── _infer_parameters ─────────────────────────────────────────────────────────

def test_infer_parameters_basic():
    """_infer_parameters derives schema from function signature."""
    params = _infer_parameters(t_read)
    assert params["type"] == "object"
    assert "path" in params["properties"]
    assert params["properties"]["path"]["type"] == "string"
    assert params["required"] == ["path"]


def test_infer_parameters_optional():
    """Parameters with defaults are not required."""

    def t_foo(path: str, flag: bool = False) -> tuple[str, dict]:
        return "", {"result": ""}

    params = _infer_parameters(t_foo)
    assert params["required"] == ["path"]
    assert "flag" in params["properties"]
    assert params["properties"]["flag"]["type"] == "boolean"


def test_infer_parameters_annotations():
    """Type annotations map to correct JSON schema types."""

    def t_multi(
        a_str: str,
        b_int: int,
        c_float: float,
        d_bool: bool,
        e_default: str = "x",
        f_any=42,
    ) -> tuple[str, dict]:
        return "", {"result": ""}

    params = _infer_parameters(t_multi)
    assert params["properties"]["a_str"]["type"] == "string"
    assert params["properties"]["b_int"]["type"] == "integer"
    assert params["properties"]["c_float"]["type"] == "number"
    assert params["properties"]["d_bool"]["type"] == "boolean"
    assert params["properties"]["e_default"]["type"] == "string"
    # Unannotated defaults to "string"
    assert params["properties"]["f_any"]["type"] == "string"
    assert params["required"] == ["a_str", "b_int", "c_float", "d_bool"]


def test_infer_parameters_skips_self_cls():
    """self and cls parameters are skipped."""

    class Foo:
        def method(self, x: str) -> tuple[str, dict]:
            return "", {"result": ""}

    params = _infer_parameters(Foo.method)
    assert "self" not in params["properties"]
    assert "x" in params["properties"]

    # Also check that instance method works
    foo = Foo()
    params2 = _infer_parameters(foo.method)
    assert "self" not in params2["properties"]
    assert "x" in params2["properties"]


def test_infer_parameters_skips_var_args():
    """*args and **kwargs are skipped."""

    def t_var(x: str, *args: int, **kwargs: str) -> tuple[str, dict]:
        return "", {"result": ""}

    params = _infer_parameters(t_var)
    assert list(params["properties"].keys()) == ["x"]
    assert params["required"] == ["x"]


# ── resolved_parameters / resolved_param_map ──────────────────────────────────

def test_resolved_parameters_inferred():
    """resolved_parameters infers from fn when parameters is None."""
    tool = Tool("read_file", "Read a file", t_read)
    params = tool.resolved_parameters
    assert params["type"] == "object"
    assert "path" in params["properties"]
    assert params["properties"]["path"]["type"] == "string"
    assert params["required"] == ["path"]


def test_resolved_param_map_identity():
    """resolved_param_map returns identity mapping when param_map is None."""
    tool = Tool("read_file", "Read a file", t_read)
    pmap = tool.resolved_param_map
    assert pmap == {"path": "path"}


def test_resolved_param_map_explicit():
    """resolved_param_map returns explicit map when set."""
    pmap = {"file_path": "path"}
    tool = Tool("read_file", "Read a file", t_read, param_map=pmap)
    assert tool.resolved_param_map == pmap


# ── build_tool_spec ───────────────────────────────────────────────────────────

def test_build_tool_spec_single():
    """build_tool_spec returns (schema_list, dispatch_dict) for one tool."""
    tool = Tool(
        "read_file", "Read a file", t_read,
        parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    )
    schema, dispatch = build_tool_spec([tool])

    # Schema
    assert len(schema) == 1
    s = schema[0]
    assert s["type"] == "function"
    assert s["function"]["name"] == "read_file"
    assert s["function"]["description"] == "Read a file"
    assert s["function"]["parameters"]["properties"]["path"]["type"] == "string"
    assert s["function"]["parameters"]["required"] == ["path"]

    # Dispatch
    assert list(dispatch.keys()) == ["read_file"]
    d = dispatch["read_file"]
    assert d["python_function"] == "t_read"
    assert d["param_map"] == {"path": "path"}


def test_build_tool_spec_multiple():
    """build_tool_spec handles multiple tools."""
    tools = [
        Tool(
            "read_file", "Read a file", t_read,
            parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        ),
        Tool(
            "write_file", "Write a file", t_write,
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        ),
    ]
    schema, dispatch = build_tool_spec(tools)

    assert len(schema) == 2
    assert len(dispatch) == 2

    assert schema[0]["function"]["name"] == "read_file"
    assert schema[1]["function"]["name"] == "write_file"

    assert dispatch["read_file"]["python_function"] == "t_read"
    assert dispatch["write_file"]["python_function"] == "t_write"


def test_build_tool_spec_with_inferred_parameters():
    """build_tool_spec infers parameters from fn when none given."""
    tool = Tool("add", "Add two integers", t_add)
    schema, dispatch = build_tool_spec([tool])

    params = schema[0]["function"]["parameters"]
    assert params["type"] == "object"
    assert params["properties"]["a"]["type"] == "integer"
    assert params["properties"]["b"]["type"] == "integer"
    assert set(params["required"]) == {"a", "b"}

    assert dispatch["add"]["python_function"] == "t_add"
    assert dispatch["add"]["param_map"] == {"a": "a", "b": "b"}


def test_build_tool_spec_with_param_map():
    """build_tool_spec respects explicit param_map."""
    tool = Tool(
        "read_file", "Read a file", t_read,
        parameters={"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]},
        param_map={"file_path": "path"},
    )
    schema, dispatch = build_tool_spec([tool])

    assert dispatch["read_file"]["param_map"] == {"file_path": "path"}
    # Schema still uses model-facing names
    assert "file_path" in schema[0]["function"]["parameters"]["properties"]


def test_build_tool_spec_empty():
    """build_tool_spec with empty list returns empty structures."""
    schema, dispatch = build_tool_spec([])
    assert schema == []
    assert dispatch == {}


# ── register_tools_in_library ─────────────────────────────────────────────────

def test_register_tools_in_library():
    """register_tools_in_library registers fn in TOOL_LIBRARY."""
    # Remove if already present from previous tests
    TOOL_LIBRARY.pop("t_read", None)
    TOOL_LIBRARY.pop("t_write", None)

    tools = [
        Tool("read_file", "Read a file", t_read,
             parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),
        Tool("write_file", "Write a file", t_write,
             parameters={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}),
    ]
    register_tools_in_library(tools)

    assert TOOL_LIBRARY.get("t_read") is t_read
    assert TOOL_LIBRARY.get("t_write") is t_write


def test_register_tools_in_library_overwrites():
    """register_tools_in_library overwrites existing entries."""
    TOOL_LIBRARY["t_read"] = None
    tools = [
        Tool("read_file", "Read a file", t_read,
             parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),
    ]
    register_tools_in_library(tools)
    assert TOOL_LIBRARY["t_read"] is t_read  # noqa: E721 — intentional identity check


# ── integration: dispatch with callable python_function ───────────────────────

def test_dispatch_with_callable_in_entry():
    """dispatch works when python_function is a direct callable (not a name string)."""
    from agentknit._core import dispatch

    tool_dispatch = {
        "read_file": {
            "python_function": t_read,
            "param_map": {"path": "path"},
        },
    }
    result, meta = dispatch("read_file", {"path": "/tmp/test.txt"}, tool_dispatch)
    assert result == "content of /tmp/test.txt"
    assert meta["result"] == "content of /tmp/test.txt"


def test_dispatch_with_callable_and_param_map():
    """dispatch translates model arg names via param_map when using callable."""
    from agentknit._core import dispatch

    def t_renamed(path: str) -> tuple[str, dict]:
        return f"got {path}", {"result": f"got {path}"}

    tool_dispatch = {
        "read_cfg": {
            "python_function": t_renamed,
            "param_map": {"file_path": "path"},
        },
    }
    result, meta = dispatch("read_cfg", {"file_path": "/tmp/cfg.yaml"}, tool_dispatch)
    assert result == "got /tmp/cfg.yaml"


# ── end-to-end: build_tool_spec + register + dispatch ─────────────────────────

def test_end_to_end():
    """build_tool_spec + register + dispatch round-trip."""
    from agentknit._core import dispatch

    # 1. Define tools
    tools = [
        Tool(
            "read_file", "Read a file", t_read,
            parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        ),
        Tool(
            "add", "Add two ints", t_add,
            parameters={"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]},
        ),
    ]

    # 2. Build spec
    schema, dispatch_dict = build_tool_spec(tools)

    # 3. Register
    register_tools_in_library(tools)

    # 4. Dispatch
    result1, meta1 = dispatch("read_file", {"path": "/tmp/foo.txt"}, dispatch_dict)
    assert result1 == "content of /tmp/foo.txt"

    result2, meta2 = dispatch("add", {"a": 2, "b": 3}, dispatch_dict)
    assert result2 == "5"
    assert meta2["result"] == 5


# ── Tool is usable as a dataclass (equality, etc.) ────────────────────────────

def test_tool_equality():
    """Tool instances compare by field values (standard dataclass equality)."""
    t1 = Tool("read", "Read", t_read)
    t2 = Tool("read", "Read", t_read)
    assert t1 == t2
    t3 = Tool("write", "Write", t_write)
    assert t1 != t3


def test_tool_repr():
    """Tool has a readable repr."""
    tool = Tool("read", "Read a file", t_read)
    r = repr(tool)
    assert "Tool(" in r
    assert "name=" in r


# ── _resolve_fn helper (imported via dispatch) ────────────────────────────────

def test_resolve_fn_string():
    """_resolve_fn looks up string names in TOOL_LIBRARY."""
    from agentknit._core import _resolve_fn

    TOOL_LIBRARY["t_read"] = t_read
    entry = {"python_function": "t_read"}
    assert _resolve_fn(entry) is t_read


def test_resolve_fn_callable():
    """_resolve_fn returns callable directly."""
    from agentknit._core import _resolve_fn

    entry = {"python_function": t_read}
    assert _resolve_fn(entry) is t_read


def test_resolve_fn_none():
    """_resolve_fn returns None for missing python_function."""
    from agentknit._core import _resolve_fn

    assert _resolve_fn({}) is None
    assert _resolve_fn({"python_function": "nonexistent"}) is None


# ── error cases ───────────────────────────────────────────────────────────────

def test_dispatch_missing_tool():
    """dispatch raises FatalToolDispatchError for unknown tool."""
    from agentknit._core import dispatch

    with pytest.raises(FatalToolDispatchError):
        dispatch("nonexistent", {}, {})


def test_dispatch_missing_function():
    """dispatch returns error when python_function name is not in library."""
    from agentknit._core import dispatch

    TOOL_LIBRARY.pop("t_nonexistent", None)
    td = {"ghost": {"python_function": "t_nonexistent", "param_map": {}}}
    result, meta = dispatch("ghost", {}, td)
    assert "ERROR" in result
    assert "t_nonexistent" in result


def test_dispatch_tool_exception_includes_type_and_frames():
    """A raising tool yields a rich error: exception type + innermost frames."""
    from agentknit._core import dispatch

    def boom(**kwargs):
        raise AttributeError("'bool' object has no attribute 'splitlines'")

    result, meta = dispatch("weird", {}, {"weird": {"python_function": boom}})
    assert result.startswith("ERROR: tool 'weird' raised AttributeError:")
    assert "splitlines" in result
    assert "boom" in result  # innermost frame
    assert meta["result"] == result


def test_dispatch_coerces_non_str_tool_result():
    """A tool returning a non-str (e.g. a bool) is coerced, not crashed on."""
    from agentknit._core import dispatch

    def truthy(**kwargs):
        return True, {"result": True}

    result, meta = dispatch("truthy", {}, {"truthy": {"python_function": truthy}})
    assert result == "True"
    assert isinstance(meta["result"], str)


def test_str_replace_accepts_bool_old_str(tmp_path):
    """A model sending a boolean where the schema says string is coerced to
    str instead of crashing with AttributeError."""
    from agentknit.tool_library import t_update

    f = tmp_path / "f.txt"
    f.write_text("hello")
    result, _ = t_update(path=str(f), old=True, new="x")  # type: ignore[arg-type]
    assert result.startswith("ERROR: old string not found")


def test_write_accepts_bool_content(tmp_path):
    """write_file coerces non-str content instead of raising."""
    from agentknit.tool_library import t_write

    f = tmp_path / "g.txt"
    result, _ = t_write(path=str(f), content=False)  # type: ignore[arg-type]
    assert result.startswith("OK:")
    assert f.read_text() == "False"


def test_dispatch_type_error_guides_model_to_fix():
    """Wrong/missing args produce an actionable error naming the signature."""
    from agentknit._core import dispatch

    result, _ = dispatch("str_replace", {"path": "/x", "old_string": "a"},
                         {"str_replace": {"python_function": "t_update",
                                          "param_map": {"path": "path",
                                                        "old_str": "old",
                                                        "new_str": "new"}}})
    assert result.startswith("ERROR: invalid arguments for tool 'str_replace'")
    assert "t_update(" in result  # expected signature shown
    assert "old_string" in result  # the offending key is listed
    assert "call the tool again" in result


def test_malformed_tool_call_json_fed_back_to_model(monkeypatch):
    """Structured call with unparseable JSON arguments becomes a tool-role
    error message (not a silent {} dispatch), so the model sees it and fixes
    the call on the next iteration."""
    import agentknit._core as core
    from agentknit.openai_compat import (_Choice, _Function, _Message,
                                         _Response, _ToolCall, _Usage)

    calls: list[int] = []

    def fake_complete(client, session, **kwargs):
        def usage():
            return _Usage(prompt_tokens=1, completion_tokens=1,
                          total_tokens=2, has_cache_proof=True)
        if calls:
            resp = _Response([_Choice(_Message("assistant", "done", None))],
                             usage())
            resp.usage = usage()
            return resp
        calls.append(1)
        tc = _ToolCall("c1", _Function("read_file", '{"path": '), type="function")
        resp = _Response([_Choice(_Message("assistant", None, [tc]))], usage())
        resp.usage = usage()
        return resp

    monkeypatch.setattr(core, "_complete", fake_complete)

    import tempfile
    from pathlib import Path

    session = {
        "messages": [],
        "tools": [],
        "structured": True,
        "usage_totals": {"prompt": 0, "completion": 0, "total": 0,
                         "cached": 0, "cache_write": 0},
        "tool_dispatch": {"read_file": {"python_function": "t_read"}},
        "non_interactive": True,
        "cache_key": "test",
        "options": ["exclude-prompt_cache_key"],
        "on_event": lambda *a, **k: None,
        "strict_cache_proof": False,
        "log_path": Path(tempfile.mkdtemp()) / "log.jsonl",
        "session_id": "test-session",
    }

    result = core.run_turn(object(), "test/model", session, "hi")
    tool_msgs = [m for m in session["messages"] if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert "malformed tool call arguments" in tool_msgs[0]["content"]
    assert "read_file" in tool_msgs[0]["content"]
    assert result.final_reply == "done"  # turn recovered and finished
