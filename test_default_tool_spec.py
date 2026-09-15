"""agentknit's own tools are reachable as a (schema, dispatch) pair."""

from __future__ import annotations

from pathlib import Path

from agentknit import default_tool_spec, dispatch


def test_shape_matches_build_tool_spec() -> None:
    schema, tool_dispatch = default_tool_spec()

    assert sorted(tool_dispatch) == ["exec_shell", "read_file", "str_replace", "write_file"]
    assert {e["function"]["name"] for e in schema} == set(tool_dispatch)
    for entry in schema:
        assert entry["type"] == "function"
        assert entry["function"]["parameters"]["type"] == "object"
    for entry in tool_dispatch.values():
        assert entry["python_function"]
        assert isinstance(entry["param_map"], dict)


def test_the_returned_dispatch_actually_runs(tmp_path: Path) -> None:
    _, tool_dispatch = default_tool_spec()
    target = tmp_path / "f.txt"

    dispatch("write_file", {"path": str(target), "content": "hi\n"}, tool_dispatch)
    text, _ = dispatch("read_file", {"path": str(target)}, tool_dispatch)

    assert "hi" in text


def test_callers_get_their_own_copy() -> None:
    schema, tool_dispatch = default_tool_spec()
    schema.clear()
    tool_dispatch["read_file"]["param_map"]["path"] = "elsewhere"

    fresh_schema, fresh_dispatch = default_tool_spec()

    assert len(fresh_schema) == 4
    assert fresh_dispatch["read_file"]["param_map"] == {}
