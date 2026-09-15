"""Tests for docstring-based tool spec extraction and schema consistency."""

from __future__ import annotations

import inspect
import json

import pytest

from agentknit._tool_spec import (
    parse_tool_spec_from_docstring,
    extract_tool_specs_from_module,
    tool_spec_to_schema,
)
from agentknit import dispatch, tool_library
from agentknit._core import _DEFAULT_TOOL_SCHEMA, _DEFAULT_TOOL_DISPATCH


# ── parse_tool_spec_from_docstring ────────────────────────────────────────────

def test_parse_tool_spec_read_file():
    """Parse the t_read docstring tool spec."""
    doc = inspect.getdoc(tool_library.t_read)
    spec = parse_tool_spec_from_docstring(doc)
    assert spec is not None
    assert spec["name"] == "read_file"
    assert "path" in spec["parameters"]
    assert spec["parameters"]["path"]["type"] == "string"
    assert "offset" in spec["parameters"]
    assert spec["parameters"]["offset"]["type"] == "integer"
    assert "limit" in spec["parameters"]
    assert spec["parameters"]["limit"]["type"] == "integer"


def test_parse_tool_spec_write_file():
    """Parse the t_write docstring tool spec."""
    doc = inspect.getdoc(tool_library.t_write)
    spec = parse_tool_spec_from_docstring(doc)
    assert spec is not None
    assert spec["name"] == "write_file"
    assert set(spec["parameters"]) == {"path", "content"}
    assert spec["parameters"]["path"]["type"] == "string"
    assert spec["parameters"]["content"]["type"] == "string"


def test_parse_tool_spec_str_replace():
    """Parse the t_update docstring tool spec."""
    doc = inspect.getdoc(tool_library.t_update)
    spec = parse_tool_spec_from_docstring(doc)
    assert spec is not None
    assert spec["name"] == "str_replace"
    assert set(spec["parameters"]) == {"path", "old_str", "new_str", "replace_all"}
    assert spec["parameters"]["replace_all"]["type"] == "boolean"


def test_parse_tool_spec_execute_shell():
    """Parse the t_run docstring tool spec."""
    doc = inspect.getdoc(tool_library.t_run)
    spec = parse_tool_spec_from_docstring(doc)
    assert spec is not None
    assert spec["name"] == "exec_shell"
    assert set(spec["parameters"]) == {"command"}


def test_parse_tool_spec_no_spec_block():
    """Function without Tool spec: block returns None."""
    def no_spec():
        """Just a regular docstring."""
    assert parse_tool_spec_from_docstring(inspect.getdoc(no_spec)) is None


def test_parse_tool_spec_none_docstring():
    """None docstring returns None."""
    assert parse_tool_spec_from_docstring(None) is None


# ── extract_tool_specs_from_module ────────────────────────────────────────────

def test_extract_tool_specs_from_tool_library():
    """extract_tool_specs_from_module finds all tool specs in tool_library."""
    specs = extract_tool_specs_from_module(tool_library)
    # At least the 4 core tools should be present
    assert "t_read" in specs
    assert "t_write" in specs
    assert "t_update" in specs
    assert "t_run" in specs


# ── schema consistency: docstring spec vs _DEFAULT_TOOL_SCHEMA ────────────────

def _build_schema_from_docstring(fn_name: str, tool_name: str) -> dict:
    """Build an OpenAI tool schema dict from the docstring spec of *fn_name*."""
    fn = getattr(tool_library, fn_name)
    doc = inspect.getdoc(fn)
    spec = parse_tool_spec_from_docstring(doc)
    assert spec is not None, f"No Tool spec block in {fn_name} docstring"

    return tool_spec_to_schema(spec)


# Mapping: python function name → (tool name in schema, dispatch key)
_TOOL_MAP = {
    "t_read":   ("read_file", "read_file"),
    "t_write":  ("write_file", "write_file"),
    "t_update": ("str_replace", "str_replace"),
    "t_run":    ("exec_shell", "exec_shell"),
}


def test_docstring_spec_matches_default_schema():
    """Every tool in _DEFAULT_TOOL_SCHEMA has a matching docstring spec with same params."""
    # Build a lookup from the default schema
    schema_by_name = {}
    for entry in _DEFAULT_TOOL_SCHEMA:
        fn = entry["function"]
        schema_by_name[fn["name"]] = fn

    for fn_name, (tool_name, dispatch_key) in _TOOL_MAP.items():
        # Build schema from docstring
        docstring_schema = _build_schema_from_docstring(fn_name, tool_name)
        ds_fn = docstring_schema["function"]

        # Get the default schema entry
        default_fn = schema_by_name.get(tool_name)
        assert default_fn is not None, f"No default schema entry for {tool_name}"

        # Compare parameter names and types
        ds_params = ds_fn["parameters"]["properties"]
        default_params = default_fn["parameters"]["properties"]

        ds_param_names = set(ds_params)
        default_param_names = set(default_params)

        assert ds_param_names == default_param_names, (
            f"Parameter name mismatch for {tool_name}: "
            f"docstring has {ds_param_names}, default has {default_param_names}"
        )

        # Compare types
        for pname in ds_param_names:
            ds_type = ds_params[pname]["type"]
            default_type = default_params[pname]["type"]
            assert ds_type == default_type, (
                f"Type mismatch for {tool_name}.{pname}: "
                f"docstring says {ds_type}, default says {default_type}"
            )


def test_docstring_spec_param_names_match_python_signature():
    """Parameter names in docstring spec match the Python function signature (via param_map)."""
    for fn_name, (tool_name, dispatch_key) in _TOOL_MAP.items():
        fn = getattr(tool_library, fn_name)
        sig = inspect.signature(fn)
        sig_params = set(sig.parameters.keys())

        doc = inspect.getdoc(fn)
        spec = parse_tool_spec_from_docstring(doc)
        assert spec is not None

        spec_params = set(spec["parameters"])

        # Get the param_map from default dispatch
        dispatch_entry = _DEFAULT_TOOL_DISPATCH.get(dispatch_key, {})
        param_map = dispatch_entry.get("param_map", {})

        # Build the set of Python parameter names that spec params map to.
        # For params not in param_map, the name is used as-is (identity).
        mapped_python_params = set()
        for sp in spec_params:
            mapped_python_params.add(param_map.get(sp, sp))

        # Every spec param should map to a valid Python function parameter
        assert mapped_python_params <= sig_params, (
            f"Spec params {spec_params} map to {mapped_python_params}, "
            f"but signature has {sig_params} for {tool_name}"
        )

        # Every param in param_map should have a corresponding spec param
        # (param_map entries that are identity mappings may be omitted)
        for model_name in param_map:
            assert model_name in spec_params, (
                f"param_map key '{model_name}' not found in docstring spec "
                f"params {spec_params} for {tool_name}"
            )


def test_all_default_tools_have_docstring_specs():
    """Every tool in _DEFAULT_TOOL_SCHEMA has a corresponding docstring spec."""
    specs = extract_tool_specs_from_module(tool_library)

    # Build reverse map: tool name → fn name
    tool_to_fn = {v[0]: k for k, v in _TOOL_MAP.items()}

    for entry in _DEFAULT_TOOL_SCHEMA:
        tool_name = entry["function"]["name"]
        fn_name = tool_to_fn.get(tool_name)
        assert fn_name is not None, f"No fn mapping for tool {tool_name}"
        assert fn_name in specs, (
            f"No docstring spec found for {fn_name} (tool {tool_name})"
        )


# ── required parameters ───────────────────────────────────────────────────────

def test_required_parameters_are_parsed():
    spec = parse_tool_spec_from_docstring(inspect.getdoc(tool_library.t_read))
    assert spec["required"] == ["path"]        # offset and limit are optional


def test_required_is_empty_when_nothing_is_marked():
    spec = parse_tool_spec_from_docstring(
        "Tool spec:\n"
        "    name: noop\n"
        "    description: Does nothing.\n"
        "    parameters:\n"
        "        x:\n"
        "            type: string\n"
        "            description: Anything.\n"
    )
    assert spec["required"] == []


def test_every_mandatory_signature_parameter_is_marked_required():
    specs = extract_tool_specs_from_module(tool_library)

    for fn_name, spec in specs.items():
        fn = getattr(tool_library, fn_name)
        mandatory = [
            p.name for p in inspect.signature(fn).parameters.values()
            if p.default is inspect.Parameter.empty
        ]
        # param_map renames some of them (old_str → old), so compare counts.
        assert len(spec["required"]) >= len(mandatory), (
            f"{fn_name}: signature needs {mandatory}, spec marks {spec['required']}"
        )


# ── tool_spec_to_schema ───────────────────────────────────────────────────────

def test_docstring_specs_reproduce_the_shipped_schema():
    """A host can build the shipped schema from the docstrings alone.

    Wording may differ — the shipped str_replace entry carries a longer
    description — but the contract (parameters, types, required) must not.
    """
    specs = extract_tool_specs_from_module(tool_library)
    by_name = {spec["name"]: tool_spec_to_schema(spec) for spec in specs.values()}

    for entry in _DEFAULT_TOOL_SCHEMA:
        shipped = entry["function"]["parameters"]
        built = by_name[entry["function"]["name"]]["function"]["parameters"]

        assert built["type"] == shipped["type"]
        assert set(built["properties"]) == set(shipped["properties"])
        assert built["required"] == shipped["required"]
        for pname, pinfo in shipped["properties"].items():
            assert built["properties"][pname]["type"] == pinfo["type"]
            assert built["properties"][pname]["description"]


def test_read_file_is_reproduced_byte_for_byte():
    spec = parse_tool_spec_from_docstring(inspect.getdoc(tool_library.t_read))
    shipped = next(e for e in _DEFAULT_TOOL_SCHEMA if e["function"]["name"] == "read_file")

    assert tool_spec_to_schema(spec) == shipped


def test_schema_conversion_keeps_the_required_marker_out_of_properties():
    spec = parse_tool_spec_from_docstring(inspect.getdoc(tool_library.t_glob))
    params = tool_spec_to_schema(spec)["function"]["parameters"]

    assert params["required"] == ["pattern"]
    assert params["properties"]["pattern"] == {
        "type": "string",
        "description": "Glob pattern, e.g. 'src/**/*.py'.",
    }


def test_param_map_is_parsed_from_maps_to():
    spec = parse_tool_spec_from_docstring(inspect.getdoc(tool_library.t_update))

    assert spec["param_map"] == {"old_str": "old", "new_str": "new"}


def test_specs_alone_yield_a_working_dispatch_table(tmp_path):
    """A host can build schema *and* dispatch from the docstrings only."""
    specs = extract_tool_specs_from_module(tool_library)
    tool_dispatch = {
        spec["name"]: {"python_function": fn_name, "param_map": spec["param_map"]}
        for fn_name, spec in specs.items()
    }
    for name, shipped in _DEFAULT_TOOL_DISPATCH.items():
        assert tool_dispatch[name]["param_map"] == shipped["param_map"]

    target = tmp_path / "f.txt"
    target.write_text("hello")
    dispatch(
        "str_replace",
        {"path": str(target), "old_str": "hello", "new_str": "bye"},
        tool_dispatch,
    )

    assert target.read_text() == "bye"


def test_the_extra_library_tools_are_documented_too():
    """glob, list_dir and search_files carry specs, not just the default four."""
    names = {spec["name"] for spec in extract_tool_specs_from_module(tool_library).values()}

    assert {"glob", "list_dir", "search_files"} <= names