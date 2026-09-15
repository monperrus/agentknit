"""Behaviour of the tool implementations in agentknit.tool_library."""

from __future__ import annotations

import json
from pathlib import Path

from agentknit import tool_library


def test_search_a_directory(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("needle = 1\n")
    (tmp_path / "b.py").write_text("hay = 2\n")

    _, meta = tool_library.t_search(path=str(tmp_path), pattern="needle")

    matches = meta["matches"]
    assert isinstance(matches, list)
    assert [(Path(str(m["file"])).name, m["line"]) for m in matches] == [("a.py", 1)]


def test_search_a_single_file(tmp_path: Path) -> None:
    """grep drops the filename for a lone file argument; -H keeps the matches."""
    target = tmp_path / "a.py"
    target.write_text("hay\nneedle = 1\n")

    result, meta = tool_library.t_search(path=str(target), pattern="needle")

    matches = meta["matches"]
    assert isinstance(matches, list)
    assert [(Path(str(m["file"])).name, m["line"], m["text"]) for m in matches] == [
        ("a.py", 2, "needle = 1")
    ]
    assert json.loads(result)["matches"] == matches


def test_canonical_tools_are_all_registered() -> None:
    for tool_name, fn_name in tool_library.CANONICAL_TOOLS.items():
        assert fn_name in tool_library.TOOL_LIBRARY, tool_name


def test_canonical_tools_are_exactly_the_ones_with_a_spec() -> None:
    """The map and the docstring blocks must not drift apart."""
    from agentknit import extract_tool_specs_from_module

    specs = extract_tool_specs_from_module(tool_library)
    documented = {spec["name"]: fn_name for fn_name, spec in specs.items()}

    assert documented == tool_library.CANONICAL_TOOLS


def test_superseded_tools_point_at_a_canonical_one() -> None:
    canonical = set(tool_library.CANONICAL_TOOLS.values())

    for old, replacement in tool_library.SUPERSEDED_TOOLS.items():
        assert old in tool_library.TOOL_LIBRARY      # still callable
        assert replacement in canonical


def test_search_without_matches_is_empty(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("hay\n")

    result, meta = tool_library.t_search(path=str(tmp_path), pattern="needle")

    assert meta["matches"] == []
    assert json.loads(result) == {"matches": []}
