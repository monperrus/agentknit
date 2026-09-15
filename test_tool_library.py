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


def test_search_without_matches_is_empty(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("hay\n")

    result, meta = tool_library.t_search(path=str(tmp_path), pattern="needle")

    assert meta["matches"] == []
    assert json.loads(result) == {"matches": []}
