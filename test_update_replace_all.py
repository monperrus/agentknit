"""Tests for agentknit.tool_library.t_update replace_all semantics."""

from __future__ import annotations

from pathlib import Path

from agentknit.tool_library import t_update
from agentknit._core import dispatch


def _write(tmp_path: Path, content: str) -> str:
    p = tmp_path / "f.txt"
    p.write_text(content)
    return str(p)


def test_default_replaces_first_occurrence_only(tmp_path: Path) -> None:
    """Without replace_all, only the first match is replaced."""
    path = _write(tmp_path, "x=1\ny=1\nz=1\n")
    r, meta = t_update(path=path, old="1", new="2")
    assert r.startswith("OK: replaced 1 of 3 occurrence(s)")
    assert "2 remaining" in r and "replace_all=True" in r
    assert Path(path).read_text() == "x=2\ny=1\nz=1\n"
    assert meta["diff_summary"] == {"path": path, "added": 1, "removed": 1}


def test_replace_all_true_replaces_every_occurrence(tmp_path: Path) -> None:
    """replace_all=True rewrites every occurrence."""
    path = _write(tmp_path, "x=1\ny=1\nz=1\n")
    r, meta = t_update(path=path, old="1", new="2", replace_all=True)
    assert r.startswith("OK: replaced 3 of 3 occurrence(s)")
    assert "remaining" not in r
    assert Path(path).read_text() == "x=2\ny=2\nz=2\n"
    assert meta["diff_summary"] == {"path": path, "added": 3, "removed": 3}


def test_single_occurrence_unchanged_by_flag(tmp_path: Path) -> None:
    """One match: default and replace_all behave identically, no hint."""
    for flag in (False, True):
        path = _write(tmp_path, "alpha\nbeta\n")
        r, _ = t_update(path=path, old="beta", new="gamma", replace_all=flag)
        assert r == (f"OK: replaced 1 of 1 occurrence(s) "
                     f"(1 line(s), 4 UTF-8 character(s)) in {path}")
        assert Path(path).read_text() == "alpha\ngamma\n"


def test_missing_old_string_still_errors(tmp_path: Path) -> None:
    """Absent old_str is refused regardless of replace_all."""
    path = _write(tmp_path, "abc\n")
    for flag in (False, True):
        r, _ = t_update(path=path, old="nope", new="x", replace_all=flag)
        assert r.startswith("ERROR: old string not found")


def test_multiline_and_unicode_counts(tmp_path: Path) -> None:
    """Line/char counts and diff summary scale with the number of matches."""
    path = _write(tmp_path, "old text\nold text\n")
    r, meta = t_update(path=path, old="old text", new="neü", replace_all=True)
    assert r.startswith("OK: replaced 2 of 2 occurrence(s) (2 line(s), 16 UTF-8 character(s))")
    assert meta["diff_summary"] == {"path": path, "added": 2, "removed": 2}
    assert Path(path).read_text() == "neü\nneü\n"


def test_dispatch_passes_replace_all_from_model_name(tmp_path: Path) -> None:
    """The str_replace tool name forwards replace_all through param_map."""
    path = _write(tmp_path, "a=1\nb=1\n")
    td = {"str_replace": {"python_function": "t_update",
                          "param_map": {"old_str": "old", "new_str": "new"}}}
    r, _ = dispatch("str_replace",
                    {"path": path, "old_str": "1", "new_str": "9", "replace_all": True},
                    td)
    assert "2 of 2 occurrence(s)" in r
    assert Path(path).read_text() == "a=9\nb=9\n"
