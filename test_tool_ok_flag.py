"""A failed tool call is machine-detectable, not only readable."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentknit import default_tool_spec, dispatch, tool_library


@pytest.fixture
def tool_dispatch() -> dict[str, dict[str, object]]:
    return default_tool_spec()[1]


def test_success_leaves_ok_unset(tool_dispatch: dict, tmp_path: Path) -> None:
    target = tmp_path / "f.txt"
    target.write_text("hi\n")

    _, meta = dispatch("read_file", {"path": str(target)}, tool_dispatch)

    assert meta.get("ok", True) is True
    assert "error" not in meta


def test_a_missing_file_is_flagged(tool_dispatch: dict, tmp_path: Path) -> None:
    _, meta = dispatch("read_file", {"path": str(tmp_path / "nope")}, tool_dispatch)

    assert meta["ok"] is False
    assert "FileNotFoundError" in str(meta["error"])


def test_a_refused_edit_is_flagged(tool_dispatch: dict, tmp_path: Path) -> None:
    target = tmp_path / "f.txt"
    target.write_text("hello\n")

    _, meta = dispatch(
        "str_replace",
        {"path": str(target), "old_str": "absent", "new_str": "x"},
        tool_dispatch,
    )

    assert meta["ok"] is False
    assert "old string not found" in str(meta["error"])


def test_wrong_argument_names_are_flagged(tool_dispatch: dict) -> None:
    _, meta = dispatch("read_file", {"nonsense": 1}, tool_dispatch)

    assert meta["ok"] is False


def test_an_unresolvable_python_function_is_flagged() -> None:
    _, meta = dispatch("x", {}, {"x": {"python_function": "t_does_not_exist"}})

    assert meta["ok"] is False


def test_a_search_timeout_is_flagged(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("grep is missing")

    monkeypatch.setattr(tool_library.subprocess, "Popen", _boom)

    _, meta = tool_library.t_search(path=str(tmp_path), pattern="x")

    assert meta["ok"] is False


def test_output_that_merely_starts_with_error_is_not_a_failure(
    tool_dispatch: dict, tmp_path: Path
) -> None:
    """The prefix belongs to the model's text; only meta decides success."""
    target = tmp_path / "log.txt"
    target.write_text("ERROR: yesterday's log line\n")

    _, meta = dispatch("read_file", {"path": str(target)}, tool_dispatch)

    assert meta.get("ok", True) is True
