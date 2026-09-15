"""Live tool output can be redirected away from stdout."""

from __future__ import annotations

import io
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from agentknit import get_tool_output_stream, set_tool_output_stream, tool_library


@pytest.fixture(autouse=True)
def _restore_default() -> Iterator[None]:
    yield
    set_tool_output_stream(None)


def test_default_is_stdout() -> None:
    assert get_tool_output_stream() is sys.stdout


def test_default_follows_a_reassigned_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    """The default is resolved at write time, so pytest's capture still sees it."""
    tool_library.t_run("echo captured")
    assert "captured" in capsys.readouterr().out


def test_exec_shell_output_goes_to_the_configured_stream(
    capsys: pytest.CaptureFixture[str],
) -> None:
    sink = io.StringIO()
    set_tool_output_stream(sink)

    result, _ = tool_library.t_run("echo redirected")

    assert "redirected" in sink.getvalue()
    assert "redirected" in result          # the tool result is unaffected
    assert "redirected" not in capsys.readouterr().out


def test_search_output_goes_to_the_configured_stream(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "a.py").write_text("needle = 1\n")
    sink = io.StringIO()
    set_tool_output_stream(sink)

    tool_library.t_search(path=str(tmp_path), pattern="needle")

    assert "needle" in sink.getvalue()
    assert "needle" not in capsys.readouterr().out


def test_a_broken_sink_does_not_fail_the_tool() -> None:
    class Broken(io.StringIO):
        def write(self, s: str) -> int:
            raise OSError("gone")

    set_tool_output_stream(Broken())

    result, meta = tool_library.t_run("echo survives")

    assert "survives" in result
    assert meta["returncode"] == 0
