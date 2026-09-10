"""Tests that environment awareness is injected into the system prompt."""

from __future__ import annotations

import getpass
import platform
import subprocess
from pathlib import Path

from agentknit._core import environment_context, init_session

_MINIMAL_SCHEMA = {
    "model": "test-model",
    "endpoint": "https://example.com",
    "tool_specs": [],
    "behaviour": {"call_delivery_mode": "structured_tool_calls"},
}


def _sys_msg(schema: dict, **kwargs) -> str:
    session = init_session(schema, **kwargs)
    return session["messages"][0]["content"]


def test_system_prompt_contains_environment_block() -> None:
    msg = _sys_msg(_MINIMAL_SCHEMA)
    assert "## Environment" in msg
    assert "Working directory:" in msg
    assert "OS:" in msg
    assert "Current date/time:" in msg
    assert "Scratchpad" in msg
    assert "Model: test-model" in msg


def test_user_identity_present() -> None:
    msg = _sys_msg(_MINIMAL_SCHEMA)
    assert f"unix user: {getpass.getuser()}" in msg


def test_model_version_shown_when_present() -> None:
    schema = dict(_MINIMAL_SCHEMA, version="2024-06-01")
    msg = _sys_msg(schema)
    assert "Model: test-model (version 2024-06-01)" in msg


def test_git_status_in_git_repo(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=tmp_path,
                   capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "tester@example.com"],
                   cwd=tmp_path, capture_output=True, check=True)
    (tmp_path / "a.txt").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "initial commit"], cwd=tmp_path,
                   capture_output=True, check=True)
    (tmp_path / "b.txt").write_text("dirty")

    with _cwd(tmp_path):
        block = environment_context("test-model")

    assert "Git: on branch " in block
    assert "last commit: initial commit" in block
    assert "b.txt" in block


class _cwd:
    """Context manager temporarily changing the process working directory."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.old = Path.cwd()

    def __enter__(self) -> None:
        import os
        os.chdir(self.path)

    def __exit__(self, *exc: object) -> None:
        import os
        os.chdir(self.old)


def test_no_git_block_outside_repo(tmp_path: Path) -> None:
    with _cwd(tmp_path):
        block = environment_context("test-model")
    assert "Git:" not in block


def test_os_and_arch_present() -> None:
    block = environment_context("test-model")
    assert platform.system() in block
    assert platform.machine() in block


def test_scratchpad_unique_per_working_directory(tmp_path: Path) -> None:
    from agentknit._core import _scratchpad_dir
    a = _scratchpad_dir(tmp_path / "proj")
    b = _scratchpad_dir(tmp_path / "other" / "proj")
    # Same basename, different paths → different scratchpads.
    assert a != b
    assert "agentknit-scratchpad-proj-" in a.name
    # Stable for the same cwd.
    assert _scratchpad_dir(tmp_path / "proj") == a
