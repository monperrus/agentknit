from __future__ import annotations

import agentknit._core as core
from agentknit._core import _build_resume_cmd, _is_framework_cli

import pytest


@pytest.fixture(autouse=True)
def _no_resume_env(monkeypatch):
    """Isolate tests from a developer/test-runner shell that exports
    AGENTKNIT_RESUME_COMMAND (a supported wrapper override)."""
    monkeypatch.delenv("AGENTKNIT_RESUME_COMMAND", raising=False)


def test_resume_command_includes_model_for_framework_cli(monkeypatch):
    monkeypatch.setattr(core.sys, "argv", ["agent-probe"])
    assert (
        _build_resume_cmd("test/model", "abc123")
        == "agent-probe test/model --session abc123"
    )


def test_resume_command_drops_model_for_wrapper_executable(monkeypatch):
    # A wrapper script pins its model; the resume hint must be the wrapper
    # itself, without the model repeated as a positional argument.
    monkeypatch.setattr(core.sys, "argv", ["/home/martin/bin/agent-glm-5.2.py"])
    assert (
        _build_resume_cmd("glm-5.2", "4e55afe7c1ce")
        == "/home/martin/bin/agent-glm-5.2.py --session 4e55afe7c1ce"
    )


def test_resume_command_uses_wrapper_override_without_model(monkeypatch):
    monkeypatch.setattr(core.sys, "argv", ["unused"])
    monkeypatch.setenv(
        "AGENTKNIT_RESUME_COMMAND",
        "/home/martin/bin/agent-deepseek-v4-flash-zen.py",
    )
    assert (
        _build_resume_cmd("/home/martin/bin/opencode-free-deepseek-v4-flash-completions.py", "cb2c8ce8e1c6")
        == "/home/martin/bin/agent-deepseek-v4-flash-zen.py --session cb2c8ce8e1c6"
    )


def test_is_framework_cli_matches_known_names():
    assert _is_framework_cli("agent-probe")
    assert _is_framework_cli("/home/martin/.local/bin/agentknit")
    assert not _is_framework_cli("/home/martin/bin/agent-glm-5.2.py")
    assert not _is_framework_cli("")


def test_is_framework_cli_matches_module_entry_point():
    assert _is_framework_cli(core.__file__)


def test_include_model_can_be_forced(monkeypatch):
    monkeypatch.setattr(core.sys, "argv", ["/home/martin/bin/agent-glm-5.2.py"])
    assert (
        _build_resume_cmd("glm-5.2", "abc", include_model=True)
        == "/home/martin/bin/agent-glm-5.2.py glm-5.2 --session abc"
    )
    monkeypatch.setattr(core.sys, "argv", ["agent-probe"])
    assert (
        _build_resume_cmd("test/model", "abc", include_model=False)
        == "agent-probe --session abc"
    )


def test_env_override_wins_over_include_model(monkeypatch):
    monkeypatch.setenv("AGENTKNIT_RESUME_COMMAND", "/usr/local/bin/agent-zen")
    assert _build_resume_cmd("m", "abc", include_model=True) == (
        "/usr/local/bin/agent-zen --session abc"
    )


def test_env_override_is_honored_when_set(monkeypatch):
    # The env var is a supported override: when present it must win, even if
    # the surrounding shell (e.g. a wrapper-based test runner) defines it.
    monkeypatch.setattr(core.sys, "argv", ["agent-probe"])
    monkeypatch.setenv("AGENTKNIT_RESUME_COMMAND", "/opt/bin/agent-custom")
    assert (
        _build_resume_cmd("m", "abc")
        == "/opt/bin/agent-custom --session abc"
    )
