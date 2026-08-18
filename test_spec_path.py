"""Tests for the ``spec_path`` parameter of ``load_specification`` (issue #13)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentknit import AgentSpecInvalidError, load_specification


def _write_spec(tmp_path: Path, name: str = "agent_spec.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps({
        "model": "deepseek-v4-flash-free",
        "endpoint": "https://example.test/v1",
        "tool_specs": [{"type": "function",
                        "function": {"name": "read_file", "parameters": {}}}],
        "tools": ["t_read"],
    }))
    return path


def test_spec_path_loads_file_directly(tmp_path, capsys) -> None:
    path = _write_spec(tmp_path)
    schema = load_specification("deepseek-v4-flash-free", "https://example.test/v1",
                                spec_path=str(path))
    assert schema["model"] == "deepseek-v4-flash-free"
    assert schema["tool_specs"][0]["function"]["name"] == "read_file"


def test_spec_path_skips_name_based_lookup(tmp_path, monkeypatch) -> None:
    """spec_path wins even when a same-named cached spec exists elsewhere."""
    path = _write_spec(tmp_path)
    # A spec named after the model in the cwd would otherwise be picked up.
    (tmp_path / "agent_spec_deepseek-v4-flash-free.json").write_text(
        json.dumps({"model": "wrong", "endpoint": "e", "tool_specs": [], "tools": []}))
    monkeypatch.chdir(tmp_path)
    schema = load_specification("deepseek-v4-flash-free", "", spec_path=str(path))
    assert schema["model"] == "deepseek-v4-flash-free"


def test_spec_path_composes_with_run_uri(tmp_path) -> None:
    """The issue's use case: run:// endpoint + spec at an arbitrary location."""
    path = _write_spec(tmp_path)
    schema = load_specification(
        "deepseek-v4-flash-free",
        "run:///home/user/bin/completions.py",
        spec_path=str(path),
    )
    # The spec file is returned verbatim — no in-memory default, no run:// rewrite.
    assert schema["endpoint"] == "https://example.test/v1"
    assert schema["tool_specs"][0]["function"]["name"] == "read_file"


def test_spec_path_relative_to_cwd(tmp_path, monkeypatch) -> None:
    _write_spec(tmp_path)
    monkeypatch.chdir(tmp_path)
    schema = load_specification("m", "e", spec_path="agent_spec.json")
    assert schema["model"] == "deepseek-v4-flash-free"


def test_spec_path_expands_user(monkeypatch, tmp_path) -> None:
    fake_home = tmp_path / "home"
    (fake_home / ".config" / "agentknit").mkdir(parents=True)
    _write_spec(fake_home / ".config" / "agentknit")
    monkeypatch.setenv("HOME", str(fake_home))
    schema = load_specification("m", "e",
                                spec_path="~/.config/agentknit/agent_spec.json")
    assert schema["model"] == "deepseek-v4-flash-free"


def test_spec_path_missing_file_raises_typed_error(tmp_path) -> None:
    with pytest.raises(AgentSpecInvalidError, match="not found"):
        load_specification("m", "e", spec_path=str(tmp_path / "nope.json"))


def test_spec_path_invalid_json_raises_typed_error(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(AgentSpecInvalidError, match="not valid JSON"):
        load_specification("m", "e", spec_path=str(bad))


def test_without_spec_path_behaviour_unchanged(tmp_path, monkeypatch) -> None:
    """Default in-memory spec is still returned for an unknown model + endpoint."""
    monkeypatch.chdir(tmp_path)
    schema = load_specification("unknown-model", "https://example.test/v1")
    assert schema["status"] == "default"


def test_cli_accepts_spec_path_flag() -> None:
    import agentknit._core as core
    # parse_args reads sys.argv; exercise it directly.
    import sys
    old = sys.argv
    sys.argv = ["agentknit", "some-model", "--spec-path", "/tmp/spec.json", "do", "it"]
    try:
        parsed = core.parse_args()
    finally:
        sys.argv = old
    assert parsed.spec_path == "/tmp/spec.json"
    assert parsed.model == "some-model"
    assert parsed.task == ["do", "it"]
