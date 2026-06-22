"""Tests that ~/.claude/CLAUDE.md is loaded into the system prompt."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from agentknit._core import init_session

_MINIMAL_SCHEMA = {
    "model": "test-model",
    "endpoint": "https://example.com",
    "inferred_tool_schema": [],
    "behaviour": {"call_delivery_mode": "structured_tool_calls"},
    "tool_dispatch": {},
}


def _sys_msg(schema: dict, **kwargs) -> str:
    session = init_session(schema, **kwargs)
    return session["messages"][0]["content"]


def test_claude_md_loaded_when_present(tmp_path: Path) -> None:
    claude_md = tmp_path / ".claude" / "CLAUDE.md"
    claude_md.parent.mkdir(parents=True)
    claude_md.write_text("# My instructions\nBe terse.")

    with patch.object(Path, "home", return_value=tmp_path):
        msg = _sys_msg(_MINIMAL_SCHEMA)

    assert "# My instructions" in msg
    assert "Be terse." in msg


def test_claude_md_absent_no_error(tmp_path: Path) -> None:
    with patch.object(Path, "home", return_value=tmp_path):
        msg = _sys_msg(_MINIMAL_SCHEMA)

    assert "# My instructions" not in msg


def test_claude_md_before_agents_md(tmp_path: Path) -> None:
    claude_md = tmp_path / ".claude" / "CLAUDE.md"
    claude_md.parent.mkdir(parents=True)
    claude_md.write_text("CLAUDE_CONTENT")

    agents_md = Path.cwd() / "AGENTS.md"
    agents_md_exists = agents_md.exists()
    agents_md_content: str | None = agents_md.read_text() if agents_md_exists else None

    agents_md.write_text("AGENTS_CONTENT")
    try:
        with patch.object(Path, "home", return_value=tmp_path):
            msg = _sys_msg(_MINIMAL_SCHEMA)
        assert msg.index("CLAUDE_CONTENT") < msg.index("AGENTS_CONTENT")
    finally:
        if agents_md_exists and agents_md_content is not None:
            agents_md.write_text(agents_md_content)
        else:
            agents_md.unlink()
