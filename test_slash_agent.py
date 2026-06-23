"""Tests for slash_agent.py — slash commands exposed as LLM-callable tools."""
from __future__ import annotations

from agentknit import t_slash_command, slash_tool_ctx


def _make_session() -> dict:
    return {
        "messages": [{"role": "system", "content": "sys"}],
        "usage_totals": {"prompt": 1000, "completion": 200, "total": 1200,
                         "cached": 100, "cache_write": 0},
        "session_id": "test-session-id",
        "model": "test-model",
        "endpoint": "https://example.com/v1",
    }


def _wire_ctx(session: dict, client=None, model: str = "test-model") -> None:
    slash_tool_ctx["session"] = session
    slash_tool_ctx["client"] = client
    slash_tool_ctx["model"] = model


# ── t_slash_command: help ─────────────────────────────────────────────────────

def test_slash_command_help_returns_string():
    _wire_ctx(_make_session())
    result, meta = t_slash_command("help")
    assert isinstance(result, str)
    assert "clear" in result and "model" in result and "usage" in result
    assert meta["result"] == result


# ── t_slash_command: usage ────────────────────────────────────────────────────

def test_slash_command_usage_shows_tokens():
    _wire_ctx(_make_session())
    result, _ = t_slash_command("usage")
    assert "1,200" in result or "1200" in result


def test_slash_command_usage_shows_session_id():
    _wire_ctx(_make_session())
    result, _ = t_slash_command("usage")
    assert "test-session-id" in result


# ── t_slash_command: clear ────────────────────────────────────────────────────

def test_slash_command_clear_resets_history():
    session = _make_session()
    session["messages"].append({"role": "user", "content": "hello"})
    session["messages"].append({"role": "assistant", "content": "hi"})
    _wire_ctx(session)
    result, _ = t_slash_command("clear")
    assert "cleared" in result.lower() or "reset" in result.lower()
    assert all(m["role"] == "system" for m in session["messages"])


def test_slash_command_clear_resets_usage_totals():
    session = _make_session()
    _wire_ctx(session)
    t_slash_command("clear")
    assert session["usage_totals"]["total"] == 0


# ── t_slash_command: model ────────────────────────────────────────────────────

def test_slash_command_model_switch():
    session = _make_session()
    _wire_ctx(session)
    result, _ = t_slash_command("model", args="new-model/v2")
    assert "new-model/v2" in result
    assert session["model"] == "new-model/v2"


# ── t_slash_command: unknown command ─────────────────────────────────────────

def test_slash_command_unknown_returns_error():
    _wire_ctx(_make_session())
    result, _ = t_slash_command("nonexistent")
    assert result.startswith("ERROR:")


# ── schema ────────────────────────────────────────────────────────────────────

def test_single_slash_command_tool_in_schema():
    import slash_agent
    names = [t["function"]["name"] for t in slash_agent._TOOL_SCHEMA]
    assert "slash_command" in names
    for old in ("slash_clear", "slash_model", "slash_usage", "slash_help"):
        assert old not in names


def test_slash_command_schema_has_enum():
    import slash_agent
    tool = next(t for t in slash_agent._TOOL_SCHEMA
                if t["function"]["name"] == "slash_command")
    props = tool["function"]["parameters"]["properties"]
    assert set(props["command"]["enum"]) == {"clear", "model", "usage", "help"}


def test_standard_tools_in_schema():
    import slash_agent
    names = {t["function"]["name"] for t in slash_agent._TOOL_SCHEMA}
    for expected in ("read_file", "write_file", "run_shell", "edit_file"):
        assert expected in names


def test_all_tools_have_dispatch_entry():
    import slash_agent
    schema_names = {t["function"]["name"] for t in slash_agent._TOOL_SCHEMA}
    dispatch_names = set(slash_agent._TOOL_DISPATCH.keys())
    assert schema_names == dispatch_names


# ── dispatch integration ──────────────────────────────────────────────────────

def test_slash_command_dispatchable():
    import slash_agent
    from agentknit._core import dispatch
    _wire_ctx(_make_session())
    result, _ = dispatch("slash_command", {"command": "help"}, slash_agent._TOOL_DISPATCH)
    assert "clear" in result


def test_slash_command_usage_dispatchable():
    import slash_agent
    from agentknit._core import dispatch
    _wire_ctx(_make_session())
    result, _ = dispatch("slash_command", {"command": "usage"}, slash_agent._TOOL_DISPATCH)
    assert "1,200" in result or "1200" in result
