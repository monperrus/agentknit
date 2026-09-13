"""Integration tests: hooks wired into the agent loop (plan-hooks.md §6.5)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import agentknit._core as core
from agentknit._core import _handle_tool_call, init_session
from agentknit import hooks as H


# ── fixtures ──────────────────────────────────────────────────────────────────

def _schema(**extra: Any) -> dict:
    return {
        "model": "test/model",
        "endpoint": "https://api.example.com/v1",
        "inferred_tool_schema": [],
        "tool_dispatch": {
            "exec_shell": {"python_function": "t_run", "param_map": {}},
            "read_file": {"python_function": "t_read", "param_map": {}},
        },
        **extra,
    }


def _session(tmp_path: Path, monkeypatch, hooks=None, **extra: Any) -> dict:
    monkeypatch.setattr(core, "LOG_BASE", tmp_path)
    monkeypatch.chdir(tmp_path)
    session = init_session(_schema(**extra), strict_cache_proof=False,
                           hooks=hooks)
    return session


class _Choice:
    def __init__(self, message) -> None:
        self.message = message


class _Response:
    def __init__(self, choices, usage=None) -> None:
        self.choices = choices
        self.usage = usage


class _Function:
    def __init__(self, name, arguments) -> None:
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, id, function) -> None:  # noqa: A002
        self.id = id
        self.function = function
        self.type = "function"


class _Message:
    def __init__(self, role, content, tool_calls=None) -> None:
        self.role = role
        self.content = content
        self.tool_calls = tool_calls


class _Usage:
    def __init__(self) -> None:
        self.prompt_tokens = 10
        self.completion_tokens = 5
        self.total_tokens = 15
        self.cached_tokens = 0
        self.cache_creation_tokens = 0


class _ScriptedClient:
    """Yields scripted responses; records requests for assertions."""

    def __init__(self, responses) -> None:
        self._responses = list(responses)
        self.requests: list[dict] = []

        class _URL:
            host = "api.example.com"

        self.base_url = _URL()

        class _Completions:
            def create(inner_self, **kwargs):
                self.requests.append(kwargs)
                return self._responses.pop(0)

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def _tool_call_response(call_id: str, name: str, args: dict) -> _Response:
    call = _ToolCall(call_id, _Function(name, json.dumps(args)))
    return _Response([_Choice(_Message("assistant", None, [call]))])


def _text_response(text: str) -> _Response:
    return _Response([_Choice(_Message("assistant", text))], _Usage())


# ── PreToolUse ────────────────────────────────────────────────────────────────

def test_pretooluse_deny_blocks_tool(tmp_path, monkeypatch) -> None:
    session = _session(tmp_path, monkeypatch)
    H.register_hook(session, "PreToolUse",
                    lambda p: {"hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": "dangerous"}},
                    matcher="Bash")
    result = _handle_tool_call("exec_shell", {"command": "rm -rf /"},
                               session, call_id="c1")
    assert "denied by hook" in result
    assert "dangerous" in result
    # the model-facing history records the denial as the tool result
    tool_msgs = [m for m in session["messages"] if m.get("role") == "tool"]
    assert tool_msgs == []  # _handle_tool_call returns; caller appends


def test_pretooluse_payload_fields(tmp_path, monkeypatch) -> None:
    session = _session(tmp_path, monkeypatch)
    seen: list[dict] = []
    H.register_hook(session, "PreToolUse", lambda p: seen.append(p) or None,
                    matcher="Bash")
    _handle_tool_call("exec_shell", {"command": "ls"}, session, call_id="c1")
    payload = seen[0]
    assert payload["tool_name"] == "Bash"
    assert payload["agentknit_tool_name"] == "exec_shell"
    assert payload["tool_input"] == {"command": "ls"}
    assert payload["tool_use_id"] == "c1"
    assert payload["hook_event_name"] == "PreToolUse"
    assert payload["session_id"] == session["session_id"]
    assert payload["cwd"] == str(Path(tmp_path))
    assert payload["model"] == "test/model"


def test_pretooluse_rewrite_updated_input(tmp_path, monkeypatch) -> None:
    session = _session(tmp_path, monkeypatch)
    calls: list[dict] = []

    import agentknit.tool_library as tl
    original = tl.TOOL_LIBRARY["t_run"]

    def spy_run(command: str = "") -> tuple[str, dict]:
        calls.append({"command": command})
        return "ran", {"result": "ran"}

    monkeypatch.setitem(tl.TOOL_LIBRARY, "t_run", spy_run)
    try:
        H.register_hook(session, "PreToolUse",
                        lambda p: {"hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "allow",
                            # Claude Bash shape: rewritten command + metadata
                            "updatedInput": {"command": "echo rewritten",
                                             "description": "meta"}}},
                        matcher="Bash")
        result = _handle_tool_call("exec_shell", {"command": "ls"}, session,
                                   call_id="c1")
        assert calls == [{"command": "echo rewritten"}]
        assert result == "ran"
    finally:
        tl.TOOL_LIBRARY["t_run"] = original


def test_pretooluse_translate_file_aliases(tmp_path, monkeypatch) -> None:
    session = _session(tmp_path, monkeypatch)
    calls: list[dict] = []

    import agentknit.tool_library as tl
    original = tl.TOOL_LIBRARY["t_read"]

    def spy_read(path: str = "") -> tuple[str, dict]:
        calls.append({"path": path})
        return "content", {"result": "content"}

    monkeypatch.setitem(tl.TOOL_LIBRARY, "t_read", spy_read)
    try:
        H.register_hook(session, "PreToolUse",
                        lambda p: {"hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "allow",
                            "updatedInput": {"file_path": "/tmp/x"}}},
                        matcher="Read")
        _handle_tool_call("read_file", {"path": "/tmp/other"}, session,
                          call_id="c1")
        assert calls == [{"path": "/tmp/x"}]
    finally:
        tl.TOOL_LIBRARY["t_read"] = original


def test_pretooluse_command_hook_script(tmp_path, monkeypatch) -> None:
    """A real ~/.local-style script behaves identically to the Python API."""
    script = tmp_path / "guard.sh"
    script.write_text(textwrap_dedent())
    session = _session(tmp_path, monkeypatch)
    H.register_hook(session, "PreToolUse", command=f"{sys.executable} {script}",
                    matcher="Bash")
    result = _handle_tool_call("exec_shell", {"command": "rm -rf /"}, session,
                               call_id="c1")
    assert "denied by hook" in result
    assert "destructive" in result


def textwrap_dedent() -> str:
    return """#!/usr/bin/env python3
import json, sys
payload = json.load(sys.stdin)
if payload["tool_input"].get("command", "").startswith("rm -rf"):
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": "destructive command"}}))
"""


# ── PostToolUse ───────────────────────────────────────────────────────────────

def test_posttooluse_replaces_result(tmp_path, monkeypatch) -> None:
    session = _session(tmp_path, monkeypatch)
    H.register_hook(session, "PostToolUse",
                    lambda p: {"hookSpecificOutput": {
                        "hookEventName": "PostToolUse",
                        "updatedToolOutput": "sanitized"}},
                    matcher="Bash")
    result = _handle_tool_call("exec_shell", {"command": "echo hi"}, session,
                               call_id="c1")
    assert result == "sanitized"


def test_posttooluse_block_replaces_result_with_reason(tmp_path, monkeypatch) -> None:
    session = _session(tmp_path, monkeypatch)
    H.register_hook(session, "PostToolUse",
                    lambda p: {"decision": "block",
                               "reason": "output needs review"},
                    matcher="Bash")
    result = _handle_tool_call("exec_shell", {"command": "echo hi"}, session,
                               call_id="c1")
    assert "output needs review" in result
    assert result.startswith("ERROR:")


# ── UserPromptSubmit / Stop / SessionStart / SessionEnd ──────────────────────

def test_userpromptsubmit_block_rejects_task(tmp_path, monkeypatch) -> None:
    session = _session(tmp_path, monkeypatch)
    H.register_hook(session, "UserPromptSubmit",
                    lambda p: {"decision": "block", "reason": "secrets in prompt"})
    client = _ScriptedClient([_text_response("should not be reached")])
    result = core.run_turn(client, "test/model", session, "my api key is ...")
    assert "secrets in prompt" in (result.final_reply or "")
    assert client.requests == []  # nothing was sent to the model


def test_stop_hook_continues_turn(tmp_path, monkeypatch) -> None:
    session = _session(tmp_path, monkeypatch)
    fired: list[bool] = []

    def stop_hook(payload: dict) -> dict | None:
        fired.append(payload["stop_hook_active"])
        if not fired[0]:
            return {"decision": "block", "reason": "run the tests first"}
        return None

    H.register_hook(session, "Stop", stop_hook)
    client = _ScriptedClient([
        _text_response("done"),           # first final answer
        _text_response("tests passed"),   # answer after continuation
    ])
    result = core.run_turn(client, "test/model", session, "ship it")
    assert fired == [False, True]  # guard: second fire sees stop_hook_active
    assert result.final_reply == "tests passed"
    # the continuation reason entered the transcript as a user message
    user_msgs = [m["content"] for m in session["messages"]
                 if m.get("role") == "user"]
    assert "run the tests first" in user_msgs


def test_sessionstart_context_in_system_prompt(tmp_path, monkeypatch) -> None:
    session = _session(tmp_path, monkeypatch, behaviour={
        "hooks": {"SessionStart": [
            {"matcher": "startup",
             "hooks": [{"type": "command",
                        "command": f"{sys.executable} -c 'print(\"workspace notes\")'"}]}]}})
    assert "workspace notes" in session["messages"][0]["content"]


def test_sessionstart_python_hook_same(tmp_path, monkeypatch) -> None:
    """Script and Python-API symmetry for SessionStart context: both forms
    land the same additionalContext in the system prompt."""
    payload_json = json.dumps({"hookSpecificOutput": {
        "hookEventName": "SessionStart", "additionalContext": "py notes"}})
    script_cmd = (f"{sys.executable} -c "
                  f"'import sys; print(sys.argv[1])' {payload_json!r}")
    session = _session(tmp_path, monkeypatch, behaviour={
        "hooks": {"SessionStart": [
            {"matcher": "startup",
             "hooks": [{"type": "command", "command": script_cmd}]}]}})
    assert "py notes" in session["messages"][0]["content"]

    # The Python twin: the same decision via the API on a live session.
    session2 = _session(tmp_path, monkeypatch)
    H.register_hook(session2, "SessionStart",
                    lambda p: {"hookSpecificOutput": {
                        "hookEventName": "SessionStart",
                        "additionalContext": "py notes"}})
    d = core._fire_hooks(session2, "SessionStart", matcher_values=["startup"],
                         source="startup")
    assert d.additional_context == "py notes"


def test_sessionend_fires_on_run_task(tmp_path, monkeypatch) -> None:
    session = _session(tmp_path, monkeypatch)
    seen: list[dict] = []
    H.register_hook(session, "SessionEnd", lambda p: seen.append(p) or None)
    client = _ScriptedClient([_text_response("ok")])
    core.run_turn(client, "test/model", session, "hi")
    core._fire_session_end(session, "other")
    assert [p["reason"] for p in seen] == ["other"]
    assert seen[0]["hook_event_name"] == "SessionEnd"


def test_hooks_enabled_false_kills_everything(tmp_path, monkeypatch) -> None:
    session = _session(tmp_path, monkeypatch)
    H.register_hook(session, "PreToolUse",
                    lambda p: {"hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny"}})
    session["hooks_enabled"] = False
    result = _handle_tool_call("exec_shell", {"command": "echo hi"}, session,
                               call_id="c1")
    assert "denied" not in result


def test_hook_warning_event_emitted(tmp_path, monkeypatch) -> None:
    session = _session(tmp_path, monkeypatch)
    warnings: list[str] = []
    core.subscribe(session, "hook_warning",
                   lambda et, d: warnings.append(d["text"]))
    H.register_hook(session, "PreToolUse",
                    lambda p: {"systemMessage": "heads up"})
    _handle_tool_call("exec_shell", {"command": "echo hi"}, session, call_id="c1")
    assert warnings == ["heads up"]


def test_pretooluse_error_fails_open(tmp_path, monkeypatch) -> None:
    session = _session(tmp_path, monkeypatch)

    def broken(payload):
        raise RuntimeError("hook bug")

    H.register_hook(session, "PreToolUse", broken)
    result = _handle_tool_call("exec_shell", {"command": "echo hi"}, session,
                               call_id="c1")
    assert "denied" not in result  # a broken hook never blocks


# ── spec behaviour.hooks discovery ────────────────────────────────────────────

def test_behaviour_hooks_inline_dict(tmp_path, monkeypatch) -> None:
    session = _session(tmp_path, monkeypatch, behaviour={
        "hooks": {"PostToolUse": [
            {"matcher": "Bash|Edit",
             "hooks": [{"type": "command", "command": "echo post"}]}]}})
    assert [e.event for e in session["hooks"]] == ["PostToolUse"]
    assert session["hooks"][0].matcher == "Bash|Edit"


def test_hooks_kwarg_path(tmp_path, monkeypatch) -> None:
    cfg = tmp_path / "hooks.json"
    cfg.write_text(json.dumps({"hooks": {"Stop": [
        {"hooks": [{"type": "command", "command": "echo stop"}]}]}}))
    session = _session(tmp_path, monkeypatch, hooks=str(cfg))
    assert [e.event for e in session["hooks"]] == ["Stop"]


def test_hook_events_in_repl_listing(tmp_path, monkeypatch, capsys) -> None:
    from agentknit.slash_commands import REGISTRY
    session = _session(tmp_path, monkeypatch)
    H.register_hook(session, "PreToolUse", lambda p: None, matcher="Bash")
    REGISTRY.dispatch("/hooks", session, object(), "test/model")
    out = capsys.readouterr().out
    assert "PreToolUse" in out and "Bash" in out
