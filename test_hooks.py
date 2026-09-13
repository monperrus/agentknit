"""Tests for the hook infrastructure (plan-hooks.md §6).

Covers:

1. the normalization contract — the compatibility corpus of stdout/exit-code
   combinations from the Claude Code and Codex docs
2. strict script ≡ Python symmetry — the same hook logic run as a subprocess
   and as an in-process function must yield identical HookDecisions
3. the matcher matrix
4. config parsing (drop-in of a full Claude/Codex settings file)
5. combination precedence
6. integration with the agent loop (PreToolUse deny/rewrite, PostToolUse
   replacement, UserPromptSubmit block, Stop continuation, SessionEnd)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from agentknit import hooks as H


# ── helpers ───────────────────────────────────────────────────────────────────

def _cmd(script: str) -> H.HookHandler:
    """Build a command handler from a python -c script body."""
    prologue = "import sys, json"
    return H.HookHandler(
        type="command",
        command=f"{sys.executable} -c {prologue + '; ' + script!r}")


def _py(fn) -> H.HookHandler:
    return H.HookHandler(type="python", fn=fn)


def _run(handler: H.HookHandler, event: str = "PreToolUse",
         payload: dict | None = None) -> H.HookDecision:
    from agentknit.hooks import _resolve_timeout
    timeout = _resolve_timeout(handler, event)
    if handler.type == "command":
        raw = H.run_command_hook(handler, payload or {}, None, timeout)
    else:
        raw = H.run_python_hook(handler.fn, payload or {}, timeout)
    return H.normalize(raw, event)


SCRIPTS = {
    # name → (command script body, equivalent python function)
    "silent": ("pass", lambda p: None),
    "allow_json": (
        "print(json.dumps({'hookSpecificOutput': {'hookEventName': 'PreToolUse', "
        "'permissionDecision': 'allow'}}))",
        lambda p: {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                          "permissionDecision": "allow"}},
    ),
    "deny_json": (
        "print(json.dumps({'hookSpecificOutput': {'hookEventName': 'PreToolUse', "
        "'permissionDecision': 'deny', 'permissionDecisionReason': 'no'}}))",
        lambda p: {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                          "permissionDecision": "deny",
                                          "permissionDecisionReason": "no"}},
    ),
    "exit2": (
        "import sys; sys.stderr.write('blocked reason\\n'); sys.exit(2)",
        None,  # expressed via HookBlock below
    ),
    "additional_context": (
        "print(json.dumps({'hookSpecificOutput': {'hookEventName': 'PreToolUse', "
        "'additionalContext': 'note'}}))",
        lambda p: {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                          "additionalContext": "note"}},
    ),
    "updated_input": (
        "print(json.dumps({'hookSpecificOutput': {'hookEventName': 'PreToolUse', "
        "'permissionDecision': 'allow', 'updatedInput': {'command': 'ls'}}}))",
        lambda p: {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                          "permissionDecision": "allow",
                                          "updatedInput": {"command": "ls"}}},
    ),
    "invalid_json": ("print('this is not json')", lambda p: "plain text"),
    "exit1": ("import sys; sys.exit(1)", None),
}


def _decision_dict(d: H.HookDecision) -> dict:
    return {k: v for k, v in vars(d).items() if v is not None and v is not False}


# ── 1. compatibility corpus (docs examples) ──────────────────────────────────

def test_exit0_silent_is_no_decision() -> None:
    assert _decision_dict(_run(_cmd(SCRIPTS["silent"][0]))) == {}


def test_exit0_json_allow() -> None:
    d = _run(_cmd(SCRIPTS["allow_json"][0]))
    assert d.permission_decision == "allow"
    assert not d.block and d.error is None


def test_exit0_json_deny() -> None:
    d = _run(_cmd(SCRIPTS["deny_json"][0]))
    assert d.permission_decision == "deny"
    assert d.permission_decision_reason == "no"
    assert not d.block  # deny comes from the JSON, not exit 2


def test_exit2_blocks_with_stderr_reason() -> None:
    d = _run(_cmd(SCRIPTS["exit2"][0]))
    assert d.block is True
    assert d.reason == "blocked reason"
    assert d.permission_decision == "deny"


def test_exit2_json_cannot_unblock() -> None:
    d = _run(_cmd(
        "import sys, json; "
        "print(json.dumps({'hookSpecificOutput': {'hookEventName': 'PreToolUse', "
        "'permissionDecision': 'allow'}})); sys.exit(2)"))
    assert d.block is True


def test_exit1_is_nonblocking_error() -> None:
    d = _run(_cmd(SCRIPTS["exit1"][0]))
    assert d.error is not None
    assert not d.block


def test_invalid_json_is_nonblocking_error() -> None:
    d = _run(_cmd(SCRIPTS["invalid_json"][0]))
    # On PreToolUse plain text is ignored, not an error (docs: "plain text on
    # stdout is ignored"); it must never block.
    assert not d.block
    assert d.additional_context is None


def test_additional_context() -> None:
    d = _run(_cmd(SCRIPTS["additional_context"][0]))
    assert d.additional_context == "note"


def test_updated_input_only_with_allow() -> None:
    d = _run(_cmd(SCRIPTS["updated_input"][0]))
    assert d.updated_input == {"command": "ls"}
    # without allow → validation error (Codex rule)
    d2 = _run(_cmd(
        "print(json.dumps({'hookSpecificOutput': {'hookEventName': 'PreToolUse', "
        "'updatedInput': {'command': 'ls'}}}))"))
    assert d2.error is not None
    assert d2.updated_input is None


def test_plain_stdout_context_on_user_prompt_submit() -> None:
    d = _run(_cmd("print('extra dev context')"), event="UserPromptSubmit")
    assert d.additional_context == "extra dev context"


def test_plain_stdout_ignored_on_pretooluse() -> None:
    d = _run(_cmd("print('noise')"), event="PreToolUse")
    assert d.additional_context is None
    assert d.error is None  # ignored, not an error


def test_plain_stdout_invalid_on_stop() -> None:
    d = _run(_cmd("print('not json')"), event="Stop")
    assert d.error is not None


def test_stop_block_continues() -> None:
    d = _run(_cmd(
        "print(json.dumps({'decision': 'block', 'reason': 'keep going'}))"),
        event="Stop")
    assert d.block is True
    assert d.reason == "keep going"


def test_stop_expects_json_on_stdout() -> None:
    # exit 0 with JSON
    d = _run(_cmd(
        "print(json.dumps({'decision': 'block', 'reason': 'r'}))"), event="Stop")
    assert d.block
    # exit 2 blocks with stderr reason
    d2 = _run(_cmd("import sys; sys.stderr.write('why\\n'); sys.exit(2)"),
              event="Stop")
    assert d2.block and d2.reason == "why"


def test_session_end_advisory_exit2() -> None:
    d = _run(_cmd("import sys; sys.stderr.write('bye\\n'); sys.exit(2)"),
             event="SessionEnd")
    assert not d.block
    assert d.system_message == "bye"


def test_continue_false_sets_stop() -> None:
    d = _run(_cmd(
        "print(json.dumps({'continue': False, 'stopReason': 'halt'}))"),
        event="Stop")
    assert d.stop is True and d.stop_reason == "halt"


def test_continue_not_supported_on_pretooluse() -> None:
    d = _run(_cmd("print(json.dumps({'continue': False}))"), event="PreToolUse")
    assert d.error is not None


def test_wrong_hookenventname_is_validation_error() -> None:
    d = _run(_cmd(
        "print(json.dumps({'hookSpecificOutput': {'hookEventName': 'PostToolUse', "
        "'permissionDecision': 'deny'}}))"), event="PreToolUse")
    assert d.error is not None
    assert d.permission_decision is None


def test_deprecated_decision_approve_maps_to_allow() -> None:
    d = _run(_cmd("print(json.dumps({'decision': 'approve'}))"),
             event="PreToolUse")
    assert d.permission_decision == "allow"


def test_deprecated_decision_block_maps_to_deny() -> None:
    d = _run(_cmd("print(json.dumps({'decision': 'block', 'reason': 'r'}))"),
             event="PreToolUse")
    assert d.permission_decision == "deny" and d.block


def test_posttooluse_updated_tool_output() -> None:
    d = _run(_cmd(
        "print(json.dumps({'hookSpecificOutput': {'hookEventName': 'PostToolUse', "
        "'updatedToolOutput': 'replaced'}}))"), event="PostToolUse")
    assert d.updated_tool_output == "replaced"


def test_hook_specific_output_rejected_on_wrong_event() -> None:
    d = _run(_cmd(
        "print(json.dumps({'hookSpecificOutput': {'hookEventName': 'SessionEnd', "
        "'updatedToolOutput': 'x'}}))"), event="SessionEnd")
    assert d.error is not None


# ── 2. strict script ≡ Python symmetry ───────────────────────────────────────

@pytest.mark.parametrize("name", ["silent", "allow_json", "deny_json",
                                  "additional_context", "updated_input",
                                  "invalid_json"])
def test_script_and_python_hooks_agree(name: str) -> None:
    script_body, py_fn = SCRIPTS[name]
    via_script = _run(_cmd(script_body))
    via_python = _run(_py(py_fn))
    assert _decision_dict(via_script) == _decision_dict(via_python)


def test_hookblock_equals_exit2() -> None:
    def block(p):
        raise H.HookBlock("blocked reason")

    via_script = _run(_cmd(SCRIPTS["exit2"][0]))
    via_python = _run(_py(block))
    assert _decision_dict(via_script) == _decision_dict(via_python)


def test_python_hook_exception_is_exit1_equivalent() -> None:
    def boom(p):
        raise RuntimeError("kaput")

    via_script = _run(_cmd(SCRIPTS["exit1"][0]))
    via_python = _run(_py(boom))
    assert (via_script.error is not None) == (via_python.error is not None)
    assert not via_script.block and not via_python.block


def test_python_hook_string_return_is_plain_stdout() -> None:
    via_script = _run(_cmd("print('plain')"), event="UserPromptSubmit")
    via_python = _run(_py(lambda p: "plain"), event="UserPromptSubmit")
    assert via_script.additional_context == via_python.additional_context


def test_python_hook_timeout() -> None:
    import time

    def slow(p):
        time.sleep(5)

    handler = H.HookHandler(type="python", fn=slow, timeout=0.3)
    d = _run(handler)
    assert d.error is not None and "timed out" in d.error


def test_command_hook_timeout() -> None:
    handler = H.HookHandler(type="command",
                            command=f"{sys.executable} -c 'import time; time.sleep(5)'",
                            timeout=0.3)
    d = _run(handler)
    assert d.error is not None and "timed out" in d.error


def test_command_hook_receives_payload_on_stdin() -> None:
    handler = H.HookHandler(
        type="command",
        command=(f"{sys.executable} -c "
                 "'import sys, json; p = json.load(sys.stdin); "
                 "print(json.dumps({\"hookSpecificOutput\": "
                 "{\"hookEventName\": \"PreToolUse\", \"permissionDecision\": \"deny\", "
                 "\"permissionDecisionReason\": p[\"tool_name\"]}}))'"))
    d = _run(handler, payload={"tool_name": "Bash", "tool_input": {"command": "ls"}})
    assert d.permission_decision == "deny"
    assert d.permission_decision_reason == "Bash"


def test_command_hook_timeout_session_end_capped() -> None:
    from agentknit.hooks import _resolve_timeout
    h = H.HookHandler(type="command", command="true", timeout=99)
    assert _resolve_timeout(h, "SessionEnd") == 3.0
    assert _resolve_timeout(h, "PreToolUse") == 99.0


# ── 3. matcher matrix ────────────────────────────────────────────────────────

@pytest.mark.parametrize("matcher,value,expected", [
    ("", "anything", True),
    ("*", "anything", True),
    ("Bash", "Bash", True),
    ("Bash", "Edit", False),
    ("Edit|Write", "Write", True),
    ("Edit, Write", "Write", True),
    ("Edit, Write", "Bash", False),
    ("Edit, Write", " Edit ", False),       # value whitespace is significant
    ("code-reviewer", "code-reviewer", True),  # hyphens stay exact-match
    ("mcp__memory__.*", "mcp__memory__create", True),
    ("mcp__memory", "mcp__memory__create", False),  # bare prefix is exact
    ("^Notebook", "NotebookEdit", True),
    ("^Notebook$", "NotebookEdit", False),
    ("Edit.*", "NotebookEdit", True),          # unanchored
    ("([", "x", False),                        # broken regex → no match, no raise
])
def test_matcher_matrix(matcher: str, value: str, expected: bool) -> None:
    assert H.matcher_matches(matcher, value) is expected


def test_entry_matches_both_canonical_and_native_names() -> None:
    entry = H.HookEntry(event="PreToolUse", matcher="Bash",
                        handler=_cmd("pass"))
    assert entry.matches(["Bash", "exec_shell"])
    entry2 = H.HookEntry(event="PreToolUse", matcher="Edit",
                         handler=_cmd("pass"))
    assert entry2.matches(["Edit", "str_replace"])


def test_canonical_tool_name() -> None:
    assert H.canonical_tool_name("exec_shell") == "Bash"
    assert H.canonical_tool_name("str_replace") == "Edit"
    assert H.canonical_tool_name("read_file") == "Read"
    assert H.canonical_tool_name("my_custom_tool") == "my_custom_tool"


def test_translate_updated_input() -> None:
    assert H.translate_updated_input({
        "file_path": "/tmp/x", "old_string": "a", "new_string": "b",
        "description": "meta", "command": "ls",
    }) == {"path": "/tmp/x", "old_str": "a", "new_str": "b", "command": "ls"}


# ── 4. config parsing ────────────────────────────────────────────────────────

FULL_CLAUDE_SETTINGS = {
    "hooks": {
        "PreToolUse": [
            {"matcher": "Bash",
             "hooks": [{"type": "command", "command": "/bin/check.sh",
                        "timeout": 30}]},
            {"matcher": "mcp__.*",
             "hooks": [{"type": "command", "command": "echo mcp"}]},
        ],
        "PostToolUse": [
            {"matcher": "Edit|Write",
             "hooks": [{"type": "command", "command": "/bin/lint.sh"}]},
        ],
        # Claude-only event agentknit never fires — must load, not crash.
        "Notification": [
            {"hooks": [{"type": "command", "command": "notify-send hi"}]},
        ],
        "SubagentStop": [
            {"hooks": [{"type": "command", "command": "echo sub"}]},
        ],
    },
}


def test_full_claude_settings_drops_in() -> None:
    entries, warnings = H.parse_hooks_config(FULL_CLAUDE_SETTINGS)
    events = {e.event for e in entries}
    assert events == {"PreToolUse", "PostToolUse", "Notification", "SubagentStop"}
    # Both are known-but-inert: they load (no warning) yet never dispatch.
    assert "SubagentStop" in H.HOOKS_NEVER_FIRE
    assert "Notification" in H.HOOKS_NEVER_FIRE
    # Unknown *event names* warn; Notification is known-but-inert, no warning.
    assert warnings == []


def test_unknown_event_warns_and_is_skipped() -> None:
    entries, warnings = H.parse_hooks_config(
        {"hooks": {"SomeFutureEvent": [
            {"hooks": [{"type": "command", "command": "x"}]}]}})
    assert entries == []
    assert any("SomeFutureEvent" in w for w in warnings)


def test_unsupported_handler_types_warn() -> None:
    entries, warnings = H.parse_hooks_config({"hooks": {"PreToolUse": [
        {"matcher": "Bash",
         "hooks": [{"type": "mcp_tool", "server": "s", "tool": "t"},
                   {"type": "prompt", "prompt": "p"}]}]}})
    assert entries == []
    assert len([w for w in warnings if "not supported" in w]) == 2


def test_hooks_json_file_roundtrip(tmp_path: Path) -> None:
    cfg = tmp_path / "hooks.json"
    cfg.write_text(json.dumps({
        "hooks": {"Stop": [
            {"hooks": [{"type": "command", "command": "echo stop",
                        "timeout": 5, "async": False}]}]}}))
    entries, warnings = H.parse_hooks_config(str(cfg))
    assert warnings == []
    assert len(entries) == 1
    e = entries[0]
    assert e.event == "Stop" and e.handler.command == "echo stop"
    assert e.handler.timeout == 5.0


def test_missing_file_is_quiet_warning() -> None:
    entries, warnings = H.parse_hooks_config(Path("/nonexistent/hooks.json"))
    assert entries == []
    assert any("not found" in w for w in warnings)


def test_exec_form_args() -> None:
    handler = H.HookHandler(type="command", command=sys.executable,
                            args=["-c", "print(1)"])
    raw = H.run_command_hook(handler, {}, None, 10)
    assert raw.rc == 0 and raw.stdout.strip() == "1"


def test_list_of_sources_merges(tmp_path: Path) -> None:
    a = tmp_path / "a.json"
    a.write_text(json.dumps({"hooks": {"Stop": [
        {"hooks": [{"type": "command", "command": "a"}]}]}}))
    b = {"hooks": {"PreToolUse": [
        {"matcher": "Bash",
         "hooks": [{"type": "command", "command": "b"}]}]}}
    entries, _ = H.parse_hooks_config([a, b])
    assert sorted((e.event, e.handler.command) for e in entries) == [
        ("PreToolUse", "b"), ("Stop", "a")]


# ── 5. combination ───────────────────────────────────────────────────────────

def _d(**kw) -> H.HookDecision:
    return H.HookDecision(**kw)


def test_deny_beats_ask_beats_allow() -> None:
    combined = H.combine_decisions([
        _d(permission_decision="allow", updated_input={"command": "x"}),
        _d(permission_decision="ask"),
        _d(permission_decision="deny", permission_decision_reason="no"),
    ])
    assert combined.permission_decision == "deny"
    assert combined.permission_decision_reason == "no"
    assert combined.updated_input is None  # from the losing hook


def test_block_counts_as_deny_in_combination() -> None:
    combined = H.combine_decisions([
        _d(permission_decision="allow"),
        _d(block=True, reason="stop it", permission_decision="deny"),
    ])
    assert combined.block and combined.permission_decision == "deny"


def test_continue_false_wins() -> None:
    combined = H.combine_decisions([
        _d(permission_decision="deny"),
        _d(stop=True, stop_reason="halt"),
    ])
    assert combined.stop and combined.stop_reason == "halt"
    assert combined.permission_decision is None


def test_context_and_messages_accumulate() -> None:
    combined = H.combine_decisions([
        _d(additional_context="one"), _d(additional_context="two"),
        _d(system_message="warn"),
    ])
    assert combined.additional_context == "one\ntwo"
    assert combined.system_message == "warn"


# ── spill ────────────────────────────────────────────────────────────────────

def test_cap_model_text_noop_under_limit() -> None:
    assert H.cap_model_text("short", 100, None) == "short"


def test_cap_model_text_spills_to_disk(tmp_path: Path) -> None:
    text = "x" * (H.HOOK_CONTEXT_LIMIT + 5)
    out = H.cap_model_text(text, H.HOOK_CONTEXT_LIMIT, tmp_path)
    assert "full hook output saved to" in out
    spilled = list((tmp_path / "hook_outputs").glob("*.txt"))
    assert len(spilled) == 1
    assert spilled[0].read_text() == text


def test_cap_model_text_truncates_without_dir() -> None:
    text = "x" * 100
    out = H.cap_model_text(text, 10, None)
    assert "omitted" in out and "saved to" not in out


# ── 6. run_hooks dispatch ────────────────────────────────────────────────────

def test_run_hooks_no_match_is_empty() -> None:
    entries = [H.HookEntry(event="PreToolUse", matcher="Bash",
                           handler=_cmd("pass"))]
    d = H.run_hooks(entries, "PreToolUse", {}, matcher_values=["Edit"])
    assert _decision_dict(d) == {}


def test_run_hooks_combines_concurrent_matches() -> None:
    entries = [
        H.HookEntry(event="PreToolUse", matcher="Bash",
                    handler=_cmd(SCRIPTS["deny_json"][0])),
        H.HookEntry(event="PreToolUse", matcher="",
                    handler=_cmd(SCRIPTS["additional_context"][0])),
    ]
    d = H.run_hooks(entries, "PreToolUse", {}, matcher_values=["Bash"])
    assert d.permission_decision == "deny"
    assert d.additional_context == "note"


def test_async_hook_decision_discarded(tmp_path: Path) -> None:
    state = H.new_hook_state()
    entries = [H.HookEntry(event="PreToolUse", matcher="",
                           handler=H.HookHandler(
                               type="command",
                               command=(f"{sys.executable} -c "
                                        "'import json; print(json.dumps({"
                                        "\"hookSpecificOutput\": {\"hookEventName\": "
                                        "\"PreToolUse\", \"additionalContext\": "
                                        "\"late\"}}))'"),
                               async_=True))]
    d = H.run_hooks(entries, "PreToolUse", {}, matcher_values=["Bash"],
                    state=state, spill_dir=tmp_path)
    assert _decision_dict(d) == {}  # async hooks never block their dispatch point
    # the informational output lands in the queue once the thread finishes
    import time
    for _ in range(100):
        if state["async_results"]:
            break
        time.sleep(0.05)
    assert state["async_results"][0]["context"] == "late"


def test_register_hook_requires_fn_or_command() -> None:
    with pytest.raises(ValueError):
        H.register_hook({}, "PreToolUse")
    with pytest.raises(ValueError):
        H.register_hook({}, "NotAnEvent", fn=lambda p: None)


def test_register_hook_python_and_load() -> None:
    session: dict = {}
    H.register_hook(session, "Stop", lambda p: {"decision": "block",
                                                "reason": "more"})
    assert len(session["hooks"]) == 1
    d = H.run_hooks(session["hooks"], "Stop", {})
    assert d.block and d.reason == "more"
