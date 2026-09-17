"""Directory-based hook discovery: dropping a script in is the registration."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from agentknit import discover_hook_dir, load_hooks
from agentknit.hooks import parse_hooks_config, run_hooks


def _script(path: Path, body: str = "#!/bin/sh\nexit 0\n", executable: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_missing_directory_is_not_an_error(tmp_path):
    entries, warnings = discover_hook_dir(tmp_path / "nope")
    assert entries == [] and warnings == []


def test_event_inferred_from_the_file_name(tmp_path):
    _script(tmp_path / "notify-stop.py")
    entries, warnings = discover_hook_dir(tmp_path)
    assert warnings == []
    assert [e.event for e in entries] == ["Stop"]
    assert entries[0].matcher == ""
    assert entries[0].handler.args == []  # exec form: no shell


def test_longest_event_suffix_wins(tmp_path):
    _script(tmp_path / "notify-subagent-stop.sh")
    entries, _ = discover_hook_dir(tmp_path)
    assert [e.event for e in entries] == ["SubagentStop"]


def test_exact_name_is_an_event(tmp_path):
    _script(tmp_path / "PreToolUse.sh")
    _script(tmp_path / "pre-compact.sh")
    entries, warnings = discover_hook_dir(tmp_path)
    assert warnings == []
    assert sorted(e.event for e in entries) == ["PreCompact", "PreToolUse"]


def test_event_directory_and_matcher_directory(tmp_path):
    _script(tmp_path / "Stop" / "notify.py")
    _script(tmp_path / "PreToolUse" / "Bash" / "guard.sh")
    entries, warnings = discover_hook_dir(tmp_path)
    assert warnings == []
    by_event = {e.event: e for e in entries}
    assert by_event["Stop"].matcher == ""
    assert by_event["PreToolUse"].matcher == "Bash"


def test_a_missing_event_name_warns_instead_of_registering(tmp_path):
    _script(tmp_path / "helpers.sh")
    entries, warnings = discover_hook_dir(tmp_path)
    assert entries == []
    assert "cannot infer a hook event" in warnings[0]


def test_a_non_executable_script_warns_instead_of_failing_with_127(tmp_path):
    path = _script(tmp_path / "Stop" / "notify.py", executable=False)
    path.chmod(0o644)
    entries, warnings = discover_hook_dir(tmp_path)
    assert entries == []
    assert "not executable" in warnings[0]


@pytest.mark.parametrize("name", [".hidden.sh", "stop.sh.disabled", "__pycache__/stop.sh"])
def test_ignored_files(tmp_path, name):
    _script(tmp_path / name)
    entries, warnings = discover_hook_dir(tmp_path)
    assert entries == [] and warnings == []


def test_nesting_deeper_than_event_matcher_warns(tmp_path):
    _script(tmp_path / "PreToolUse" / "Bash" / "deeper" / "guard.sh")
    entries, warnings = discover_hook_dir(tmp_path)
    assert entries == []
    assert "nested too deep" in warnings[0]


def test_parse_hooks_config_accepts_a_directory(tmp_path):
    _script(tmp_path / "notify-stop.py")
    entries, warnings = parse_hooks_config(tmp_path)
    assert [e.event for e in entries] == ["Stop"] and warnings == []


def test_json_and_directory_layers_merge_additively(tmp_path):
    config = tmp_path / "hooks.json"
    config.write_text(json.dumps({"hooks": {"Stop": [{"hooks": [
        {"type": "command", "command": "true"}]}]}}))
    hooks_dir = tmp_path / "hooks"
    _script(hooks_dir / "notify-stop.py")
    session: dict = {}
    entries, warnings = load_hooks(session, [config, hooks_dir])
    assert len(entries) == 2 and warnings == []
    assert len(session["hooks"]) == 2


def test_a_discovered_script_actually_runs(tmp_path):
    _script(tmp_path / "Stop" / "block.sh",
            body="#!/bin/sh\necho 'run the tests first' >&2\nexit 2\n")
    session: dict = {}
    load_hooks(session, tmp_path)
    decision = run_hooks(session["hooks"], "Stop",
                         {"hook_event_name": "Stop", "session_id": "t"}, cwd=str(tmp_path))
    assert decision.block is True
    assert "run the tests first" in (decision.reason or "")


def test_a_discovered_script_receives_the_payload_on_stdin(tmp_path):
    out = tmp_path / "seen.json"
    _script(tmp_path / "Stop" / "capture.sh",
            body=f"#!/bin/sh\ncat > {out}\nexit 0\n")
    session: dict = {}
    load_hooks(session, tmp_path)
    run_hooks(session["hooks"], "Stop",
              {"hook_event_name": "Stop", "session_id": "abc"}, cwd=str(tmp_path))
    assert json.loads(out.read_text())["session_id"] == "abc"


def test_a_script_path_with_spaces_is_spawned_without_a_shell(tmp_path):
    marker = tmp_path / "ran"
    _script(tmp_path / "Stop" / "my hook.sh", body=f"#!/bin/sh\ntouch {marker}\nexit 0\n")
    session: dict = {}
    load_hooks(session, tmp_path)
    run_hooks(session["hooks"], "Stop",
              {"hook_event_name": "Stop", "session_id": "t"}, cwd=str(tmp_path))
    assert marker.exists()


def test_discovery_is_deterministic(tmp_path):
    for name in ("c-stop.sh", "a-stop.sh", "b-stop.sh"):
        _script(tmp_path / name)
    first = [e.source for e in discover_hook_dir(tmp_path)[0]]
    second = [e.source for e in discover_hook_dir(tmp_path)[0]]
    assert first == second == sorted(first)


def test_user_layer_is_picked_up_by_a_session(tmp_path, monkeypatch):
    home = tmp_path / "home"
    _script(home / ".agentknit" / "hooks" / "notify-stop.py")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    entries, _ = discover_hook_dir(Path.home() / ".agentknit" / "hooks")
    assert [e.event for e in entries] == ["Stop"]
    assert os.access(entries[0].handler.command, os.X_OK)


# ── deduplication: a hooks.json entry and a discovered script are one hook ────


def test_the_same_script_from_two_layers_runs_once(tmp_path):
    from agentknit.hooks import HookEntry, HookHandler, dedupe_entries
    script = _script(tmp_path / "hooks" / "notify-stop.py")
    from_json = HookEntry(event="Stop", matcher="",
                          handler=HookHandler(type="command",
                                              command=str(tmp_path / "hooks" / "./notify-stop.py")),
                          source="hooks.json")
    from_dir = discover_hook_dir(tmp_path / "hooks")[0][0]
    assert from_dir.handler.command == str(script)
    kept = dedupe_entries([from_json, from_dir])
    assert len(kept) == 1 and kept[0].source == "hooks.json"


def test_dedupe_keeps_different_events_and_matchers(tmp_path):
    from agentknit.hooks import HookEntry, HookHandler, dedupe_entries
    script = str(_script(tmp_path / "guard.sh"))

    def entry(event, matcher):
        return HookEntry(event=event, matcher=matcher,
                         handler=HookHandler(type="command", command=script), source="x")

    kept = dedupe_entries([entry("PreToolUse", "Bash"), entry("PreToolUse", "Edit"),
                           entry("Stop", ""), entry("PreToolUse", "Bash")])
    assert len(kept) == 3


def test_dedupe_never_touches_shell_one_liners_or_python_hooks(tmp_path):
    from agentknit.hooks import HookEntry, HookHandler, dedupe_entries

    def shell(cmd):
        return HookEntry(event="Stop", matcher="",
                         handler=HookHandler(type="command", command=cmd), source="x")

    py = HookEntry(event="Stop", matcher="",
                   handler=HookHandler(type="python", fn=lambda p: None), source="api")
    kept = dedupe_entries([shell("echo a && echo b"), shell("echo a && echo b"), py, py])
    assert len(kept) == 4
