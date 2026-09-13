"""`/tool list|activate|remove` — runtime toolset manipulation.

Remove parks the tool spec *and* retires its dispatch entry (the model cannot
call a tool it can no longer see); activate restores it verbatim, brings in a
TOOL_LIBRARY function (schema from its `Tool spec:` docstring or inferred from
its signature), or resolves a legacy alias.
"""

from __future__ import annotations

from pathlib import Path

import agentknit
from agentknit import _core
from agentknit.slash_commands import REGISTRY


def _session(tmp_path: Path) -> dict:
    spec = {
        "model": "m", "endpoint": "http://x",
        "tool_specs": _core._DEFAULT_TOOL_SCHEMA,
        # explicit dispatch: earlier tests may have replaced t_read & co in
        # TOOL_LIBRARY with different signatures, and the ordered "tools"
        # path re-validates param maps against them.
        "tool_dispatch": dict(_core._DEFAULT_TOOL_DISPATCH),
        "behaviour": {"call_delivery_mode": "structured_tool_calls"},
    }
    session = agentknit.init_session(spec, non_interactive=True)
    session["log_path"] = tmp_path / "log.jsonl"
    return session


def _names(session: dict) -> list[str]:
    return [t["function"]["name"] for t in session["tools"]]


def test_list_shows_active_tools(capsys):
    s = _session(Path("/tmp"))
    assert REGISTRY.dispatch("/tool list", s, client=None, model="m") is True
    out = capsys.readouterr().out
    for n in ("read_file", "write_file", "str_replace", "exec_shell"):
        assert n in out
    assert "Usage: /tool" in out


def test_bare_tool_defaults_to_list(capsys):
    s = _session(Path("/tmp"))
    REGISTRY.dispatch("/tool", s, client=None, model="m")
    assert "Active tools (4)" in capsys.readouterr().out


def test_remove_drops_spec_and_dispatch(capsys):
    s = _session(Path("/tmp"))
    REGISTRY.dispatch("/tool remove exec_shell", s, client=None, model="m")
    assert "exec_shell" not in _names(s)
    assert "exec_shell" not in s["tool_dispatch"]
    assert "Tool removed: exec_shell" in capsys.readouterr().out


def test_remove_unknown_tool_fails_cleanly(capsys):
    s = _session(Path("/tmp"))
    REGISTRY.dispatch("/tool remove glob", s, client=None, model="m")
    assert len(_names(s)) == 4  # unchanged
    assert "not active" in capsys.readouterr().out


def test_activate_restores_removed_tool_verbatim(capsys):
    s = _session(Path("/tmp"))
    before = [t for t in s["tools"] if t["function"]["name"] == "read_file"][0]
    REGISTRY.dispatch("/tool remove read_file", s, client=None, model="m")
    REGISTRY.dispatch("/tool activate read_file", s, client=None, model="m")
    assert _names(s)[-1] == "read_file"
    # verbatim: same dispatch entry (with its param_map), same spec object
    assert s["tool_dispatch"]["read_file"]["python_function"] == "t_read"
    assert s["tools"][-1] == before
    assert "read_file" not in (s.get("_removed_tools") or {})


def test_activate_library_function_with_docstring_spec(capsys):
    s = _session(Path("/tmp"))
    REGISTRY.dispatch("/tool activate t_glob", s, client=None, model="m")
    assert "glob" in _names(s)
    assert s["tool_dispatch"]["glob"]["python_function"] == "t_glob"
    # the activated tool really works end-to-end
    text, _meta = agentknit.dispatch("glob", {"pattern": "/nonexistent-*"},
                                     s["tool_dispatch"])
    assert "(no matches)" in text


def test_activate_library_function_infers_schema():
    s = _session(Path("/tmp"))
    # t_list_dir has no "Tool spec:" docstring → schema inferred from signature
    REGISTRY.dispatch("/tool activate t_list_dir", s, client=None, model="m")
    spec = s["tools"][-1]
    assert spec["function"]["name"] == "list_dir"
    assert "path" in spec["function"]["parameters"]["properties"]


def test_activate_resolves_legacy_alias():
    s = _session(Path("/tmp"))
    REGISTRY.dispatch("/tool activate execute_shell_command", s, client=None,
                      model="m")
    assert "exec_shell" in _names(s)


def test_activate_unknown_tool_reports_known_names(capsys):
    s = _session(Path("/tmp"))
    REGISTRY.dispatch("/tool activate nope", s, client=None, model="m")
    out = capsys.readouterr().out
    assert "Unknown tool 'nope'" in out
    assert "read_file" in out


def test_activate_is_idempotent():
    s = _session(Path("/tmp"))
    REGISTRY.dispatch("/tool activate read_file", s, client=None, model="m")
    assert _names(s).count("read_file") == 1


def test_list_hides_ask_user_tools_in_non_interactive(capsys):
    s = _session(Path("/tmp"))  # non_interactive=True
    REGISTRY.dispatch("/tool list", s, client=None, model="m")
    out = capsys.readouterr().out
    # candidates are printed with their model-facing name (t_ stripped)
    assert "ask_user" not in out


def test_unknown_subcommand(capsys):
    s = _session(Path("/tmp"))
    REGISTRY.dispatch("/tool frobnicate x", s, client=None, model="m")
    assert "Unknown /tool sub-command" in capsys.readouterr().out


def test_missing_tool_name(capsys):
    s = _session(Path("/tmp"))
    REGISTRY.dispatch("/tool activate", s, client=None, model="m")
    assert "Usage: /tool activate <tool_name>" in capsys.readouterr().out


def test_remove_retires_legacy_alias_dispatch_too():
    s = _session(Path("/tmp"))
    REGISTRY.dispatch("/tool remove exec_shell", s, client=None, model="m")
    assert "execute_shell_command" not in s["tool_dispatch"]
    REGISTRY.dispatch("/tool activate exec_shell", s, client=None, model="m")
    assert "execute_shell_command" in s["tool_dispatch"]


def test_tool_command_via_llm_tool():
    """t_slash_command('tool', ...) runs the same handler."""
    import agentknit.slash_commands as sc
    s = _session(Path("/tmp"))
    sc.slash_tool_ctx.update(session=s, client=None, model="m")
    text, meta = sc.t_slash_command("tool", "remove exec_shell")
    assert "Tool removed: exec_shell" in text
    assert "exec_shell" not in _names(s)
    assert "result" in meta
