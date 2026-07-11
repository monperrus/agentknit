from pathlib import Path

import pytest

from agentknit import BubblewrapToolExecutor, SandboxPolicy, init_session


def executor(tmp_path: Path) -> BubblewrapToolExecutor:
    return BubblewrapToolExecutor(SandboxPolicy(workspace=tmp_path))


def test_file_tools_stay_inside_workspace(tmp_path: Path) -> None:
    e = executor(tmp_path)
    entry = {"python_function": "t_write", "param_map": {}}
    result, _ = e.execute("write_file", {"path": "nested/a.txt", "content": "hello"}, entry, session={})
    assert result == "wrote 5 bytes"
    result, _ = e.execute("read_file", {"path": "nested/a.txt"},
                          {"python_function": "t_read", "param_map": {}}, session={})
    assert result == "hello"


@pytest.mark.parametrize("path", ["/tmp/x", "../../x"])
def test_rejects_host_and_traversal_paths(tmp_path: Path, path: str) -> None:
    with pytest.raises(ValueError, match="absolute paths|escapes"):
        executor(tmp_path).execute("read_file", {"path": path},
            {"python_function": "t_read", "param_map": {}}, session={})


def test_rejects_symlink_escape(tmp_path: Path) -> None:
    (tmp_path / "outside").symlink_to("/tmp")
    with pytest.raises(ValueError, match="escapes"):
        executor(tmp_path).execute("read_file", {"path": "outside/x"},
            {"python_function": "t_read", "param_map": {}}, session={})


def test_session_records_sandbox_policy(tmp_path: Path) -> None:
    session = init_session({"model": "test", "tool_specs": [], "behaviour": {}}, tool_executor=executor(tmp_path))
    assert session["tool_executor"].policy.workspace == tmp_path


def test_custom_and_async_tools_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="not supported"):
        executor(tmp_path).execute("custom", {}, {"python_function": "custom"}, session={})


def test_shell_runs_in_workspace_with_a_minimal_environment(tmp_path: Path) -> None:
    result, _ = executor(tmp_path).execute("execute_shell_command",
        {"command": "printf '%s:%s' \"$PWD\" \"${SECRET:-missing}\""},
        {"python_function": "t_run", "param_map": {}}, session={})
    assert '"stdout":"/workspace:missing"' in result
