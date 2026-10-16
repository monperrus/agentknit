from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest

from agentknit import BubblewrapToolExecutor, SandboxPolicy, init_session


def executor(tmp_path: Path) -> BubblewrapToolExecutor:
    # File-policy tests do not invoke bwrap; a known executable keeps them
    # portable while production construction still checks the default runtime.
    return BubblewrapToolExecutor(SandboxPolicy(workspace=tmp_path), binary="true")


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


def _bwrap_unusable_reason() -> str | None:
    """Why Bubblewrap cannot build a sandbox on this host, or None if it can.

    An installed bwrap is not a usable one: Ubuntu's
    ``kernel.apparmor_restrict_unprivileged_userns=1`` lets it create the user
    namespace but strips the capabilities it needs to bring up loopback in the
    new network namespace ("loopback: Failed RTM_NEWADDR"), and container
    runners often forbid user namespaces outright.  Probe with the same
    namespace flags the executor uses.
    """
    if shutil.which("bwrap") is None:
        return "Bubblewrap is not installed on this runner"
    probe = subprocess.run(["bwrap", "--unshare-net", "--ro-bind", "/", "/", "true"],
                           capture_output=True, text=True)
    if probe.returncode != 0:
        return f"Bubblewrap cannot create a sandbox on this host: {probe.stderr.strip()}"
    return None


def test_shell_runs_in_workspace_with_a_minimal_environment(tmp_path: Path) -> None:
    reason = _bwrap_unusable_reason()
    if reason is not None:
        pytest.skip(reason)
    e = BubblewrapToolExecutor(SandboxPolicy(workspace=tmp_path))
    result, _ = e.execute("exec_shell",
        {"command": "printf '%s:%s' \"$PWD\" \"${SECRET:-missing}\""},
        {"python_function": "t_run", "param_map": {}}, session={})
    assert '"stdout":"/workspace:missing"' in result


def test_failing_bubblewrap_fails_closed(tmp_path: Path) -> None:
    """A sandbox runtime that cannot start runs nothing and reports the failure."""
    e = BubblewrapToolExecutor(SandboxPolicy(workspace=tmp_path), binary="false")
    result, data = e.execute("exec_shell", {"command": "echo escaped"},
        {"python_function": "t_run", "param_map": {}}, session={})
    assert data["returncode"] != 0
    assert "escaped" not in result


def test_missing_bubblewrap_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="refusing to fall back"):
        BubblewrapToolExecutor(SandboxPolicy(workspace=tmp_path), binary="does-not-exist")
