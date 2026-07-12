"""Opt-in isolated executors for agent tools on Linux."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, TypedDict


class ToolSessionContext(TypedDict, total=False):
    """The safe subset of session state made available to executors."""
    session_id: str


class ToolExecutor(Protocol):
    def execute(self, tool_name: str, args: dict, dispatch_entry: dict, *,
                session: ToolSessionContext) -> tuple[str, dict]: ...


class LocalToolExecutor:
    """The historic in-controller dispatcher, used unless an executor is supplied."""

    def execute(self, tool_name: str, args: dict, dispatch_entry: dict, *,
                session: ToolSessionContext) -> tuple[str, dict]:
        # Import lazily to avoid a _core import cycle.
        from ._core import dispatch
        return dispatch(tool_name, args, {tool_name: dispatch_entry})


@dataclass(frozen=True)
class SandboxPolicy:
    workspace: Path
    network: str = "none"
    writable_paths: tuple[str, ...] = (".",)
    read_only_paths: tuple[str, ...] = ()
    timeout_seconds: int = 60
    environment: dict[str, str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "workspace", Path(self.workspace).resolve())
        if self.network != "none":
            raise ValueError("only network='none' is currently supported")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not self.workspace.is_dir():
            raise ValueError(f"workspace is not a directory: {self.workspace}")

    def metadata(self) -> dict:
        data = asdict(self)
        data["workspace"] = str(self.workspace)
        return data


class BubblewrapToolExecutor:
    """Execute supported built-ins inside an unprivileged Bubblewrap sandbox.

    Only the standard file tools and synchronous shell tool are supported.
    Custom Python callables and background tools are deliberately rejected:
    running them in the controller would defeat the isolation promise.
    """

    _FILE_TOOLS = {"t_read", "t_write", "t_update"}
    _SHELL_TOOLS = {"t_run"}

    def __init__(self, policy: SandboxPolicy, *, binary: str = "bwrap") -> None:
        self.policy = policy
        self.binary = shutil.which(binary)
        if self.binary is None:
            raise RuntimeError("Bubblewrap is unavailable; refusing to fall back to local execution")

    def execute(self, tool_name: str, args: dict, dispatch_entry: dict, *,
                session: ToolSessionContext) -> tuple[str, dict]:
        fn = dispatch_entry.get("python_function")
        if fn in self._FILE_TOOLS:
            return self._file_tool(str(fn), args, dispatch_entry)
        if fn in self._SHELL_TOOLS:
            return self._shell_tool(args, dispatch_entry)
        raise RuntimeError(
            f"tool {tool_name!r} ({fn!r}) is not supported by BubblewrapToolExecutor; "
            "custom and async tools must provide a sandbox adapter"
        )

    def _path(self, value: object) -> Path:
        if not isinstance(value, str) or not value:
            raise ValueError("path must be a non-empty relative path")
        raw = Path(value)
        if raw.is_absolute():
            raise ValueError("absolute paths are not allowed in a sandbox")
        candidate = (self.policy.workspace / raw).resolve()
        try:
            candidate.relative_to(self.policy.workspace)
        except ValueError as exc:
            raise ValueError("path escapes the sandbox workspace") from exc
        return candidate

    def _file_tool(self, fn: str, args: dict, entry: dict) -> tuple[str, dict]:
        mapped = {entry.get("param_map", {}).get(k, k): v for k, v in args.items()}
        path = self._path(mapped.get("path"))
        if fn == "t_read":
            try:
                text = path.read_text()
                offset, limit = mapped.get("offset"), mapped.get("limit")
                if offset is not None or limit is not None:
                    lines = text.splitlines(keepends=True)
                    text = "".join(lines[offset or 0:(offset or 0) + limit if limit else None])
                return text, {"result": text}
            except OSError as exc:
                result = f"ERROR: {exc}"
                return result, {"result": result}
        if fn == "t_write":
            path.parent.mkdir(parents=True, exist_ok=True)
            content = str(mapped.get("content", ""))
            path.write_text(content)
            result = f"wrote {len(content)} bytes"
            return result, {"result": result}
        old, new = str(mapped.get("old", "")), str(mapped.get("new", ""))
        text = path.read_text()
        if old not in text:
            result = "ERROR: old text not found"
            return result, {"result": result}
        path.write_text(text.replace(old, new))
        result = f"OK: replaced {text.count(old)} occurrence(s) in {mapped.get('path')}"
        return result, {"result": result}

    def _bwrap_command(self, command: str) -> list[str]:
        cmd = [self.binary, "--die-with-parent", "--new-session", "--unshare-net",
               "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
               "--ro-bind", str(self.policy.workspace), "/workspace"]
        for root in ("/usr", "/bin", "/lib", "/lib64", "/etc"):
            if Path(root).exists():
                cmd += ["--ro-bind", root, root]
        for item in self.policy.writable_paths:
            path = self._path(item)
            if item != ".":
                path.mkdir(parents=True, exist_ok=True)
            target = "/workspace" if item == "." else f"/workspace/{item}"
            cmd += ["--bind", str(path), target]
        env = self.policy.environment or {"PATH": "/usr/bin:/bin"}
        for key, value in env.items():
            cmd += ["--setenv", key, value]
        return cmd + ["--chdir", "/workspace", "/bin/bash", "-c", command]

    def _shell_tool(self, args: dict, entry: dict) -> tuple[str, dict]:
        command = {entry.get("param_map", {}).get(k, k): v for k, v in args.items()}.get("command")
        if not isinstance(command, str):
            raise ValueError("command must be a string")
        try:
            completed = subprocess.run(self._bwrap_command(command), text=True,
                capture_output=True, timeout=self.policy.timeout_seconds, env={})
            data = {"stdout": completed.stdout, "stderr": completed.stderr,
                    "returncode": completed.returncode}
        except subprocess.TimeoutExpired as exc:
            data = {"stdout": exc.stdout or "", "stderr": exc.stderr or "",
                    "error": f"command timed out after {self.policy.timeout_seconds} s"}
        result = json.dumps(data, separators=(",", ":"))
        return result, {**data, "result": result}
