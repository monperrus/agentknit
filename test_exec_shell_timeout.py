"""exec_shell's max wall time: TTL - margin (default 600 - 20 = 580 s)."""

from __future__ import annotations

import json
import time

from agentknit.tool_library import (
    DEFAULT_TOOL_TTL_S,
    EXEC_SHELL_MARGIN_S,
    _exec_shell_timeout_s,
    _tool_context,
    t_run,
)


def _reset_context() -> None:
    for attr in ("tool_ttl_seconds", "tool_dispatch"):
        if hasattr(_tool_context, attr):
            delattr(_tool_context, attr)


def test_default_timeout_is_ttl_minus_margin() -> None:
    _reset_context()
    assert _exec_shell_timeout_s() == DEFAULT_TOOL_TTL_S - EXEC_SHELL_MARGIN_S == 580


def test_configured_ttl_applies_margin() -> None:
    _reset_context()
    _tool_context.tool_ttl_seconds = 300
    assert _exec_shell_timeout_s() == 280


def test_invalid_ttl_falls_back_to_default() -> None:
    for bad in (None, 0, -5, "abc"):
        _reset_context()
        _tool_context.tool_ttl_seconds = bad
        assert _exec_shell_timeout_s() == 580


def test_tiny_ttl_clamps_to_one_second() -> None:
    _reset_context()
    _tool_context.tool_ttl_seconds = 10  # below the margin
    assert _exec_shell_timeout_s() == 1


def test_t_run_times_out_with_ttl_minus_margin() -> None:
    _reset_context()
    _tool_context.tool_ttl_seconds = 21  # -> 1 s timeout
    try:
        start = time.monotonic()
        result, _ = t_run("sleep 30")
        elapsed = time.monotonic() - start
    finally:
        _reset_context()
    assert elapsed < 10, f"t_run took {elapsed:.1f}s; expected the 1 s cap"
    payload = json.loads(result)
    assert payload["error"] == "command timed out after 1 s"
    assert "did not finish within 1 seconds" in payload["hint"]
