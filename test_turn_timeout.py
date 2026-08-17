"""Cooperative timeout behaviour for ``run_turn``."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import agentknit._core as core


def _session(tmp_path) -> dict:
    return {
        "messages": [{"role": "system", "content": "system"}],
        "tools": [],
        "structured": True,
        "tool_dispatch": {},
        "session_id": "timeout-test",
        "cache_key": "timeout-test",
        "endpoint": "https://api.example.test/v1",
        "options": [],
        "streaming": False,
        "provider": None,
        "max_output_tokens": None,
        "usage_totals": {"prompt": 0, "completion": 0, "total": 0,
                         "cached": 0, "cache_write": 0},
        "strict_cache_proof": False,
        "llm_call_count": 0,
        "non_interactive": True,
        "log_path": tmp_path / "session.jsonl",
        "on_event": None,
        "_event_handlers": {},
    }


def test_timeout_raises_after_an_inflight_completion_returns(monkeypatch, tmp_path) -> None:
    callbacks = []
    timers = []

    class _Timer:
        def __init__(self, delay, callback) -> None:
            self.delay = delay
            self.callback = callback
            self.cancelled = False
            timers.append(self)

        def start(self) -> None:
            callbacks.append(self.callback)

        def cancel(self) -> None:
            self.cancelled = True

    def _complete(*_args, **_kwargs):
        # Simulate expiry while a blocking completion is in progress.  The
        # completion still returns; run_turn raises at its next safe boundary.
        callbacks[0]()
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content="too late", tool_calls=None,
        ))], usage=None)

    monkeypatch.setattr(core.threading, "Timer", _Timer)
    monkeypatch.setattr(core, "_complete", _complete)

    with pytest.raises(TimeoutError, match="run_turn timed out"):
        core.run_turn(object(), "model", _session(tmp_path), "hello", timeout=10)

    assert timers[0].delay == 10
    assert timers[0].cancelled


def test_negative_timeout_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        core.run_turn(object(), "model", _session(tmp_path), "hello", timeout=-1)
