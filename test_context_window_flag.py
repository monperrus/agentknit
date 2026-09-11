"""--context-window CLI flag: sets/overrides schema["context_window"] after
load_specification(), composable with run:// models and spec files (issue #32).
"""

from __future__ import annotations

import sys

import pytest

import agentknit._core as core


def test_parse_args_context_window() -> None:
    old = sys.argv
    sys.argv = ["agentknit", "some-model", "--context-window", "922000", "do", "it"]
    try:
        parsed = core.parse_args()
    finally:
        sys.argv = old
    assert parsed.context_window == 922000
    assert parsed.model == "some-model"
    assert parsed.task == ["do", "it"]


def test_parse_args_context_window_defaults_to_none() -> None:
    old = sys.argv
    sys.argv = ["agentknit", "some-model", "do", "it"]
    try:
        parsed = core.parse_args()
    finally:
        sys.argv = old
    assert parsed.context_window is None


def test_context_window_applied_on_top_of_loaded_spec(monkeypatch) -> None:
    """The flag overrides schema["context_window"] after load_specification."""
    applied: dict[str, int] = {}

    def fake_load(model, endpoint, spec_path=None):
        return {"model": model, "endpoint": endpoint, "status": "default",
                "context_window": 128000,
                "tool_specs": [], "tools": [],
                "behaviour": {"call_delivery_mode": "structured_tool_calls"}}

    def fake_validate(schema):
        pass

    def fake_pricing(schema):
        pass

    def fake_client(schema):
        return None

    def fake_session(schema, **opts):
        applied["cw"] = schema.get("context_window")
        raise SystemExit(0)  # stop main() before any turn runs

    monkeypatch.setattr(core, "load_specification", fake_load)
    monkeypatch.setattr(core, "validate_schema", fake_validate)
    monkeypatch.setattr(core, "check_and_display_pricing", fake_pricing)
    monkeypatch.setattr(core, "create_client", fake_client)
    monkeypatch.setattr(core, "init_session", fake_session)

    old = sys.argv
    sys.argv = ["agentknit", "m", "--context-window", "922000", "t"]
    try:
        with pytest.raises(SystemExit):
            core.main()
    finally:
        sys.argv = old
    assert applied["cw"] == 922000


def test_context_window_rejects_non_positive(monkeypatch) -> None:
    monkeypatch.setattr(core, "load_specification",
                        lambda m, e, spec_path=None: {"model": m, "endpoint": e})
    old = sys.argv
    sys.argv = ["agentknit", "m", "--context-window", "0", "t"]
    try:
        with pytest.raises(SystemExit) as exc:
            core.main()
    finally:
        sys.argv = old
    assert "positive" in str(exc.value)
