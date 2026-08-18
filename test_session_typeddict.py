"""Tests for the ``Session`` TypedDict (issue #3, cheaper TypedDict alternative)."""

from __future__ import annotations

from agentknit import Session, init_session
from agentknit._core import _REQUIRED_SESSION_KEYS

_MINIMAL_SCHEMA = {
    "model": "test-model",
    "endpoint": "https://example.com",
    "tool_specs": [],
    "behaviour": {"call_delivery_mode": "structured_tool_calls"},
}


def test_session_is_a_plain_dict() -> None:
    """TypedDict is types-only: the runtime object is still a dict."""
    session = init_session(_MINIMAL_SCHEMA)
    assert isinstance(session, dict)


def test_session_required_keys_are_declared() -> None:
    """Every key init_session() produces must be declared on the TypedDict.

    This is the single-source-of-truth property from issue #3: adding a
    field to init_session() without declaring it here fails this test.
    """
    declared = set(Session.__annotations__)
    missing = _REQUIRED_SESSION_KEYS - declared
    assert not missing, f"undeclared session keys: {sorted(missing)}"


def test_declared_required_keys_are_validated() -> None:
    """Conversely, every *required* TypedDict key must be in the restore check.

    NotRequired keys (log_path, auth, durable, underscore runtime state) are
    exempt — they may be absent from sessions saved by older versions.
    """
    required = {k for k, t in Session.__annotations__.items()
                if "NotRequired" not in str(t)}
    extra = required - _REQUIRED_SESSION_KEYS
    assert not extra, f"declared required but not restored/validated: {sorted(extra)}"


def test_session_dict_fields_present_after_init(tmp_path, monkeypatch) -> None:
    import agentknit._core as core
    monkeypatch.setattr(core, "LOG_BASE", tmp_path)
    session = init_session(_MINIMAL_SCHEMA)
    for key in _REQUIRED_SESSION_KEYS:
        assert key in session, f"missing {key!r}"
    assert isinstance(session["messages"], list)
    assert session["messages"][0]["role"] == "system"
    assert session["usage_totals"]["total"] == 0
