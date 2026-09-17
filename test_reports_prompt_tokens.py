"""A backend that reports no input tokens makes cache proof unmeasurable.

Cursor's protocol, for one, never sends a prompt-token count. Strict cache
proof then aborts on the first call, and the only escape was
`--no-strict-cache-proof`, which also switches the check off for every genuine
caching regression. `reports_prompt_tokens: false` states the narrower fact.
"""

from __future__ import annotations

import pytest

from agentknit._core import _enforce_cache_proof, init_session
from agentknit.exceptions import CacheProofError


class _Usage:
    """A usage object with no cache accounting at all."""

    has_cache_proof = False
    cached_tokens = 0
    prompt_tokens = 0
    cache_creation_tokens = 0


def _session(**schema_extra):
    schema = {
        "model": "m",
        "endpoint": "run:///bin/true",
        "tool_specs": [],
        "tool_dispatch": {},
        **schema_extra,
    }
    return init_session(schema)


def test_missing_accounting_still_aborts_by_default():
    session = _session()
    session["llm_call_count"] = 1
    with pytest.raises(CacheProofError):
        _enforce_cache_proof(session, _Usage())


def test_declared_unmeasurable_does_not_abort():
    session = _session(reports_prompt_tokens=False)
    session["llm_call_count"] = 1
    _enforce_cache_proof(session, _Usage())          # must not raise
    assert session.get("_cache_status") == "unmeasurable"


def test_unmeasurable_is_announced_once():
    events = []
    session = _session(reports_prompt_tokens=False)
    session["on_event"] = lambda name, data: events.append(name)
    session["llm_call_count"] = 1

    _enforce_cache_proof(session, _Usage())
    _enforce_cache_proof(session, _Usage())
    _enforce_cache_proof(session, _Usage())

    assert events.count("cache_unmeasurable") == 1


def test_strict_flag_still_wins_when_disabled():
    """--no-strict-cache-proof keeps working, independently of the new field."""
    session = _session()
    session["strict_cache_proof"] = False
    session["llm_call_count"] = 1
    _enforce_cache_proof(session, _Usage())          # must not raise
    # The blanket flag short-circuits first, so no unmeasurable notice is due.
    assert session.get("_cache_status") == "ok"
