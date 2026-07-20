"""Tests for the cold-resume cache-proof relaxation in agentknit._core.

Strict cache-proof mode normally aborts a turn when the provider reports no
cache hit after the first LLM call.  When a session is resumed after a long
pause, the provider's prefix cache has expired through no fault of the
caller, so enforcing the check would spuriously break the output.  These
tests pin the behaviour described in :func:`_enforce_cache_proof`.
"""

from __future__ import annotations

import datetime

import pytest

from agentknit._core import (
    CACHE_COLD_GAP_SECONDS,
    _enforce_cache_proof,
    _last_message_age_seconds,
)
from agentknit.exceptions import CacheProofError


# ── helpers ──────────────────────────────────────────────────────────────────

class _Usage:
    """Minimal stand-in for the Usage object produced by openai_compat."""

    def __init__(self, *, has_cache_proof: bool = False, cached_tokens: int = 0) -> None:
        self.has_cache_proof = has_cache_proof
        self.cached_tokens = cached_tokens


def _session(
    *,
    last_ts: str | None = None,
    llm_call_count: int = 2,
    strict: bool = True,
) -> dict:
    messages = [{"role": "system", "content": "sys"}]
    if last_ts is not None:
        messages.append({"role": "user", "content": "hi", "ts": last_ts})
    events: list[tuple[str, dict]] = []

    def _on_event(event_type: str, data: dict) -> None:
        events.append((event_type, data))

    return {
        "messages": messages,
        "llm_call_count": llm_call_count,
        "strict_cache_proof": strict,
        "on_event": _on_event,
        "_event_handlers": {},
    }


def _iso(seconds_ago: float) -> str:
    return (datetime.datetime.now() - datetime.timedelta(seconds=seconds_ago)).isoformat(
        timespec="seconds"
    )


# ── _last_message_age_seconds ────────────────────────────────────────────────

def test_age_seconds_returns_none_without_timestamp():
    session = _session(last_ts=None)
    assert _last_message_age_seconds(session) is None


def test_age_seconds_measures_gap():
    session = _session(last_ts=_iso(30))
    age = _last_message_age_seconds(session)
    assert age is not None and 20 <= age <= 60


def test_age_seconds_ignores_garbage_timestamp():
    session = _session(last_ts="not-a-date")
    assert _last_message_age_seconds(session) is None


# ── _enforce_cache_proof: disabled / first call ──────────────────────────────

def test_enforce_noop_when_strict_disabled():
    session = _session(last_ts=_iso(10_000), strict=False)
    # Should not raise even with a total miss.
    _enforce_cache_proof(session, _Usage(has_cache_proof=False, cached_tokens=0))


def test_enforce_noop_on_first_call():
    session = _session(last_ts=_iso(10_000), llm_call_count=1)
    _enforce_cache_proof(session, _Usage(has_cache_proof=False, cached_tokens=0))


# ── _enforce_cache_proof: hot path still raises ──────────────────────────────

def test_enforce_raises_on_miss_within_cache_window():
    # Recent message: cache should still be warm, so a miss is a real error.
    session = _session(last_ts=_iso(60))
    with pytest.raises(CacheProofError):
        _enforce_cache_proof(session, _Usage(has_cache_proof=False, cached_tokens=0))


def test_enforce_raises_when_no_cache_proof_within_window():
    session = _session(last_ts=_iso(60))
    with pytest.raises(CacheProofError):
        _enforce_cache_proof(session, _Usage(has_cache_proof=False, cached_tokens=123))


# ── _enforce_cache_proof: cold resume ────────────────────────────────────────

def test_cold_resume_does_not_raise_on_miss():
    session, events = _capturing_session(_iso(CACHE_COLD_GAP_SECONDS + 600))
    # No raise, and a cache_cold event is emitted.
    _enforce_cache_proof(session, _Usage(has_cache_proof=False, cached_tokens=0))
    assert events and events[0][0] == "cache_cold"


def _capturing_session(last_ts: str | None, **kw) -> tuple[dict, list[tuple[str, dict]]]:
    events: list[tuple[str, dict]] = []
    session = _session(last_ts=last_ts, **kw)
    session["on_event"] = lambda et, data: events.append((et, data))
    return session, events


def test_cold_resume_emits_cache_cold_event():
    session, events = _capturing_session(_iso(CACHE_COLD_GAP_SECONDS + 600))
    _enforce_cache_proof(session, _Usage(has_cache_proof=False, cached_tokens=0))
    assert events and events[0][0] == "cache_cold"
    assert events[0][1]["age"] > CACHE_COLD_GAP_SECONDS
    assert session["_cache_cold_warned"] is True


def test_cold_resume_with_real_cache_hit_does_not_warn():
    # Even on a cold resume, a genuine cache hit is the happy path — no warn.
    session, events = _capturing_session(_iso(CACHE_COLD_GAP_SECONDS + 600))
    _enforce_cache_proof(session, _Usage(has_cache_proof=True, cached_tokens=999))
    assert not events
    assert "_cache_cold_warned" not in session


def test_boundary_just_under_threshold_still_raises():
    # One second under the threshold: still "warm", so a miss raises.
    session, events = _capturing_session(_iso(CACHE_COLD_GAP_SECONDS - 1))
    with pytest.raises(CacheProofError):
        _enforce_cache_proof(session, _Usage(has_cache_proof=False, cached_tokens=0))
    assert not events
