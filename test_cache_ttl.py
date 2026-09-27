"""Tests for the prefix-cache TTL knob (``cache_ttl_seconds``).

Providers expire their prefix caches on their own schedule — Kimi 5 minutes
by default, DeepSeek's disk cache "a few hours to a few days", OpenAI
~5–10 min.  A session can now declare that horizon, and strict cache-proof
mode classifies a resumed turn as *cold* (dim notice, no failure) using the
declared TTL instead of the blanket ``CACHE_COLD_GAP_SECONDS`` guess.  These
tests pin: the default, the declared override, warmth anchoring on the last
cache proof, and the plumbing of the knob through ``init_session``.
"""

from __future__ import annotations

import time
from typing import Any

import agentknit._core as core
from agentknit._core import (
    CACHE_COLD_GAP_SECONDS,
    DEFAULT_CACHE_TTL_SECONDS,
    DEFAULT_MIN_CACHEABLE_TOKENS,
    _cache_ttl_seconds,
    _cache_warmth,
    _enforce_cache_proof,
)


# ── helpers ──────────────────────────────────────────────────────────────────

class _Usage:
    """Minimal stand-in for the Usage object produced by openai_compat."""

    prompt_tokens = 5_000
    cache_creation_tokens = 0

    def __init__(self, *, has_cache_proof: bool = False, cached_tokens: int = 0) -> None:
        self.has_cache_proof = has_cache_proof
        self.cached_tokens = cached_tokens


def _session(**kw: Any) -> dict:
    session: dict[str, Any] = {
        "messages": [],
        "llm_call_count": 2,
        "strict_cache_proof": True,
        "on_event": lambda et, data: None,
        "_event_handlers": {},
        "_cache_status": "ok",
    }
    session.update(kw)
    return session


# ── _cache_ttl_seconds ───────────────────────────────────────────────────────

def test_ttl_defaults_to_cold_gap_when_undeclared() -> None:
    assert _cache_ttl_seconds(_session()) == DEFAULT_CACHE_TTL_SECONDS
    assert DEFAULT_CACHE_TTL_SECONDS == CACHE_COLD_GAP_SECONDS


def test_ttl_uses_declared_value() -> None:
    session = _session(cache_ttl_seconds=300)
    assert _cache_ttl_seconds(session) == 300


def test_ttl_falls_back_on_garbage() -> None:
    # Zero / negative / non-numeric mean "not declared", not "expires now".
    assert _cache_ttl_seconds(_session(cache_ttl_seconds=0)) == DEFAULT_CACHE_TTL_SECONDS
    assert _cache_ttl_seconds(_session(cache_ttl_seconds=-5)) == DEFAULT_CACHE_TTL_SECONDS
    assert _cache_ttl_seconds(_session(cache_ttl_seconds="soon")) == DEFAULT_CACHE_TTL_SECONDS


# ── cold-resume classification uses the declared TTL ─────────────────────────

def _capture_events(session: dict) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    session["on_event"] = lambda et, data: events.append((et, data))
    return events


def _msg(seconds_ago: float) -> dict:
    import datetime
    ts = (datetime.datetime.now().astimezone()
          - datetime.timedelta(seconds=seconds_ago)).isoformat(timespec="seconds")
    return {"role": "user", "content": "hi", "ts": ts}


def test_declared_ttl_makes_medium_gap_cold() -> None:
    # 20 minutes is far under the 3600s default but past Kimi's 5m TTL:
    # with the knob declared it must be a cold resume, not a warning.
    session = _session(cache_ttl_seconds=300, messages=[_msg(20 * 60)])
    events = _capture_events(session)
    _enforce_cache_proof(session, _Usage(has_cache_proof=False, cached_tokens=0))
    assert [e[0] for e in events] == ["cache_cold"]
    assert events[0][1]["ttl"] == 300
    assert session["_cache_cold_warned"] is True


def test_gap_under_declared_ttl_still_warns() -> None:
    # Within the TTL window a miss is a real surprise: warn, don't excuse.
    session = _session(cache_ttl_seconds=3600, messages=[_msg(60)])
    events = _capture_events(session)
    _enforce_cache_proof(session, _Usage(has_cache_proof=False, cached_tokens=0))
    assert [e[0] for e in events] == ["cache_proof_missing"]
    assert session["_cache_status"] == "missing"


def test_undeclared_session_keeps_old_behaviour() -> None:
    # Backward compatibility: no cache_ttl_seconds → 3600s threshold, so a
    # 20-minute gap still warns (exactly the pre-knob classification).
    session = _session(messages=[_msg(20 * 60)])
    events = _capture_events(session)
    _enforce_cache_proof(session, _Usage(has_cache_proof=False, cached_tokens=0))
    assert [e[0] for e in events] == ["cache_proof_missing"]


# ── cache warmth ──────────────────────────────────────────────────────────────

def test_warmth_none_before_any_cache_proof() -> None:
    assert _cache_warmth(_session(cache_ttl_seconds=300)) is None


def test_warmth_counts_down_from_last_cache_proof() -> None:
    session = _session(cache_ttl_seconds=300)
    session["_cache_last_proof_ts"] = time.time() - 100
    w = _cache_warmth(session)
    assert w is not None and 195 <= w["expires_in"] <= 200
    assert w["ttl_seconds"] == 300


def test_warmth_negative_once_past_ttl() -> None:
    session = _session(cache_ttl_seconds=300)
    session["_cache_last_proof_ts"] = time.time() - 400
    w = _cache_warmth(session)
    assert w is not None and w["expires_in"] < 0


def test_cache_proof_stamps_warmth_anchor() -> None:
    session = _session(cache_ttl_seconds=300, messages=[_msg(60)])
    before = time.time()
    _enforce_cache_proof(session, _Usage(has_cache_proof=True, cached_tokens=1))
    stamped = session.get("_cache_last_proof_ts")
    assert stamped is not None and before <= stamped <= time.time()


# ── init_session plumbing ─────────────────────────────────────────────────────

def _fresh_schema() -> dict:
    return {"model": "m", "endpoint": "https://example.com"}


def _init(session_schema: dict) -> Any:
    # init_session touches the filesystem for logs; HOME is sandboxed by
    # conftest, so this stays machine-independent.
    return core.init_session(session_schema)


def test_init_session_sets_cache_ttl_from_schema() -> None:
    session = _init({**_fresh_schema(), "cache_ttl_seconds": 300})
    assert session["cache_ttl_seconds"] == 300
    assert session["_cache_last_proof_ts"] is None


def test_init_session_defaults_cache_ttl() -> None:
    session = _init(_fresh_schema())
    assert session["cache_ttl_seconds"] == DEFAULT_CACHE_TTL_SECONDS


def test_restore_backfills_cache_ttl_on_old_snapshots() -> None:
    old = _init(_fresh_schema())
    legacy = {k: v for k, v in old.items() if k != "cache_ttl_seconds"}
    legacy.pop("_cache_last_proof_ts", None)
    # A snapshot saved before the feature lacks the keys; restore must not
    # fail validation and must backfill the default.
    restored = core.init_session(_fresh_schema(), session=legacy)
    assert restored["cache_ttl_seconds"] == DEFAULT_CACHE_TTL_SECONDS
    assert restored.get("min_cacheable_tokens", DEFAULT_MIN_CACHEABLE_TOKENS) == 0


def test_restore_overrides_cache_ttl_from_kwarg() -> None:
    restored = core.init_session(_fresh_schema(),
                                 session=_init(_fresh_schema()),
                                 cache_ttl_seconds=42)
    assert restored["cache_ttl_seconds"] == 42
