"""Tests for the min_cacheable_tokens relaxation in agentknit._core.

Providers with a minimum cacheable prompt size (e.g. Anthropic Claude Haiku
~4096 tokens, GPT-5.6-class models ~1024 tokens) cache nothing by design
below that floor.  Strict cache-proof mode should not treat those legitimate
zero-cache responses as failures.  These tests pin the behaviour described
in :func:`_enforce_cache_proof`.
"""

from __future__ import annotations

import pytest

from agentknit._core import DEFAULT_MIN_CACHEABLE_TOKENS, _enforce_cache_proof
from agentknit.exceptions import CacheProofError


class _Usage:
    """Minimal stand-in for the Usage object produced by openai_compat."""

    def __init__(self, *, has_cache_proof: bool = True, cached_tokens: int = 0,
                 prompt_tokens: int = 0) -> None:
        self.has_cache_proof = has_cache_proof
        self.cached_tokens = cached_tokens
        self.prompt_tokens = prompt_tokens


def _session(*, min_cacheable_tokens: int | None = None, llm_call_count: int = 2,
             strict: bool = True) -> dict:
    events: list[tuple[str, dict]] = []

    def _on_event(event_type: str, data: dict) -> None:
        events.append((event_type, data))

    session = {
        "messages": [{"role": "system", "content": "sys"}],
        "llm_call_count": llm_call_count,
        "strict_cache_proof": strict,
        "on_event": _on_event,
        "_event_handlers": {},
        "_events": events,
    }
    if min_cacheable_tokens is not None:
        session["min_cacheable_tokens"] = min_cacheable_tokens
    return session


def test_default_min_cacheable_tokens_is_zero():
    assert DEFAULT_MIN_CACHEABLE_TOKENS == 0


def test_no_raise_when_prompt_below_min_cacheable_tokens():
    session = _session(min_cacheable_tokens=4096)
    # claude-haiku-completions.py: prompts under ~4096 tokens never cache.
    _enforce_cache_proof(session, _Usage(cached_tokens=0, prompt_tokens=1200))
    assert session["_events"][-1][0] == "cache_below_minimum"


def test_raises_when_prompt_at_or_above_min_cacheable_tokens():
    session = _session(min_cacheable_tokens=4096)
    with pytest.raises(CacheProofError):
        _enforce_cache_proof(session, _Usage(cached_tokens=0, prompt_tokens=5000))


def test_raises_when_no_min_cacheable_tokens_configured():
    # Without a configured floor, any zero-cache-hit call after the first
    # is treated as a genuine miss, preserving prior (strict) behaviour.
    session = _session()
    with pytest.raises(CacheProofError):
        _enforce_cache_proof(session, _Usage(cached_tokens=0, prompt_tokens=1200))


def test_cache_hit_never_raises_regardless_of_min_cacheable_tokens():
    session = _session(min_cacheable_tokens=4096)
    _enforce_cache_proof(session, _Usage(cached_tokens=500, prompt_tokens=5000))
    assert not session["_events"]
