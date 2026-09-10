"""Cache-write calls must count as cache proof.

Only the *first* call of a session raises CacheProofError when no cache
accounting is exposed at all.  After that, the turn's tokens are already
paid, so missing proof degrades to a temporary ``cache_proof_missing``
warning (``session["_cache_status"] == "missing"``) instead of aborting.
"""

import pytest

import agentknit._core as core


class _U:
    def __init__(self, prompt, cached=0, creation=0, proof=True):
        self.prompt_tokens = prompt
        self.cached_tokens = cached
        self.cache_creation_tokens = creation
        self.has_cache_proof = proof


def _session(min_tokens=4096, llm_call_count=5):
    events = []

    def _on_event(event_type, data):
        events.append((event_type, data))

    return {"strict_cache_proof": True, "llm_call_count": llm_call_count,
            "min_cacheable_tokens": min_tokens, "_cache_status": "ok",
            "on_event": _on_event, "_event_handlers": {}, "_events": events}


def test_write_only_call_is_proof():
    """First call to cross the provider's cache floor writes the cache; read stays 0."""
    core._enforce_cache_proof(_session(), _U(prompt=4900, creation=4900))


def test_read_call_is_proof():
    core._enforce_cache_proof(_session(), _U(prompt=4900, cached=4000))


def test_no_cache_over_floor_warns_and_continues():
    session = _session()
    core._enforce_cache_proof(session, _U(prompt=4900))
    assert session["_events"][-1][0] == "cache_proof_missing"
    assert session["_cache_status"] == "missing"


def test_no_cache_under_floor_is_expected():
    session = _session()
    core._enforce_cache_proof(session, _U(prompt=2413))
    assert session["_events"][-1][0] == "cache_below_minimum"
    assert session["_cache_status"] == "ok"


def test_missing_proof_field_after_first_call_warns():
    session = _session()
    core._enforce_cache_proof(session, _U(prompt=4900, proof=False))
    assert session["_events"][-1][0] == "cache_proof_missing"
    assert session["_cache_status"] == "missing"


def test_missing_proof_field_on_first_call_raises():
    with pytest.raises(core.CacheProofError):
        core._enforce_cache_proof(_session(llm_call_count=1), _U(prompt=4900, proof=False))


def test_first_call_cache_write_counts_as_accounting():
    # A cache write on the very first call proves accounting exists.
    session = _session(llm_call_count=1)
    core._enforce_cache_proof(session, _U(prompt=4900, proof=False, creation=4900))
    assert session["_cache_status"] == "ok"
