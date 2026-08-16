"""Cache-write calls must count as cache proof, not raise CacheProofError."""

import pytest

import agentknit._core as core


class _U:
    def __init__(self, prompt, cached=0, creation=0, proof=True):
        self.prompt_tokens = prompt
        self.cached_tokens = cached
        self.cache_creation_tokens = creation
        self.has_cache_proof = proof


def _session(min_tokens=4096):
    return {"strict_cache_proof": True, "llm_call_count": 5,
            "min_cacheable_tokens": min_tokens}


def test_write_only_call_is_proof():
    """First call to cross the provider's cache floor writes the cache; read stays 0."""
    core._enforce_cache_proof(_session(), _U(prompt=4900, creation=4900))


def test_read_call_is_proof():
    core._enforce_cache_proof(_session(), _U(prompt=4900, cached=4000))


def test_no_cache_over_floor_raises():
    with pytest.raises(core.CacheProofError):
        core._enforce_cache_proof(_session(), _U(prompt=4900))


def test_no_cache_under_floor_is_expected():
    core._enforce_cache_proof(_session(), _U(prompt=2413))


def test_missing_proof_field_raises():
    with pytest.raises(core.CacheProofError):
        core._enforce_cache_proof(_session(), _U(prompt=4900, proof=False))
