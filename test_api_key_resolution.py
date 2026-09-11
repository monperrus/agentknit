"""API-key resolution in create_client/_get_key_for_schema.

Covers the resolution order documented in specification.md:
keyring → key_env → API_KEY → OPENROUTER_API_KEY, plus the rule that a
missing key must never trigger OpenRouter key *rotation* machinery when the
endpoint is not openrouter.ai (issue: OPENROUTER_API_KEY must not be required
for third-party endpoints).
"""

from __future__ import annotations

import pytest

import agentknit._core as _core
from agentknit.exceptions import AuthenticationError

ZAI = "https://api.z.ai/api/coding/paas/v4"
OR = "https://openrouter.ai/api/v1"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("Z_AI_KEY", raising=False)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)


def test_keyring_hit_wins(monkeypatch) -> None:
    monkeypatch.setattr("keyring.get_password", lambda s, u: "kr-key")
    got = _core._get_key_for_schema(
        {"keyring_service": "z.ai", "keyring_username": "api_key", "endpoint": ZAI}
    )
    assert got == "kr-key"


def test_keyring_miss_falls_back_to_uppercased_env(monkeypatch) -> None:
    monkeypatch.setattr("keyring.get_password", lambda s, u: None)
    monkeypatch.setenv("API_KEY", "env-key")
    got = _core._get_key_for_schema(
        {"keyring_service": "z.ai", "keyring_username": "api_key", "endpoint": ZAI}
    )
    assert got == "env-key"


def test_keyring_miss_raises_without_openrouter_fallback(monkeypatch) -> None:
    """Configured keyring source that yields nothing must raise, not fall
    through to OPENROUTER_API_KEY rotation."""
    monkeypatch.setattr("keyring.get_password", lambda s, u: None)
    called = []
    monkeypatch.setattr(_core, "get_api_key", lambda: called.append(1))
    with pytest.raises(AuthenticationError):
        _core._get_key_for_schema(
            {"keyring_service": "z.ai", "keyring_username": "api_key",
             "endpoint": ZAI}
        )
    assert not called


def test_key_env_hit(monkeypatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "m-key")
    got = _core._get_key_for_schema({"key_env": "MISTRAL_API_KEY", "endpoint": OR})
    assert got == "m-key"


def test_key_env_missing_raises(monkeypatch) -> None:
    with pytest.raises(AuthenticationError, match="MISTRAL_API_KEY"):
        _core._get_key_for_schema({"key_env": "MISTRAL_API_KEY", "endpoint": OR})


def test_openrouter_endpoint_uses_rotation(monkeypatch) -> None:
    monkeypatch.setattr(_core, "get_api_key", lambda: "or-key")
    got = _core._get_key_for_schema({"endpoint": OR})
    assert got == "or-key"


def test_non_openrouter_no_key_source_no_rotation(monkeypatch) -> None:
    """Third-party endpoint with no key source: never call ensure_api_key."""
    called = []
    monkeypatch.setattr(_core, "get_api_key", lambda: called.append(1))
    monkeypatch.setattr("agentknit.keys._get_raw_key", lambda n: None)
    with pytest.raises(AuthenticationError):
        _core._get_key_for_schema({"endpoint": ZAI})
    assert not called


def test_non_openrouter_generic_api_key(monkeypatch) -> None:
    """API_KEY is the generic channel: no provider's name in it."""
    monkeypatch.setenv("API_KEY", "generic")
    monkeypatch.setattr(_core, "get_api_key", lambda: (_ for _ in ()).throw(
        AssertionError("rotation must not run")))
    assert _core._get_key_for_schema({"endpoint": ZAI}) == "generic"


def test_generic_api_key_wins_over_openrouter(monkeypatch) -> None:
    """With both set, the generic name wins.

    A machine with OPENROUTER_API_KEY exported for OpenRouter work must not
    send that key to a third-party endpoint once the operator has said, by
    setting API_KEY, which key that endpoint takes.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("API_KEY", "the-endpoints-own-key")
    monkeypatch.setattr(_core, "get_api_key", lambda: (_ for _ in ()).throw(
        AssertionError("rotation must not run")))
    assert _core._get_key_for_schema({"endpoint": ZAI}) == "the-endpoints-own-key"


def test_generic_api_key_does_not_divert_openrouter(monkeypatch) -> None:
    """An OpenRouter endpoint still rotates, whatever API_KEY holds."""
    monkeypatch.setenv("API_KEY", "generic")
    monkeypatch.setattr(_core, "get_api_key", lambda: "rotated")
    assert _core._get_key_for_schema({"endpoint": OR}) == "rotated"


def test_non_openrouter_plain_openrouter_env(monkeypatch) -> None:
    """OPENROUTER_API_KEY still works as a plain channel.

    Kept on purpose: wrappers exported it as the generic channel before there
    was a generic name.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "plain")
    monkeypatch.setattr(_core, "get_api_key", lambda: (_ for _ in ()).throw(
        AssertionError("rotation must not run")))
    got = _core._get_key_for_schema({"endpoint": ZAI})
    assert got == "plain"


def test_run_uri_endpoint_does_not_require_key(monkeypatch) -> None:
    """run:// endpoints never reach key resolution at all."""
    from agentknit.openai_compat import SubprocessOpenAI
    client = _core.create_client(
        {"endpoint": "run:///bin/true", "model": "x", "inferred_tool_schema": []}
    )
    assert isinstance(client, SubprocessOpenAI)


def test_localhost_endpoint_is_not_openrouter(monkeypatch) -> None:
    assert not _core._endpoint_is_openrouter("http://localhost:8000/v1")
    assert _core._endpoint_is_openrouter(OR)
    assert _core._endpoint_is_openrouter("https://openrouter.ai/api/v1")
    assert not _core._endpoint_is_openrouter(None)
