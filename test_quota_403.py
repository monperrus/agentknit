"""Tests for HTTP 403 handling in agentknit.openai_compat.

Kimi's Coding Plan endpoint reports an exhausted quota window as HTTP 403
with no Retry-After header — the same condition other providers report as
429. The reset time lives on ``GET <base>/usages``, so a 403 there is
resolved by asking that endpoint and waiting through the window. Any other
403 aborts, but with the provider's body in the error message instead of a
bare "403 Client Error: Forbidden for url: …".
"""

from __future__ import annotations

import datetime
from unittest.mock import patch

import pytest
import requests

from agentknit.exceptions import RateLimitError
from agentknit.openai_compat import OpenAI, _kimi_quota_reset_delay

_KIMI_BASE = "https://api.kimi.com/coding/v1"


def _iso_in(seconds: float) -> str:
    reset = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=seconds)
    return reset.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _usages(remaining_5h: str, reset_in: float = 600.0) -> dict:
    return {
        "usage": {"limit": "100", "remaining": "79", "resetTime": _iso_in(5 * 86400)},
        "limits": [{
            "window": {"duration": 300, "timeUnit": "TIME_UNIT_MINUTE"},
            "detail": {"limit": "100", "remaining": remaining_5h,
                       "resetTime": _iso_in(reset_in)},
        }],
    }


class _FakeResponse:
    def __init__(self, status_code: int, headers: dict | None = None,
                 json_data: dict | None = None, text: str = "",
                 url: str = "https://x/chat/completions", reason: str = "Forbidden"):
        self.status_code = status_code
        self.headers = headers or {}
        self.ok = status_code < 400
        self._json = json_data or {}
        self.text = text
        self.url = url
        self.reason = reason

    def json(self):
        return self._json


def _client(base_url: str = _KIMI_BASE) -> OpenAI:
    return OpenAI(api_key="x", base_url=base_url, max_rpm=1000)


# ── _kimi_quota_reset_delay ──────────────────────────────────────────────────

def test_kimi_delay_reads_exhausted_window_reset():
    resp = _FakeResponse(200, json_data=_usages("0", reset_in=600))
    with patch("agentknit.openai_compat.requests.get", return_value=resp):
        delay = _kimi_quota_reset_delay(_KIMI_BASE, {})
    assert delay is not None and 570 <= delay <= 600


def test_kimi_delay_none_when_no_window_is_exhausted():
    # Quota left but still a 403 — the refusal is about permissions, not
    # the window, so waiting would be pointless.
    resp = _FakeResponse(200, json_data=_usages("42"))
    with patch("agentknit.openai_compat.requests.get", return_value=resp):
        assert _kimi_quota_reset_delay(_KIMI_BASE, {}) is None


def test_kimi_delay_none_when_usages_endpoint_fails():
    with patch("agentknit.openai_compat.requests.get",
               side_effect=requests.exceptions.ConnectionError("boom")):
        assert _kimi_quota_reset_delay(_KIMI_BASE, {}) is None


def test_kimi_delay_none_when_reset_is_implausibly_far():
    resp = _FakeResponse(200, json_data=_usages("0", reset_in=7 * 86400))
    with patch("agentknit.openai_compat.requests.get", return_value=resp):
        assert _kimi_quota_reset_delay(_KIMI_BASE, {}) is None


def test_kimi_delay_zero_when_window_already_reset():
    resp = _FakeResponse(200, json_data=_usages("0", reset_in=-30))
    with patch("agentknit.openai_compat.requests.get", return_value=resp):
        assert _kimi_quota_reset_delay(_KIMI_BASE, {}) == 0.0


# ── _retry_post ──────────────────────────────────────────────────────────────

def test_403_quota_waits_for_the_window_then_succeeds():
    client = _client()
    resp_403 = _FakeResponse(403, text="quota exhausted")
    resp_ok = _FakeResponse(200, json_data={"choices": []})
    posts = [resp_403, resp_ok]
    seen = []

    with patch("agentknit.openai_compat.requests.post", side_effect=lambda *a, **k: posts.pop(0)):
        with patch("agentknit.openai_compat.requests.get",
                   return_value=_FakeResponse(200, json_data=_usages("0", reset_in=120))):
            with patch("agentknit.openai_compat.time.sleep") as sleep:
                resp = client.chat.completions._retry_post(
                    "https://x", {}, {}, on_rate_limit_wait=lambda *a: seen.append(a))

    assert resp is resp_ok
    assert sleep.called
    assert len(seen) == 1 and "quota exhausted (HTTP 403)" in seen[0][2]


def test_403_quota_gives_up_after_the_retry_cap():
    client = _client()
    resp_403 = _FakeResponse(403, text='{"error":{"message":"quota exceeded"}}')

    with patch("agentknit.openai_compat.requests.post", return_value=resp_403):
        with patch("agentknit.openai_compat.requests.get",
                   return_value=_FakeResponse(200, json_data=_usages("0", reset_in=60))):
            with patch("agentknit.openai_compat.time.sleep") as sleep:
                with pytest.raises(RateLimitError) as caught:
                    client.chat.completions._retry_post("https://x", {}, {})

    assert sleep.call_count == 2
    assert caught.value.status_code == 403
    assert "quota exceeded" in str(caught.value)


def test_403_with_retry_after_header_is_waited_through_on_any_host():
    client = _client("https://example.test/v1")
    posts = [_FakeResponse(403, {"retry-after": "1"}), _FakeResponse(200, json_data={})]

    with patch("agentknit.openai_compat.requests.post", side_effect=lambda *a, **k: posts.pop(0)):
        with patch("agentknit.openai_compat.time.sleep") as sleep:
            resp = client.chat.completions._retry_post("https://x", {}, {})

    assert resp.status_code == 200
    assert sleep.called


def test_403_permission_denied_raises_http_error_with_body():
    # No quota window exhausted and no quota wording: a real permission
    # refusal, whose reason must reach the caller.
    client = _client()
    resp_403 = _FakeResponse(403, text='{"error":{"message":"invalid api key"}}')

    with patch("agentknit.openai_compat.requests.post", return_value=resp_403):
        with patch("agentknit.openai_compat.requests.get",
                   return_value=_FakeResponse(200, json_data=_usages("42"))):
            with pytest.raises(requests.exceptions.HTTPError) as caught:
                client.chat.completions._retry_post("https://x", {}, {})

    assert "invalid api key" in str(caught.value)
    assert "403" in str(caught.value)


def test_403_on_non_kimi_host_never_queries_usages():
    client = _client("https://example.test/v1")
    resp_403 = _FakeResponse(403, text="nope")

    with patch("agentknit.openai_compat.requests.post", return_value=resp_403):
        with patch("agentknit.openai_compat.requests.get") as get:
            with pytest.raises(requests.exceptions.HTTPError):
                client.chat.completions._retry_post("https://x", {}, {})

    assert not get.called


def test_other_error_statuses_carry_the_body_too():
    client = _client("https://example.test/v1")
    resp_400 = _FakeResponse(400, text="context length exceeded", reason="Bad Request")

    with patch("agentknit.openai_compat.requests.post", return_value=resp_400):
        with pytest.raises(requests.exceptions.HTTPError) as caught:
            client.chat.completions.create(model="m", messages=[])

    assert "context length exceeded" in str(caught.value)
