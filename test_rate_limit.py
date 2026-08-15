"""Tests for HTTP 429 handling in agentknit.openai_compat.

A 429 with a header telling us when it's safe to retry (Retry-After,
retry-after-ms, or x-ratelimit-reset-requests) is retried automatically.
A 429 with none of those headers raises RateLimitError instead of looping
forever on an unknown delay.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agentknit.exceptions import RateLimitError
from agentknit.openai_compat import OpenAI, _retry_delay_seconds


# ── _retry_delay_seconds ─────────────────────────────────────────────────────

def test_retry_delay_prefers_retry_after_ms():
    assert _retry_delay_seconds({"retry-after-ms": "1500"}) == 1.5


def test_retry_delay_falls_back_to_retry_after_seconds():
    assert _retry_delay_seconds({"retry-after": "3"}) == 3.0


def test_retry_delay_falls_back_to_ratelimit_reset():
    assert _retry_delay_seconds({"x-ratelimit-reset-requests": "2.5"}) == 2.5


def test_retry_delay_none_when_no_headers():
    assert _retry_delay_seconds({}) is None


def test_retry_delay_none_on_garbage_values():
    assert _retry_delay_seconds({"retry-after-ms": "not-a-number"}) is None


def test_retry_delay_parses_http_date():
    import datetime
    from email.utils import format_datetime

    future = format_datetime(
        datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=10)
    )
    delay = _retry_delay_seconds({"retry-after": future})
    assert delay is not None and 5 <= delay <= 15


# ── OpenAI._retry_post ───────────────────────────────────────────────────────

class _FakeResponse:
    def __init__(self, status_code: int, headers: dict, json_data: dict | None = None):
        self.status_code = status_code
        self.headers = headers
        self.ok = status_code < 400
        self._json = json_data or {}

    def json(self):
        return self._json


def _client() -> OpenAI:
    return OpenAI(api_key="x", base_url="https://example.test/v1", max_rpm=1000)


def test_retry_post_raises_rate_limit_error_without_header():
    client = _client()
    resp_429 = _FakeResponse(429, {})
    with patch("agentknit.openai_compat.requests.post", return_value=resp_429):
        with pytest.raises(RateLimitError):
            client.chat.completions._retry_post("https://x", {}, {})


def test_retry_post_retries_then_succeeds_with_header():
    client = _client()
    resp_429 = _FakeResponse(429, {"retry-after": "0"})
    resp_ok = _FakeResponse(200, {}, {"choices": []})
    calls = [resp_429, resp_ok]

    def _fake_post(*args, **kwargs):
        return calls.pop(0)

    with patch("agentknit.openai_compat.requests.post", side_effect=_fake_post):
        with patch("agentknit.openai_compat.time.sleep"):
            resp = client.chat.completions._retry_post("https://x", {}, {})
    assert resp is resp_ok


def test_create_raises_rate_limit_error_without_header():
    client = _client()
    resp_429 = _FakeResponse(429, {})
    with patch("agentknit.openai_compat.requests.post", return_value=resp_429):
        with pytest.raises(RateLimitError):
            client.chat.completions.create(model="m", messages=[])
