"""Tests for HTTP 429 handling in agentknit.openai_compat.

A 429 with a header telling us when it's safe to retry (Retry-After,
retry-after-ms, or x-ratelimit-reset-requests) is retried automatically.
A 429 with none of those headers raises RateLimitError instead of looping
forever on an unknown delay — except on Z.ai, which reports the reset time
only in the JSON body ("Your limit will reset at 2026-08-16 05:42:29",
Beijing time), so the body is parsed as a last resort there.
"""

from __future__ import annotations

import datetime
from unittest.mock import patch

import pytest

from agentknit.exceptions import RateLimitError
from agentknit.openai_compat import (
    OpenAI,
    _retry_delay_for_z_ai,
    _retry_delay_seconds,
)

_ZAI_BODY = ('{"error":{"code":"1308","message":"Usage limit reached for '
             '5 hour. Your limit will reset at ')


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
    from email.utils import format_datetime

    future = format_datetime(
        datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=10)
    )
    delay = _retry_delay_seconds({"retry-after": future})
    assert delay is not None and 5 <= delay <= 15


# ── _retry_delay_for_z_ai ────────────────────────────────────────────────────

def _reset_at_in(seconds: float) -> str:
    """Timestamp string as Z.ai writes it: Beijing time (UTC+8), no zone."""
    reset = (datetime.datetime.now(datetime.timezone.utc)
             + datetime.timedelta(seconds=seconds)
             + datetime.timedelta(hours=8))
    return reset.strftime("%Y-%m-%d %H:%M:%S")


def test_z_ai_delay_parses_reset_timestamp():
    delay = _retry_delay_for_z_ai(_ZAI_BODY + _reset_at_in(120) + '"}}')
    assert delay is not None and 110 <= delay <= 130


def test_z_ai_delay_parses_anthropic_style_bracketed_message():
    delay = _retry_delay_for_z_ai(
        '{"type":"error","error":{"type":"rate_limit_error","code":"1308",'
        '"message":"[1308][Usage limit reached for 5 hour. Your limit will '
        'reset at ' + _reset_at_in(60) + '][20260816025434dc219c5ac73d44ec]"}}'
    )
    assert delay is not None and 50 <= delay <= 70


def test_z_ai_delay_none_without_reset_timestamp():
    assert _retry_delay_for_z_ai(
        '{"error":{"code":"1113","message":"Insufficient balance or no '
        'resource package. Please recharge."}}'
    ) is None


def test_z_ai_delay_none_on_empty_or_malformed_body():
    assert _retry_delay_for_z_ai(None) is None
    assert _retry_delay_for_z_ai("") is None
    assert _retry_delay_for_z_ai(_ZAI_BODY + "not-a-date" + '"}}') is None


def test_z_ai_delay_none_when_reset_is_too_far_away():
    # The quota window is 5h, so a reset a week out means the timestamp (or
    # its timezone) was misread — fail closed instead of sleeping for days.
    assert _retry_delay_for_z_ai(_ZAI_BODY + _reset_at_in(7 * 86400) + '"}}') is None


def test_z_ai_delay_zero_when_reset_time_already_passed():
    assert _retry_delay_for_z_ai(_ZAI_BODY + _reset_at_in(-30) + '"}}') == 0.0


# ── OpenAI._retry_post ───────────────────────────────────────────────────────

class _FakeResponse:
    def __init__(self, status_code: int, headers: dict, json_data: dict | None = None,
                 text: str = ""):
        self.status_code = status_code
        self.headers = headers
        self.ok = status_code < 400
        self._json = json_data or {}
        self.text = text

    def json(self):
        return self._json


def _client(base_url: str = "https://example.test/v1") -> OpenAI:
    return OpenAI(api_key="x", base_url=base_url, max_rpm=1000)


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


def test_retry_post_parses_z_ai_body_and_retries():
    client = _client("https://api.z.ai/api/coding/paas/v4")
    resp_429 = _FakeResponse(429, {}, text=_ZAI_BODY + _reset_at_in(0) + '"}}')
    resp_ok = _FakeResponse(200, {}, {"choices": []})
    calls = [resp_429, resp_ok]

    def _fake_post(*args, **kwargs):
        return calls.pop(0)

    with patch("agentknit.openai_compat.requests.post", side_effect=_fake_post):
        with patch("agentknit.openai_compat.time.sleep") as sleep:
            resp = client.chat.completions._retry_post("https://x", {}, {})
    assert resp is resp_ok
    assert sleep.called


def test_retry_post_ignores_body_on_non_zai_hosts():
    # Body parsing is gated on z.ai/bigmodel.cn hosts — a 429 body elsewhere
    # must still raise instead of being scraped for timestamps.
    client = _client()
    resp_429 = _FakeResponse(429, {}, text=_ZAI_BODY + _reset_at_in(60) + '"}}')
    with patch("agentknit.openai_compat.requests.post", return_value=resp_429):
        with pytest.raises(RateLimitError):
            client.chat.completions._retry_post("https://x", {}, {})


def test_create_raises_rate_limit_error_without_header():
    client = _client()
    resp_429 = _FakeResponse(429, {})
    with patch("agentknit.openai_compat.requests.post", return_value=resp_429):
        with pytest.raises(RateLimitError):
            client.chat.completions.create(model="m", messages=[])
