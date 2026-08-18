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
import requests

from agentknit._core import _rate_limit_wait_callback
from agentknit.exceptions import RateLimitError
from agentknit.openai_compat import (
    OpenAI,
    _retry_delay_for_z_ai,
    _retry_delay_seconds,
    _z_ai_error_details,
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


def test_z_ai_error_details_extracts_business_error_fields():
    assert _z_ai_error_details(
        '{"error":{"code":"1305","message":"The service may be temporarily overloaded"}}'
    ) == ("1305", "The service may be temporarily overloaded")


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


def test_retry_post_retries_read_timeouts_with_exponential_backoff():
    client = _client()
    resp_ok = _FakeResponse(200, {}, {"choices": []})
    calls = [requests.exceptions.ReadTimeout("read timed out")] * 5 + [resp_ok]

    with patch("agentknit.openai_compat.requests.post", side_effect=calls):
        with patch("agentknit.openai_compat.time.sleep") as sleep:
            resp = client.chat.completions._retry_post("https://x", {}, {})

    assert resp is resp_ok
    assert [call.args[0] for call in sleep.call_args_list] == [60, 120, 240, 480, 960]


def test_retry_post_reraises_after_five_read_timeout_retries():
    client = _client()
    timeout = requests.exceptions.ReadTimeout("read timed out")

    with patch("agentknit.openai_compat.requests.post", side_effect=[timeout] * 6):
        with patch("agentknit.openai_compat.time.sleep") as sleep:
            with pytest.raises(requests.exceptions.ReadTimeout):
                client.chat.completions._retry_post("https://x", {}, {})

    assert [call.args[0] for call in sleep.call_args_list] == [60, 120, 240, 480, 960]


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


def test_z_ai_non_retryable_rate_limit_preserves_business_error_details():
    client = _client("https://api.z.ai/api/coding/paas/v4")
    resp_429 = _FakeResponse(
        429, {}, text=(
            '{"error":{"code":"1305","message":"The service may be temporarily '
            'overloaded, please try again later"}}'
        ),
    )
    with patch("agentknit.openai_compat.requests.post", return_value=resp_429):
        with pytest.raises(RateLimitError) as caught:
            client.chat.completions._retry_post("https://x", {}, {})

    assert caught.value.error_code == "1305"
    assert caught.value.error_message == "The service may be temporarily overloaded, please try again later"


def test_create_raises_rate_limit_error_without_header():
    client = _client()
    resp_429 = _FakeResponse(429, {})
    with patch("agentknit.openai_compat.requests.post", return_value=resp_429):
        with pytest.raises(RateLimitError):
            client.chat.completions.create(model="m", messages=[])


def test_retry_post_calls_on_rate_limit_wait_before_sleeping():
    client = _client()
    resp_429 = _FakeResponse(429, {"retry-after": "2"})
    resp_ok = _FakeResponse(200, {}, {"choices": []})
    calls = [resp_429, resp_ok]
    seen = []

    def _fake_post(*args, **kwargs):
        return calls.pop(0)

    with patch("agentknit.openai_compat.requests.post", side_effect=_fake_post):
        with patch("agentknit.openai_compat.time.sleep") as sleep:
            resp = client.chat.completions._retry_post(
                "https://x", {}, {}, on_rate_limit_wait=lambda *a: seen.append(a)
            )

    assert resp is resp_ok
    assert sleep.called
    assert len(seen) == 1
    delay, resume_at, fmt = seen[0]
    assert delay == 2.0
    assert resume_at.tzinfo is not None
    assert "rate-limited" in fmt and "waiting 2.0s" in fmt


# ── _core._rate_limit_wait_callback ──────────────────────────────────────────

def test_rate_limit_wait_callback_emits_session_event():
    events = []
    session = {
        "on_event": lambda et, data: events.append((et, data)),
        "_event_handlers": {},
    }
    resume_at = datetime.datetime.now(datetime.timezone.utc)

    _rate_limit_wait_callback(session)(2.5, resume_at, "  [rate-limited] waiting 2.5s …")

    assert len(events) == 1
    event_type, data = events[0]
    assert event_type == "rate_limit_wait"
    assert data["delay"] == 2.5
    assert data["resume_at"] == resume_at.isoformat()
    assert data["fmt"] == "  [rate-limited] waiting 2.5s …"
