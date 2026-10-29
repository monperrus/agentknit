"""A stalled SSE stream (server goes silent mid-response) must surface as a
retryable read timeout instead of hanging ``iter_lines()`` for the full flat
client timeout.

The stall watchdog in openai_compat closes a streaming response that has
produced no line for ``_STREAM_STALL_TIMEOUT_SECONDS``; the closed stream
makes the consumer raise ``httpx.ReadTimeout``, which ``_retry_post``
already retries (1, 2, 4, 8, 16 minutes) before surfacing the error.

Positive: silent stream raises ReadTimeout well under the flat timeout.
Negative: an active stream flows through untouched by the watchdog.
Boundary: a gap shorter than the stall bound is tolerated, then data flows.
"""

from __future__ import annotations

import threading
import time

import httpx
import pytest

from agentknit import openai_compat
from agentknit.openai_compat import OpenAI, _StallWatchdog


class _StalledStream(httpx.SyncByteStream):
    """Yields one SSE chunk, then blocks forever (until the stream is closed
    from another thread, which releases the event and ends iteration)."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self._released = threading.Event()

    def __iter__(self):
        yield from self._chunks
        # The silent hop: stay here until the watchdog closes the response.
        self._released.wait(timeout=30)

    def close(self) -> None:
        self._released.set()


def _client_with_stream(stream: httpx.SyncByteStream, **kwargs) -> OpenAI:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              stream=stream)

    client = OpenAI(api_key="x", base_url="https://example.test/v1",
                    max_rpm=1000, **kwargs)
    client.chat.completions._client._http = httpx.Client(
        transport=httpx.MockTransport(handler))
    return client


def test_watchdog_closes_silent_response():
    """Known-positive: a response with no activity is closed after the bound."""
    stream = _StalledStream([])
    resp = httpx.Response(200, stream=stream, request=httpx.Request("GET", "http://x"))
    watchdog = _StallWatchdog(resp, stall_timeout=0.2)
    watchdog.start()
    t0 = time.monotonic()
    # The consumer blocks in iter_lines(); close() from the watchdog must
    # release it within ~2s, not the flat 300s client timeout.
    list(resp.iter_lines())
    elapsed = time.monotonic() - t0
    watchdog.stop()
    assert elapsed < 5.0, f"watchdog did not close the silent stream ({elapsed:.1f}s)"


def test_stalled_stream_is_retried(monkeypatch):
    """A mid-stream stall raises ReadTimeout inside create(), which the retry
    loop treats like a pre-response timeout: it sleeps and retries, so the
    observable behaviour with retries intact is a *delayed* error, not a hang.
    Patch the retry delay to ~0 and count attempts to prove the retry fires."""
    monkeypatch.setattr(openai_compat, "_STREAM_STALL_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(openai_compat, "_STALL_WATCHDOG_POLL_SECONDS", 0.05)
    monkeypatch.setattr(openai_compat, "_MAX_READ_TIMEOUT_RETRIES", 1)
    monkeypatch.setattr(openai_compat, "_READ_TIMEOUT_INITIAL_DELAY_SECONDS", 0.01)
    attempts = 0

    class _CountingStall(httpx.SyncByteStream):
        def __init__(self) -> None:
            self._released = threading.Event()

        def __iter__(self):
            nonlocal attempts
            attempts += 1
            yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
            self._released.wait(timeout=30)

        def close(self) -> None:
            self._released.set()

    def handler(request: httpx.Request) -> httpx.Response:
        # A fresh stalled stream per attempt — a real retry re-opens a new
        # connection, so the mock must too.
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              stream=_CountingStall())

    client = OpenAI(api_key="x", base_url="https://example.test/v1", max_rpm=1000)
    client.chat.completions._client._http = httpx.Client(
        transport=httpx.MockTransport(handler))
    t0 = time.monotonic()
    with pytest.raises(httpx.ReadTimeout):
        client.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "hi"}],
            on_content_delta=lambda s: None)
    elapsed = time.monotonic() - t0
    assert attempts == 2, f"stall was not retried ({attempts} attempts)"
    assert elapsed < 10.0


def test_stalled_stream_retries_then_raises(monkeypatch):
    """With no retry budget left, the stall surfaces as ReadTimeout quickly."""
    monkeypatch.setattr(openai_compat, "_STREAM_STALL_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(openai_compat, "_STALL_WATCHDOG_POLL_SECONDS", 0.05)
    monkeypatch.setattr(openai_compat, "_MAX_READ_TIMEOUT_RETRIES", 0)
    chunks = [b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n']
    client = _client_with_stream(_StalledStream(chunks))

    t0 = time.monotonic()
    with pytest.raises(httpx.ReadTimeout):
        client.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "hi"}],
            on_content_delta=lambda s: None)
    assert time.monotonic() - t0 < 5.0


def test_active_stream_unaffected(monkeypatch):
    """Known-negative: a stream that keeps producing lines completes normally."""
    monkeypatch.setattr(openai_compat, "_STREAM_STALL_TIMEOUT_SECONDS", 0.5)
    monkeypatch.setattr(openai_compat, "_STALL_WATCHDOG_POLL_SECONDS", 0.05)

    class _SlowButActive(httpx.SyncByteStream):
        def __iter__(self):
            yield b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
            time.sleep(0.2)  # boundary: gap < stall bound
            yield b'data: {"choices":[{"delta":{"content":"b"}}]}\n\n'
            yield b"data: [DONE]\n\n"

        def close(self) -> None:
            pass

    client = _client_with_stream(_SlowButActive())
    pieces: list[str] = []
    resp = client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "hi"}],
        on_content_delta=pieces.append)
    assert pieces == ["a", "b"]
    assert resp.choices[0].message.content == "ab"
