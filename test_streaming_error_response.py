"""A non-2xx status on a *streaming* request must not crash with httpx's
``ResponseNotRead`` ("Attempted to access streaming response content,
without having called `read()`").

httpx.Client.send(request, stream=True) leaves the body unread — accessing
.text/.json() before .read() raises. _retry_post's error branches (429,
403, retryable 5xx) and the caller's status>=400 check all inspect the
body, so a streaming request must read it eagerly once it knows it's an
error rather than a real SSE stream.
"""

from __future__ import annotations

import httpx
import pytest

from agentknit.openai_compat import OpenAI


class _UnreadStream(httpx.SyncByteStream):
    """A byte stream that behaves like a real (unread) network response,
    unlike httpx.MockTransport's default in-memory Response, which is
    pre-consumed and hides the bug this test guards against."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def __iter__(self):
        yield from self._chunks

    def close(self) -> None:
        pass


def _client_with_status(status_code: int, body: bytes) -> OpenAI:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, headers={"content-type": "application/json"},
                              stream=_UnreadStream([body]))

    client = OpenAI(api_key="x", base_url="https://example.test/v1", max_rpm=1000)
    client.chat.completions._client._http = httpx.Client(transport=httpx.MockTransport(handler))
    return client


def test_streaming_400_raises_http_error_not_response_not_read():
    body = b'{"error":{"message":"bad request"}}'
    client = _client_with_status(400, body)

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}],
                                        on_content_delta=lambda s: None)
    assert "bad request" in str(exc_info.value)


def test_streaming_403_raises_http_error_not_response_not_read():
    body = b'{"error":{"message":"forbidden"}}'
    client = _client_with_status(403, body)

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}],
                                        on_content_delta=lambda s: None)
    assert "forbidden" in str(exc_info.value)
