"""A streaming chat-completions request must ask for usage accounting.

The OpenAI chat-completions streaming contract only attaches a ``usage``
object to the final SSE chunk when the request carries
``stream_options: {"include_usage": true}``; without it the stream ends with
no usage at all.  That made every streamed response look like "no cache
accounting from the server" (e.g. Kimi k3, whose ``cached_tokens`` live on
the usage object), tripping the strict cache-proof warning on every call.
"""

from __future__ import annotations

import httpx

from agentknit.openai_compat import OpenAI


def _client_capture_payload(body: bytes) -> tuple[OpenAI, list[dict]]:
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body,
        )

    client = OpenAI(api_key="x", base_url="https://example.test/v1", max_rpm=1000)
    client.chat.completions._client._http = httpx.Client(transport=httpx.MockTransport(handler))
    return client, captured


_STREAM = (
    b'data: {"choices":[{"index":0,"delta":{"content":"hi"}}]}\n\n'
    b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
    b'"usage":{"prompt_tokens":100,"completion_tokens":1,"total_tokens":101,'
    b'"prompt_tokens_details":{"cached_tokens":64}}}\n\n'
    b"data: [DONE]\n\n"
)


def test_streaming_request_sets_include_usage():
    client, captured = _client_capture_payload(_STREAM)
    client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "hi"}],
        on_content_delta=lambda _: None,
    )
    assert captured[0]["stream"] is True
    assert captured[0]["stream_options"] == {"include_usage": True}


def test_streaming_response_parses_cache_accounting_from_usage_chunk():
    client, _ = _client_capture_payload(_STREAM)
    resp = client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "hi"}],
        on_content_delta=lambda _: None,
    )
    assert resp.usage.cached_tokens == 64
    assert resp.usage.has_cache_proof is True
