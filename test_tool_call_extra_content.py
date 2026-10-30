"""Provider extras on a tool call must be replayed in the assistant history.

Gemini 3 attaches a ``thought_signature`` to every function call it returns,
inside an ``extra_content`` sibling of the call:

    {"id": "call_1", "type": "function",
     "function": {"name": "exec_shell", "arguments": "{...}"},
     "extra_content": {"google": {"thought_signature": "..."}}}

The signature is binding for the next request: a conversation that ends with
a function *response* (the tool message agentknit sends right after a call)
is rejected with

    400 INVALID_ARGUMENT "Function call is missing a thought_signature in
    functionCall parts (...) Additional data, function call `...`, position 2"

when the previous function call is replayed without it -- i.e. every tool call
works once and then the follow-up turn fails.  agentknit dropped
``extra_content`` when it serialized the call into the history, so Gemini 3
"chats but cannot tool call".

The contract under test: any field of a returned tool call that is not one
agentknit interprets itself (``id``, ``index``, ``type``, ``function``,
``custom``) is carried, unchanged, into the assistant history message.
"""

from __future__ import annotations

import json

import httpx

from agentknit._core import _normalise_for_resume, _tool_call_history_item
from agentknit.openai_compat import OpenAI, _parse_tool_call

SIGNATURE = "gd0qY0sFjk0K1oV3vQ=="
EXTRA = {"google": {"thought_signature": SIGNATURE}}


def _client(body: bytes, content_type: str = "application/json") -> OpenAI:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": content_type}, content=body)

    client = OpenAI(api_key="x", base_url="https://example.test/v1", max_rpm=1000)
    client.chat.completions._client._http = httpx.Client(transport=httpx.MockTransport(handler))
    return client


_NON_STREAMING = json.dumps({
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "exec_shell", "arguments": "{\"command\":\"ls\"}"},
                "extra_content": EXTRA,
            }],
        },
    }],
    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
}).encode()

_STREAMING = (
    b'data: {"choices":[{"index":0,"delta":{"role":"assistant","tool_calls":'
    b'[{"index":0,"id":"call_1","type":"function","extra_content":{"google":'
    b'{"thought_signature":"' + SIGNATURE.encode() + b'"}},'
    b'"function":{"name":"exec_shell","arguments":"{\\"command\\":\\"ls\\"}"}}]}}]}\n\n'
    b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    b"data: [DONE]\n\n"
)


def test_non_streaming_call_keeps_the_provider_extra() -> None:
    client = _client(_NON_STREAMING)
    resp = client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "hi"}])
    tc = resp.choices[0].message.tool_calls[0]
    assert tc.extra == {"extra_content": EXTRA}
    item = _tool_call_history_item(tc)
    assert item["extra_content"] == EXTRA
    assert item["id"] == "call_1"
    assert item["function"]["name"] == "exec_shell"


def test_streamed_call_keeps_the_provider_extra() -> None:
    client = _client(_STREAMING, content_type="text/event-stream")
    resp = client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "hi"}],
        on_content_delta=lambda _: None)
    tc = resp.choices[0].message.tool_calls[0]
    assert tc.extra == {"extra_content": EXTRA}
    assert _tool_call_history_item(tc)["extra_content"] == EXTRA


def test_streamed_index_is_not_replayed_as_a_provider_extra() -> None:
    """``index`` positions the delta in the stream; it is not part of the call."""
    client = _client(_STREAMING, content_type="text/event-stream")
    resp = client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "hi"}],
        on_content_delta=lambda _: None)
    assert "index" not in _tool_call_history_item(resp.choices[0].message.tool_calls[0])


def test_a_plain_call_gains_no_extra_field() -> None:
    """Providers that send nothing extra must not see a new empty object."""
    tc = _parse_tool_call({"id": "c", "type": "function",
                           "function": {"name": "f", "arguments": "{}"}})
    assert tc.extra is None
    assert _tool_call_history_item(tc) == {
        "id": "c", "type": "function",
        "function": {"name": "f", "arguments": "{}"}}


def test_custom_call_keeps_the_provider_extra() -> None:
    tc = _parse_tool_call({"id": "c", "type": "custom",
                           "custom": {"name": "f", "input": "raw"},
                           "extra_content": EXTRA})
    assert _tool_call_history_item(tc) == {
        "id": "c", "type": "custom", "custom": {"name": "f", "input": "raw"},
        "extra_content": EXTRA}


def test_extra_survives_resume_normalisation() -> None:
    """A resumed session replays the same history, signature included."""
    msgs = [
        {"role": "user", "content": "list the files"},
        {"role": "assistant", "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "exec_shell", "arguments": "{}"},
            "extra_content": EXTRA}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "[]"},
        {"role": "user", "content": "thanks"},
    ]
    normalised = _normalise_for_resume(msgs)
    assistant = next(m for m in normalised if m["role"] == "assistant")
    assert assistant["tool_calls"][0]["extra_content"] == EXTRA
