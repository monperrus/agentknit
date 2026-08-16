"""Tests for custom tool calls (``type == "custom"``) in the response path.

A custom tool call arrives as
``{"id": ..., "type": "custom", "custom": {"name": ..., "input": "<raw text>"}}``
and must not be JSON-decoded: the raw input string is dispatched as a single
``input`` kwarg (issue #22).
"""

from __future__ import annotations

import json

from agentknit.openai_compat import _parse_response, _parse_tool_call


CUSTOM_CALL = {
    "id": "call_1",
    "type": "custom",
    "custom": {"name": "t_grammar", "input": "SELECT * FROM users;"},
}

FUNCTION_CALL = {
    "id": "call_2",
    "type": "function",
    "function": {"name": "read_file", "arguments": "{\"path\": \"/tmp\"}"},
}


def _resp(tool_calls: list[dict]) -> object:
    return _parse_response({
        "choices": [{"message": {"role": "assistant", "content": None,
                                 "tool_calls": tool_calls}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    })


def test_parse_custom_tool_call() -> None:
    """custom_tool_call is decoded with type='custom' and raw input preserved."""
    resp = _resp([CUSTOM_CALL])
    tc = resp.choices[0].message.tool_calls[0]
    assert tc.type == "custom"
    assert tc.id == "call_1"
    assert tc.function.name == "t_grammar"
    assert tc.custom_input == "SELECT * FROM users;"
    # Raw input mirrored into function.arguments for legacy readers.
    assert tc.function.arguments == "SELECT * FROM users;"


def test_parse_function_tool_call_unchanged() -> None:
    """Function calls keep the exact same representation as before."""
    resp = _resp([FUNCTION_CALL])
    tc = resp.choices[0].message.tool_calls[0]
    assert tc.type == "function"
    assert tc.custom_input is None
    assert tc.function.name == "read_file"
    assert json.loads(tc.function.arguments) == {"path": "/tmp"}


def test_parse_mixed_calls() -> None:
    """A response can mix function and custom calls."""
    resp = _resp([FUNCTION_CALL, CUSTOM_CALL])
    tcs = resp.choices[0].message.tool_calls
    assert [t.type for t in tcs] == ["function", "custom"]


def test_parse_non_string_custom_input() -> None:
    """Non-string custom input (object) is serialized, not dropped."""
    tc = _parse_tool_call({"id": "c", "type": "custom",
                           "custom": {"name": "t_x", "input": {"a": 1}}})
    assert tc.type == "custom"
    assert json.loads(tc.custom_input) == {"a": 1}


def test_history_item_round_trip() -> None:
    """Custom calls are re-serialized in their original shape for history."""
    from agentknit._core import _tool_call_history_item
    tc = _parse_tool_call(CUSTOM_CALL)
    item = _tool_call_history_item(tc)
    assert item == {"id": "call_1", "type": "custom",
                    "custom": {"name": "t_grammar",
                               "input": "SELECT * FROM users;"}}


def test_history_item_function_unchanged() -> None:
    """Function calls serialize exactly as before."""
    from agentknit._core import _tool_call_history_item
    tc = _parse_tool_call(FUNCTION_CALL)
    assert _tool_call_history_item(tc) == {
        "id": "call_2", "type": "function",
        "function": {"name": "read_file",
                     "arguments": "{\"path\": \"/tmp\"}"},
    }


def test_normalise_for_resume_flattens_custom_calls() -> None:
    """Resume flattening renders custom calls as readable text."""
    from agentknit._core import _normalise_for_resume
    msgs = [
        {"role": "assistant",
         "tool_calls": [{"id": "c", "type": "custom",
                         "custom": {"name": "t_g", "input": "raw text"}}]},
        {"role": "tool", "tool_call_id": "c", "content": "42"},
    ]
    out = _normalise_for_resume(msgs)
    assert out == [
        {"role": "assistant",
         "content": "[Tool call: t_g(raw text)]\n\n[Tool result: 42]",
         "ts": None},
    ]


def test_run_turn_dispatches_custom_call_without_json(monkeypatch) -> None:
    """_run_turn dispatches the raw custom input as {'input': ...}."""
    from agentknit._core import _run_turn
    import agentknit._core as core

    seen: list[tuple[str, dict]] = []

    def fake_handle(name, args, session, *, call_id=None):
        seen.append((name, args))
        return "42"

    monkeypatch.setattr(core, "_handle_tool_call", fake_handle)

    calls = []

    class FakeResp:
        class choices:  # noqa: N801
            pass

    # Build a minimal response object with a custom tool call.
    from agentknit.openai_compat import _Response, _Choice, _Message, _ToolCall, _Function, _Usage

    def fake_complete(client, session, **kwargs):
        if calls:
            # Second call: plain final answer.
            msg = _Message("assistant", "done", None)
            resp = _Response([_Choice(msg)], _Usage(1, 1, 2))
            resp.usage = _Usage(prompt_tokens=1, completion_tokens=1,
                                total_tokens=2, has_cache_proof=True)
            return resp
        calls.append(1)
        tc = _ToolCall("c1", _Function("t_grammar", "not json at all!"),
                       type="custom", custom_input="not json at all!")
        msg = _Message("assistant", None, [tc])
        resp = _Response([_Choice(msg)], _Usage(1, 1, 2))
        resp.usage = _Usage(prompt_tokens=1, completion_tokens=1,
                            total_tokens=2, has_cache_proof=True)
        return resp

    monkeypatch.setattr(core, "_complete", fake_complete)

    session = {
        "messages": [],
        "tools": [],
        "structured": True,
        "usage_totals": {"prompt": 0, "completion": 0, "total": 0,
                         "cached": 0, "cache_write": 0},
        "tool_dispatch": {"t_grammar": {}},
        "non_interactive": True,
        "cache_key": "test",
        "options": ["exclude-prompt_cache_key"],
        "on_event": lambda *a, **k: None,
    }

    class _NullJournal:
        def turn_start(self, *a, **k): ...
        def message(self, *a, **k): ...
        def turn_end(self, *a, **k): ...

    session["_journal"] = _NullJournal()
    session["strict_cache_proof"] = False

    import tempfile
    from pathlib import Path
    session["log_path"] = Path(tempfile.mkdtemp()) / "log.jsonl"
    session["session_id"] = "test-session"

    result = _run_turn(object(), "test/model", session, "run the grammar tool")
    assert seen == [("t_grammar", {"input": "not json at all!"})]
    assert result.final_reply == "done"
