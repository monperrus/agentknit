"""Deferred tool loading: kimi (system-message tools) and glm (tool_reference) modes."""

from __future__ import annotations

import json

import pytest

import agentknit._core as core
from agentknit import Tool, build_tool_spec, register_tools_in_library
from agentknit import _deferred_tools as deferred
from agentknit.openai_compat import (
    SubprocessOpenAI, _BaseURL, _Choice, _Function, _Message, _Response, _SubprocessChat,
    _SubprocessCompletions, _ToolCall, _Usage,
)


def t_weather(city: str) -> tuple[str, dict]:
    return f"{city}: 17C", {"result": "ok"}


def t_noop(query: str) -> tuple[str, dict]:
    return "noop", {"result": "ok"}


TOOLS = [
    Tool("get_weather", "Get the current weather for a city.", t_weather, deferred=True),
    Tool("get_stock_price", "Latest stock price for a ticker.", t_noop, deferred=True),
    Tool("echo", "Echo text back.", t_noop),
]


class ScriptedClient(SubprocessOpenAI):
    """Returns canned replies in order and records every request."""

    def __init__(self, replies: list[_Message]) -> None:
        self._binary_path = "stub"
        self.base_url = _BaseURL("")
        self.chat = _SubprocessChat(self)
        self.replies = list(replies)
        self.requests: list[dict] = []

    def _complete(self, model: str, messages: list[dict], **kwargs) -> _Response:
        self.requests.append({"messages": json.loads(json.dumps(messages)), **kwargs})
        usage = _Usage(prompt_tokens=10, completion_tokens=1, total_tokens=11,
                       cached_tokens=5, has_cache_proof=True)
        return _Response([_Choice(self.replies.pop(0))], usage)


@pytest.fixture(autouse=True)
def _stub_completions(monkeypatch):
    def _create(self, *, model, messages, **kwargs):
        return self._client._complete(model, messages, **kwargs)
    monkeypatch.setattr(_SubprocessCompletions, "create", _create)


def call(cid: str, name: str, **args) -> _Message:
    return _Message("assistant", None, [_ToolCall(cid, _Function(name, json.dumps(args)))])


def answer(text: str) -> _Message:
    return _Message("assistant", text, None)


def session_for(mode: str | None, tmp_path) -> dict:
    schema_tools, dispatch = build_tool_spec(TOOLS)
    register_tools_in_library(TOOLS)
    # Same shape run_agent builds from direct Tool definitions.
    schema = {"model": "test/model", "endpoint": "https://api.test/v1",
              "inferred_tool_schema": schema_tools, "tool_dispatch": dispatch}
    return core.init_session(schema, non_interactive=True, strict_cache_proof=False,
                             bare=True, session_dir=tmp_path, tool_loading=mode)


def names(tools: list[dict]) -> list[str]:
    return [deferred.tool_name(t) for t in tools]


def test_build_tool_spec_marks_deferred_inside_function() -> None:
    schema, _ = build_tool_spec(TOOLS)
    assert schema[0]["function"]["defer_loading"] is True
    assert "defer_loading" not in schema[2]["function"]


def test_eager_default_strips_marker_and_keeps_all_tools(tmp_path) -> None:
    s = session_for(None, tmp_path)
    assert names(s["tools"]) == ["get_weather", "get_stock_price", "echo"]
    assert not any(deferred.is_deferred(t) for t in s["tools"])


def test_keyword_search_ranks_by_overlap() -> None:
    catalog, _ = build_tool_spec(TOOLS[:2])
    assert deferred.keyword_search("weather in Paris", catalog) == ["get_weather"]
    assert deferred.keyword_search("zzz", catalog) == []


def test_kimi_reveals_with_trailing_system_message(tmp_path) -> None:
    s = session_for("kimi", tmp_path)
    assert names(s["tools"]) == ["search_tools", "echo"]
    client = ScriptedClient([
        call("c1", "search_tools", query="weather"),
        call("c2", "get_weather", city="Paris"),
        answer("Paris: 17C"),
    ])
    result = core.run_turn(client, "test/model", s, "Weather in Paris?")
    assert result.final_reply == "Paris: 17C"
    # The request after the search carries the reveal at the very end, and
    # `tools` is byte-identical to the first request (cache prefix intact).
    second = client.requests[1]
    assert second["tools"] == client.requests[0]["tools"]
    last = second["messages"][-1]
    assert last["role"] == "system" and "content" not in last
    assert names(last["tools"]) == ["get_weather"]
    assert not deferred.is_deferred(last["tools"][0])
    assert second["messages"][-2]["role"] == "tool"
    # The revealed tool really ran.
    assert "Paris: 17C" in client.requests[2]["messages"][-1]["content"]


def test_kimi_does_not_reveal_twice(tmp_path) -> None:
    s = session_for("kimi", tmp_path)
    client = ScriptedClient([
        call("c1", "search_tools", query="weather"),
        call("c2", "search_tools", query="weather city"),
        answer("done"),
    ])
    core.run_turn(client, "test/model", s, "Weather?")
    reveals = [m for m in s["messages"] if m.get("role") == "system" and m.get("tools")]
    assert len(reveals) == 1


def test_glm_returns_tool_references_and_keeps_tools_deferred(tmp_path) -> None:
    s = session_for("glm", tmp_path)
    assert names(s["tools"]) == ["search_tools", "echo", "get_weather", "get_stock_price"]
    assert [deferred.is_deferred(t) for t in s["tools"]] == [False, False, True, True]
    client = ScriptedClient([
        call("c1", "search_tools", query="weather"),
        call("c2", "get_weather", city="Paris"),
        answer("Paris: 17C"),
    ])
    result = core.run_turn(client, "test/model", s, "Weather in Paris?")
    assert result.final_reply == "Paris: 17C"
    tool_msg = client.requests[1]["messages"][-1]
    assert tool_msg["role"] == "tool"
    assert tool_msg["content"] == [{"type": "tool_reference", "name": "get_weather"}]
    assert client.requests[1]["tools"] == client.requests[0]["tools"]


def test_glm_compaction_rewrites_references_and_undefers(tmp_path) -> None:
    s = session_for("glm", tmp_path)
    client = ScriptedClient([
        call("c1", "search_tools", query="weather"),
        answer("found it"),
    ])
    core.run_turn(client, "test/model", s, "Find a weather tool")
    s["messages"].append({"role": "user", "content": "next"})
    prefix = deferred.before_compaction(s["messages"])
    assert not any(isinstance(m.get("content"), list) for m in prefix)
    # Simulate compaction having summarized the reference away.
    s["messages"] = [m for m in s["messages"] if not isinstance(m.get("content"), list)]
    deferred.after_compaction(s)
    weather = next(t for t in s["tools"] if deferred.tool_name(t) == "get_weather")
    stock = next(t for t in s["tools"] if deferred.tool_name(t) == "get_stock_price")
    assert not deferred.is_deferred(weather) and deferred.is_deferred(stock)


def test_unknown_mode_rejected(tmp_path) -> None:
    with pytest.raises(ValueError):
        session_for("anthropic", tmp_path)
