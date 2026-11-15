"""Regression tests for deterministic tool-loop detection and automatic breaks."""

from __future__ import annotations

import json
from types import SimpleNamespace as NS
from typing import Any

import pytest

import agentknit
from agentknit import init_session, run_turn, subscribe


def _call(name: str, args: dict[str, Any], call_id: str) -> Any:
    return NS(id=call_id, type="function", custom_input=None,
              function=NS(name=name, arguments=json.dumps(args)))


def _tool_response(name: str, args: dict[str, Any], call_id: str) -> Any:
    return NS(choices=[NS(message=NS(content=None,
                                    tool_calls=[_call(name, args, call_id)]))],
              usage=None)


def _text_response(text: str) -> Any:
    return NS(choices=[NS(message=NS(content=text, tool_calls=None))], usage=None)


class _Client:
    """A deterministic model that repeats each supplied tool call, then answers.

    A single-element script repeats forever, modelling a model stuck on one
    call; the loop detector must break that turn.  ``max_requests`` turns a
    detector regression into a fast test failure instead of an unbounded run
    that fills the disk with journal records.
    """

    def __init__(self, script: list[Any], max_requests: int = 50) -> None:
        self.script = list(script)
        self.requests = 0
        self.max_requests = max_requests
        self.repeat = len(self.script) == 1
        self.chat = NS(completions=NS(create=self._create))
        self.base_url = NS(host="example.test")

    def _create(self, **kwargs: Any) -> Any:
        self.requests += 1
        if self.requests > self.max_requests:
            raise AssertionError(
                f"model called {self.requests} times without the loop detector "
                "breaking the turn")
        response = self.script[0] if self.repeat else self.script.pop(0) if self.script else _text_response("done")
        return response



def _session(tmp_path: Any, results: list[str], **overrides: Any) -> Any:
    specs, dispatch = agentknit.default_tool_spec()
    calls = iter(results)

    def probe(path: str = "") -> tuple[str, dict[str, Any]]:
        return next(calls), {"result": "ok"}

    dispatch["read_file"]["python_function"] = probe
    session = init_session({"model": "m", "endpoint": "https://example.test/v1",
                            "tool_specs": specs, "tool_dispatch": dispatch},
                           non_interactive=True, strict_cache_proof=False,
                           session_dir=tmp_path / "session", **overrides)
    return session


def _run(tmp_path: Any, responses: list[Any], results: list[str], **overrides: Any) -> Any:
    session = _session(tmp_path, results, **overrides)
    client = _Client(responses)
    events: list[dict[str, Any]] = []
    subscribe(session, "loop_detected", lambda _kind, data: events.append(data))
    result = run_turn(client, "m", session, "diagnose this", sender={"kind": "supervisor", "name": "test"})
    return result, client, session, events


def test_identical_action_and_observation_breaks_at_threshold(tmp_path: Any) -> None:
    result, client, _session, events = _run(
        tmp_path, [_tool_response("read_file", {"path": "a"}, "c")],
        ["no useful output"] * 3)
    assert "Loop detected" in result.final_reply
    assert "3 consecutive" in result.final_reply
    assert client.requests == 3
    assert len(events) == 1
    assert events[0]["repetitions"] == 3
    assert events[0]["period"] == 1
    assert events[0]["tool_names"] == ["read_file"]
    assert _session["messages"][-1]["role"] == "assistant"


def test_boundary_allows_two_identical_steps(tmp_path: Any) -> None:
    result, client, _session, events = _run(
        tmp_path,
        [_tool_response("read_file", {"path": "a"}, "c")] * 2
        + [_text_response("done")], ["same"] * 2)
    assert result.final_reply == "done"
    assert client.requests == 3
    assert not events


def test_changed_observation_is_not_a_loop(tmp_path: Any) -> None:
    result, client, _session, events = _run(
        tmp_path,
        [_tool_response("read_file", {"path": "a"}, "c")] * 5
        + [_text_response("done")],
        ["same"] * 2 + ["changed"] + ["same"] * 2)
    assert result.final_reply == "done"
    assert client.requests == 6
    assert not events


def test_distinct_arguments_do_not_accumulate(tmp_path: Any) -> None:
    result, client, _session, events = _run(
        tmp_path, [_tool_response("read_file", {"path": str(i)}, "c") for i in range(10)],
        ["same"] * 10 + ["done"])
    assert result.final_reply == "done"
    assert client.requests == 11
    assert not events


def test_short_action_observation_cycle_is_detected(tmp_path: Any) -> None:
    # Three repetitions of a two-step cycle need six steps: A B A B A B.
    script = [_tool_response("read_file", {"path": str(i % 2)}, "c")
              for i in range(6)]
    result, client, _session, events = _run(tmp_path, script, ["same"] * 6)
    assert client.requests == 6
    assert events[0]["period"] == 2
    assert events[0]["repetitions"] == 3


def test_opt_out_preserves_unbounded_behavior(tmp_path: Any) -> None:
    result, client, _session, events = _run(
        tmp_path,
        [_tool_response("read_file", {"path": "a"}, "c")] * 10
        + [_text_response("done")],
        ["same"] * 10, loop_detection_enabled=False)
    assert result.final_reply == "done"
    assert client.requests == 11
    assert not events


def test_inline_loop_is_detected(tmp_path: Any) -> None:
    session = _session(tmp_path, ["same"] * 3)
    session["structured"] = False
    call = json.dumps({"name": "read_file", "arguments": {"path": "a"}})
    client = _Client([_text_response(call)])
    result = run_turn(client, "m", session, "diagnose")
    assert "Loop detected" in result.final_reply
    assert client.requests == 3


def test_invalid_configuration_is_rejected(tmp_path: Any) -> None:
    with pytest.raises(ValueError, match="loop_detection_max_repeats"):
        _session(tmp_path, [], loop_detection_max_repeats=1)
    with pytest.raises(ValueError, match="loop_detection_window"):
        _session(tmp_path, [], loop_detection_window=0)
