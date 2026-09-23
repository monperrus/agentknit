"""``step_reducer``: replace each tool step in the history by what a policy keeps.

Also covers the two APIs that make a reducer short to write:
``side_query(max_tokens=, preamble=, count_usage=)`` and ``execute_tool_call``.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace as NS
from typing import Any

import agentknit
from agentknit import StepReduction, init_session, run_turn, side_query


def _tool_resp(path: str) -> Any:
    tc = NS(id="c1", type="function", custom_input=None,
            function=NS(name="read_file", arguments=json.dumps({"path": path})))
    return NS(choices=[NS(message=NS(content=None, tool_calls=[tc]))], usage=None)


def _text_resp(text: str) -> Any:
    return NS(choices=[NS(message=NS(content=text, tool_calls=None))],
              usage=NS(prompt_tokens=50, completion_tokens=7, total_tokens=57,
                       cached_tokens=40, cache_creation_tokens=0))


class _Client:
    """Scripted: the turn's requests pop *script*; side queries answer *digest*."""

    def __init__(self, script: list[Any], digest: str = "read it: SECRET") -> None:
        self.script = list(script)
        self.digest = digest
        self.requests: list[dict[str, Any]] = []
        self.side: list[dict[str, Any]] = []
        self.base_url = NS(host="api.example.test")
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw: Any) -> Any:
        if kw.get("tool_choice") == "none":
            self.side.append(copy.deepcopy(kw))
            return _text_resp(self.digest)
        self.requests.append(copy.deepcopy(kw))
        return self.script.pop(0)


def _session(tmp_path, reducer, structured: bool = True) -> Any:
    specs, dispatch = agentknit.default_tool_spec()
    session = init_session({"model": "m", "endpoint": "https://api.example.test/v1",
                            "tool_specs": specs, "tool_dispatch": dispatch},
                           non_interactive=True, strict_cache_proof=False,
                           session_dir=tmp_path / "sess", step_reducer=reducer)
    session["structured"] = structured
    return session


def _file(tmp_path) -> str:
    p = tmp_path / "big.txt"
    p.write_text("RAW-CONTENT " * 200)
    return str(p)


def _digest_reducer(session, step, *, client, model):
    digest = side_query(client, model, session, "summarize", max_tokens=33,
                        preamble=None, count_usage=True)
    return [{"role": "assistant", "content": f"[digest] {digest}"},
            {"role": "user", "content": "go on"}]


def test_reducer_replaces_step_in_history_and_next_request(tmp_path) -> None:
    client = _Client([_tool_resp(_file(tmp_path)), _text_resp("done")])
    session = _session(tmp_path, _digest_reducer)
    res = run_turn(client, "m", session, "read the file")

    assert res.final_reply == "done"
    history = json.dumps(session["messages"])
    assert "RAW-CONTENT" not in history
    assert all(m["role"] != "tool" for m in session["messages"])
    second = client.requests[1]["messages"]
    assert second[-2]["content"] == "[digest] read it: SECRET"
    assert second[-1]["content"] == "go on"
    assert "RAW-CONTENT" not in json.dumps(second)


def test_reducer_sees_raw_step_and_summary_shares_prefix(tmp_path) -> None:
    seen: list[list[dict[str, Any]]] = []

    def reducer(session, step, *, client, model):
        seen.append(step)
        return _digest_reducer(session, step, client=client, model=model)

    client = _Client([_tool_resp(_file(tmp_path)), _text_resp("done")])
    run_turn(client, "m", _session(tmp_path, reducer), "read the file")

    (step,) = seen
    assert [m["role"] for m in step] == ["assistant", "tool"]
    assert "RAW-CONTENT" in step[1]["content"]
    (summ,) = client.side
    act = client.requests[0]["messages"]
    assert summ["messages"][:len(act)] == act            # cached prefix reused
    assert summ["messages"][-1] == {"role": "user", "content": "summarize"}
    assert summ["max_tokens"] == 33


def test_reducer_returning_none_keeps_step(tmp_path) -> None:
    client = _Client([_tool_resp(_file(tmp_path)), _text_resp("done")])
    session = _session(tmp_path, lambda *a, **k: None)
    run_turn(client, "m", session, "read the file")
    assert any(m["role"] == "tool" for m in session["messages"])


def test_step_reduction_final_reply_ends_turn(tmp_path) -> None:
    events: list[dict[str, Any]] = []

    def reducer(session, step, *, client, model):
        return StepReduction([{"role": "assistant", "content": "[digest] ok"},
                              {"role": "user", "content": "go on"}],
                             final_reply="42")

    client = _Client([_tool_resp(_file(tmp_path))])
    session = _session(tmp_path, reducer)
    agentknit.subscribe(session, "step_reduced", lambda _t, d: events.append(d))
    res = run_turn(client, "m", session, "read the file")

    assert res.final_reply == "42"
    assert len(client.requests) == 1
    assert session["messages"][-1]["role"] == "assistant"
    assert session["messages"][-1]["content"] == "42"
    (ev,) = events
    assert ev["raw_messages"] == 2 and ev["kept_messages"] == 2
    assert ev["final_reply"] == "42"


def test_reduction_is_journaled(tmp_path) -> None:
    client = _Client([_tool_resp(_file(tmp_path)), _text_resp("done")])
    session = _session(tmp_path, _digest_reducer)
    run_turn(client, "m", session, "read the file")
    journal = agentknit._journal_path("m", session["session_id"], tmp_path / "sess")
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    (reset,) = [r for r in records if r.get("type") == "reset_messages"
                and r.get("reason") == "step_reducer"]
    assert "RAW-CONTENT" not in json.dumps(reset["messages"])


def test_inline_mode_step_is_reduced(tmp_path) -> None:
    call = json.dumps({"name": "read_file", "arguments": {"path": _file(tmp_path)}})
    client = _Client([_text_resp(call), _text_resp("done")])
    session = _session(tmp_path, _digest_reducer, structured=False)

    def side(**kw: Any) -> Any:  # inline: no tool_choice, question merged into last user msg
        if kw["messages"][-1]["content"].endswith("summarize"):
            client.side.append(kw)
            return _text_resp("inline digest")
        return client._create(**kw)

    client.chat.completions.create = side
    res = run_turn(client, "m", session, "read the file")
    assert res.final_reply == "done"
    assert "RAW-CONTENT" not in json.dumps(session["messages"])
    assert session["messages"][-3]["content"] == "[digest] inline digest"


def test_no_reducer_is_unchanged_behaviour(tmp_path) -> None:
    client = _Client([_tool_resp(_file(tmp_path)), _text_resp("done")])
    session = _session(tmp_path, None)
    run_turn(client, "m", session, "read the file")
    assert [m["role"] for m in session["messages"][-3:]] == ["assistant", "tool", "assistant"]


def test_side_query_count_usage(tmp_path) -> None:
    session = _session(tmp_path, None)
    usage: list[dict[str, Any]] = []
    agentknit.subscribe(session, "usage", lambda _t, d: usage.append(d))
    client = _Client([])
    side_query(client, "m", session, "q")
    assert session["usage_totals"]["prompt"] == 0 and not usage
    assert client.side[-1]["messages"][-1]["content"].startswith("[Side question")
    side_query(client, "m", session, "q", count_usage=True)
    assert session["usage_totals"]["prompt"] == 50
    assert session["usage_totals"]["cached"] == 40
    assert usage[-1]["purpose"] == "side_query"


def test_execute_tool_call_runs_full_runtime(tmp_path) -> None:
    session = _session(tmp_path, None)
    calls: list[dict[str, Any]] = []
    agentknit.subscribe(session, "tool_call", lambda _t, d: calls.append(d))
    out = agentknit.execute_tool_call(session, "read_file", {"path": _file(tmp_path)})
    assert "RAW-CONTENT" in out
    assert calls[0]["name"] == "read_file"
