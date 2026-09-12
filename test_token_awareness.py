"""Tests for model-facing token awareness in agentknit._core.

Token awareness injects the *true* token count (as reported by the API's
usage block) into the model's own context: a budget declaration in the
system prompt, a countdown suffixed onto tool results, and an
edge-triggered checkpoint reminder near the compaction trigger.  The
number is never padded or fabricated — a fake counter is a bug.
"""

from __future__ import annotations

import contextlib
import io

import agentknit._core as _core
from agentknit import load_specification
from agentknit.openai_compat import (
    SubprocessOpenAI, _Message, _Choice, _Usage, _Response, _SubprocessChat,
)
from agentknit.openai_compat import _SubprocessCompletions

MODEL = "test/model"
ENDPOINT = "https://api.test/v1"


class _FakeToolFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: str) -> None:
        self.id = call_id
        self.type = "function"
        self.custom_input = None
        self.function = _FakeToolFunction(name, arguments)


class ScriptedOpenAI(SubprocessOpenAI):
    """Stub client returning a scripted sequence of (message, prompt_tokens)."""

    def __init__(self, script: list[tuple[_Message, int]]) -> None:
        from agentknit.openai_compat import _BaseURL
        self._binary_path = "stub"
        self.base_url = _BaseURL("")
        self.requests: list[dict] = []
        self._script = list(script)
        self.chat = _SubprocessChat(self)

    def _complete(self, model: str, messages: list[dict],
                  **kwargs: dict) -> _Response:
        self.requests.append({"model": model, "messages": list(messages),
                              **kwargs})
        msg, prompt_tok = self._script.pop(0)
        usage = _Usage(prompt_tokens=prompt_tok, completion_tokens=10,
                       total_tokens=prompt_tok + 10, has_cache_proof=True)
        return _Response(choices=[_Choice(msg)], usage=usage)


def _patch_create(monkeypatch) -> None:
    def _create(self, *, model, messages, **kwargs):
        return self._client._complete(model, messages, **kwargs)

    monkeypatch.setattr(_SubprocessCompletions, "create", _create)


def _quiet(monkeypatch) -> None:
    monkeypatch.setattr(_core, "_default_event_handler", lambda *_a: None)


def _run_scripted(monkeypatch, script, schema_extra: dict | None = None,
                  **init_kwargs):
    _patch_create(monkeypatch)
    _quiet(monkeypatch)
    schema = load_specification(MODEL, ENDPOINT)
    schema.update(schema_extra or {})
    client = ScriptedOpenAI(script)
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        session = _core.init_session(schema, non_interactive=True, durable=False,
                                     **init_kwargs)
        _core.run_turn(client, MODEL, session, "do the thing")
    return session, client


def _all_content(messages: list[dict]) -> str:
    return "\n".join(str(m.get("content") or "") for m in messages)


def test_system_prompt_declares_budget_by_default(monkeypatch) -> None:
    """Default (enabled): system prompt carries the budget tag and the
    persistence sentence; countdown falls back to the compaction trigger."""
    _quiet(monkeypatch)
    schema = load_specification(MODEL, ENDPOINT)
    session = _core.init_session(schema, durable=False)
    sys_msg = session["messages"][0]["content"]
    assert "<budget:token_budget>100000</budget:token_budget>" in sys_msg
    assert "normal operation, not a deadline" in sys_msg
    assert session["token_awareness_enabled"] is True
    assert session["token_awareness_budget_tokens"] == 100_000
    assert session["token_awareness_reminder_tokens"] == 6144


def test_disabled_removes_everything(monkeypatch) -> None:
    """Disabled: no budget tag in the system prompt and nothing injected
    into tool results."""
    tool_msg = _Message(role="assistant", content=None,
                        tool_calls=[_FakeToolCall("c1", "read_file",
                                                  '{"path": "x"}')])
    done_msg = _Message(role="assistant", content="done", tool_calls=None)
    session, _ = _run_scripted(
        monkeypatch, [(tool_msg, 5000), (done_msg, 6000)],
        token_awareness_enabled=False)
    assert "<budget:token_budget>" not in session["messages"][0]["content"]
    assert "<system_warning>" not in _all_content(session["messages"])
    assert "context_window_reminder" not in _all_content(session["messages"])


def test_countdown_injected_after_tool_call_exact_numbers(monkeypatch) -> None:
    """The countdown carries the exact server-reported prompt size."""
    tool_msg = _Message(role="assistant", content=None,
                        tool_calls=[_FakeToolCall("c1", "read_file",
                                                  '{"path": "x"}')])
    done_msg = _Message(role="assistant", content="done", tool_calls=None)
    session, _ = _run_scripted(monkeypatch, [(tool_msg, 35000), (done_msg, 36000)])
    tool_results = [m for m in session["messages"] if m.get("role") == "tool"]
    assert len(tool_results) == 1
    assert ("<system_warning>Token usage: 35000/100000; 65000 remaining"
            "</system_warning>") in tool_results[0]["content"]
    # The final answer path is untouched.
    assert session["messages"][-1]["content"] == "done"


def test_no_injection_without_tool_calls(monkeypatch) -> None:
    """A turn that ends with plain text never gets the countdown."""
    done_msg = _Message(role="assistant", content="done", tool_calls=None)
    session, _ = _run_scripted(monkeypatch, [(done_msg, 35000)])
    assert "<system_warning>" not in _all_content(session["messages"])
    assert session["messages"][-1]["content"] == "done"


def test_reminder_fires_once_on_threshold_crossing(monkeypatch) -> None:
    """Edge-triggered: the checkpoint reminder appears only on the call
    where remaining crosses below the threshold, not on later calls."""
    t1 = _Message(role="assistant", content=None,
                  tool_calls=[_FakeToolCall("c1", "read_file", '{"path": "a"}')])
    t2 = _Message(role="assistant", content=None,
                  tool_calls=[_FakeToolCall("c2", "read_file", '{"path": "b"}')])
    done = _Message(role="assistant", content="done", tool_calls=None)
    session, _ = _run_scripted(
        monkeypatch, [(t1, 95000), (t2, 96000), (done, 97000)])
    tool_results = [m for m in session["messages"] if m.get("role") == "tool"]
    assert len(tool_results) == 2
    assert "context_window_reminder" in tool_results[0]["content"]
    assert "5000 tokens remain before compaction" in tool_results[0]["content"]
    # Second tool result: still below threshold → countdown only, no repeat.
    assert "<system_warning>" in tool_results[1]["content"]
    assert "context_window_reminder" not in tool_results[1]["content"]


def test_reminder_rearms_after_compaction_reopens_headroom(monkeypatch) -> None:
    """Anti padded-countdown: the counter is derived from real usage, so
    when prompt size drops (post-compaction) remaining goes back up and a
    later crossing fires the reminder again."""
    budget, reminder = 10000, 2000
    msgs = [
        (_Message(role="assistant", content=None,
                  tool_calls=[_FakeToolCall("c1", "read_file", '{"path": "a"}')]), 9000),   # below → fires
        (_Message(role="assistant", content=None,
                  tool_calls=[_FakeToolCall("c2", "read_file", '{"path": "b"}')]), 3000),   # compacted: above
        (_Message(role="assistant", content=None,
                  tool_calls=[_FakeToolCall("c3", "read_file", '{"path": "c"}')]), 8500),   # below → fires again
        (_Message(role="assistant", content="done", tool_calls=None), 8600),
    ]
    session, _ = _run_scripted(
        monkeypatch, msgs,
        token_awareness_budget_tokens=budget,
        token_awareness_reminder_tokens=reminder)
    tool_results = [m for m in session["messages"] if m.get("role") == "tool"]
    assert len(tool_results) == 3
    assert "context_window_reminder" in tool_results[0]["content"]
    assert "Token usage: 3000/10000; 7000 remaining" in tool_results[1]["content"]
    assert "context_window_reminder" not in tool_results[1]["content"]
    assert "context_window_reminder" in tool_results[2]["content"]


def test_update_every_skips_calls(monkeypatch) -> None:
    """update_every=3 injects on LLM calls 3, 6, … only."""
    msgs = [
        (_Message(role="assistant", content=None,
                  tool_calls=[_FakeToolCall(f"c{i}", "read_file", '{"path": "x"}')]), 1000 * i)
        for i in range(1, 4)
    ]
    msgs.append((_Message(role="assistant", content="done", tool_calls=None), 4000))
    session, _ = _run_scripted(monkeypatch, msgs, token_awareness_update_every=3)
    tool_results = [m for m in session["messages"] if m.get("role") == "tool"]
    assert len(tool_results) == 3
    assert "<system_warning>" not in tool_results[0]["content"]
    assert "<system_warning>" not in tool_results[1]["content"]
    assert "Token usage: 3000/100000; 97000 remaining" in tool_results[2]["content"]


def test_budget_fallback_chain(monkeypatch) -> None:
    """explicit kwarg → schema context_window → compaction trigger."""
    _quiet(monkeypatch)
    schema = load_specification(MODEL, ENDPOINT)
    s1 = _core.init_session(schema, durable=False,
                            token_awareness_budget_tokens=123456)
    assert s1["token_awareness_budget_tokens"] == 123456
    assert "<budget:token_budget>123456</budget:token_budget>" in s1["messages"][0]["content"]

    schema2 = load_specification(MODEL, ENDPOINT)
    schema2["context_window"] = 200000
    s2 = _core.init_session(schema2, durable=False)
    assert s2["token_awareness_budget_tokens"] == 200000

    schema3 = load_specification(MODEL, ENDPOINT)
    schema3["compaction_trigger_tokens"] = 50000
    s3 = _core.init_session(schema3, durable=False)
    assert s3["token_awareness_budget_tokens"] == 50000


def test_token_budget_event_and_usage_log(monkeypatch, tmp_path) -> None:
    """The token_budget event fires with the countdown data and the usage
    log record gains token_budget_remaining."""
    events: list[tuple[str, dict]] = []
    tool_msg = _Message(role="assistant", content=None,
                        tool_calls=[_FakeToolCall("c1", "read_file", '{"path": "x"}')])
    done_msg = _Message(role="assistant", content="done", tool_calls=None)
    _patch_create(monkeypatch)
    schema = load_specification(MODEL, ENDPOINT)
    client = ScriptedOpenAI([(tool_msg, 35000), (done_msg, 36000)])
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        session = _core.init_session(schema, non_interactive=True, durable=False,
                                     on_event=lambda et, d: events.append((et, d)))
        _core.run_turn(client, MODEL, session, "do the thing")
    budget_events = [d for et, d in events if et == "token_budget"]
    assert budget_events and budget_events[0]["used"] == 35000
    assert budget_events[0]["budget"] == 100000
    assert budget_events[0]["remaining"] == 65000
    assert budget_events[0]["below_reminder_threshold"] is False
    log_lines = session["log_path"].read_text().splitlines()
    import json
    usage_records = [json.loads(line) for line in log_lines
                     if '"type": "usage"' in line]
    assert usage_records[0]["token_budget_remaining"] == 65000


def test_old_session_dict_backfilled_on_restore(monkeypatch) -> None:
    """Sessions snapshotted before token awareness get defaults on restore
    (backward compatibility); the countdown continues from real usage."""
    _quiet(monkeypatch)
    schema = load_specification(MODEL, ENDPOINT)
    session = _core.init_session(schema, durable=False)
    # Simulate an old snapshot: strip the token-awareness keys.
    for k in [k for k in session if k.startswith("token_awareness")]:
        del session[k]
    restored = _core.init_session(schema, session=session, durable=False)
    assert restored["token_awareness_enabled"] is True
    assert restored["token_awareness_budget_tokens"] == 100000
    assert restored["token_awareness_reminder_tokens"] == 6144
    assert restored["token_awareness_update_every"] == 1


def test_snapshot_metadata_records_knobs(monkeypatch, tmp_path) -> None:
    """Snapshot metadata keeps the four knobs so a resumed session keeps
    identical countdown semantics."""
    _quiet(monkeypatch)
    schema = load_specification(MODEL, ENDPOINT)
    session = _core.init_session(schema, durable=False, session_dir=tmp_path,
                                 token_awareness_budget_tokens=250000)
    session["messages"].append({"role": "user", "content": "hi"})
    _core._save_messages_snapshot(session)
    import json
    payload = json.loads((tmp_path / "messages.json").read_text())
    ta = payload["metadata"]["token_awareness"]
    assert ta["enabled"] is True
    assert ta["budget_tokens"] == 250000
    assert ta["reminder_tokens"] == 6144
    assert ta["update_every"] == 1


def test_resume_strips_stale_usage_warnings_keeps_latest() -> None:
    """On resume, mid-history Token usage warnings are stale readings and
    must be stripped; only the most recent one survives."""
    from agentknit._core import _normalise_for_resume

    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a1",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "t", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1",
         "content": "res1\n\n<system_warning>Token usage: 100/1000; "
                    "900 remaining</system_warning>"},
        {"role": "assistant", "content": "a2",
         "tool_calls": [{"id": "c2", "type": "function",
                         "function": {"name": "t", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c2",
         "content": "res2\n\n<system_warning>Token usage: 200/1000; "
                    "800 remaining</system_warning>"},
    ]
    out = _normalise_for_resume(msgs)
    contents = [m.get("content") or "" for m in out]
    # The older reading is stripped entirely.
    assert contents[3] == "res1"
    # The latest reading survives verbatim.
    assert "Token usage: 200/1000" in contents[5]
    # No warning left anywhere else.
    assert sum("Token usage" in c for c in contents) == 1


def test_resume_strips_warning_with_checkpoint_reminder() -> None:
    """A stale warning carrying the near-full checkpoint reminder is
    removed as one block."""
    from agentknit._core import _normalise_for_resume

    stale = ("res\n\n<system_warning>Token usage: 990/1000; 10 remaining"
             "</system_warning>\n<context_window_reminder>\nnearly full\n"
             "</context_window_reminder>")
    out = _normalise_for_resume([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a1",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "t", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": stale},
        {"role": "assistant", "content": "a2",
         "tool_calls": [{"id": "c2", "type": "function",
                         "function": {"name": "t", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c2",
         "content": "later\n\n<system_warning>Token usage: 50/1000; "
                    "950 remaining</system_warning>"},
    ])
    assert out[3]["content"] == "res"
    assert "context_window_reminder" not in out[3]["content"]
    assert "Token usage: 50/1000" in out[5]["content"]
