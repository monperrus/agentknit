"""Mid-turn compaction must not orphan the rest of the turn.

``compact_session`` installs a *new* ``session["messages"]`` list.  ``_run_turn``
used to keep appending to (and sending) the old one, so after a mid-turn
compaction the later tool steps and the final answer never reached the
session, and the following requests went out uncompacted.
"""

from __future__ import annotations

import json
from types import SimpleNamespace as NS
from typing import Any

import agentknit


def _usage() -> Any:
    return NS(prompt_tokens=10, completion_tokens=1, total_tokens=11,
              cached_tokens=5, cache_creation_tokens=0)


def _tool(i: int) -> Any:
    tc = NS(id=f"c{i}", type="function", custom_input=None,
            function=NS(name="exec_shell",
                        arguments=json.dumps({"command": f"echo STEP{i}"})))
    return NS(choices=[NS(message=NS(content=None, tool_calls=[tc]))], usage=_usage())


def test_turn_continues_on_the_compacted_history(tmp_path) -> None:
    script = [_tool(1), _tool(2),
              NS(choices=[NS(message=NS(content="done", tool_calls=None))], usage=_usage())]
    sent: list[list[dict[str, Any]]] = []

    def create(**kw: Any) -> Any:
        if "summar" in json.dumps(kw["messages"][-1]).lower():   # the compaction call
            return NS(choices=[NS(message=NS(content="SUMMARY", tool_calls=None))],
                      usage=_usage())
        sent.append(kw["messages"])
        return script.pop(0)

    client = NS(base_url=NS(host="api.example.test"),
                chat=NS(completions=NS(create=create)))
    fired: list[int] = []

    def policy(session, usage, phase) -> bool:
        if phase == "mid_turn" and not fired:
            fired.append(1)
            return True
        return False

    specs, dispatch = agentknit.default_tool_spec()
    session = agentknit.init_session(
        {"model": "m", "endpoint": "https://api.example.test/v1",
         "tool_specs": specs, "tool_dispatch": dispatch},
        non_interactive=True, strict_cache_proof=False, compaction_policy=policy,
        session_dir=tmp_path / "sess", on_event=lambda *_a: None)
    session["messages"] += [{"role": "user", "content": "old " * 500},
                            {"role": "assistant", "content": "old answer " * 300}]

    res = agentknit.run_turn(client, "m", session, "go")

    assert res.final_reply == "done"
    assert any(m.get("compacted_summary") for m in session["messages"])
    assert any(m.get("compacted_summary") for m in sent[-1])
    assert "STEP2" in json.dumps(session["messages"])
    assert session["messages"][-1]["content"] == "done"
