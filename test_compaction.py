"""Tests for context compaction in agentknit._core."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agentknit._core import (
    compact_session,
    _compact_session,
    _maybe_compact,
    _apply_compaction_policy,
    DEFAULT_COMPACTION_TRIGGER_TOKENS,
    DEFAULT_COMPACTION_TARGET_TOKENS,
    DEFAULT_COMPACTION_KEEP_LAST_TURNS,
)


# ── helpers ───────────────────────────────────────────────────────────────────

class _FakeUsage:
    def __init__(self, prompt_tokens: int = 0) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = 0
        self.total_tokens = prompt_tokens
        self.cached_tokens = 0
        self.cache_creation_tokens = 0
        self.has_cache_proof = False


class _FakeMessage:
    def __init__(self, content: str | None = None) -> None:
        self.content = content
        self.tool_calls = None


class _FakeChoice:
    def __init__(self, content: str | None = None) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str | None = None) -> None:
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()
        self.provider = None


class _FakeClient:
    def __init__(self, content: str | None = None) -> None:
        self._content = content
        self.calls: list[dict] = []

    def chat_completions_create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self._content)


# Monkey-patch the completions create onto the fake client so it looks like
# ``client.chat.completions.create(...)``.
class _FakeCompletions:
    def __init__(self, client: _FakeClient) -> None:
        self._client = client

    def create(self, **kwargs):
        return self._client.chat_completions_create(**kwargs)


class _FakeChat:
    def __init__(self, client: _FakeClient) -> None:
        self.completions = _FakeCompletions(client)


def _make_fake_client(content: str | None = "summary text") -> _FakeClient:
    client = _FakeClient(content)
    client.chat = _FakeChat(client)
    return client


def _make_session(messages: list[dict] | None = None, **overrides) -> dict:
    defaults = {
        "messages": messages or [],
        "compaction_enabled": True,
        "compaction_trigger_tokens": DEFAULT_COMPACTION_TRIGGER_TOKENS,
        "compaction_target_tokens": DEFAULT_COMPACTION_TARGET_TOKENS,
        "compaction_keep_last_turns": DEFAULT_COMPACTION_KEEP_LAST_TURNS,
        "session_id": "test-session",
        "model": "test-model",
        "log_path": MagicMock(),
        "usage_totals": {"prompt": 0, "completion": 0, "total": 0,
                         "cached": 0, "cache_write": 0},
        "on_event": lambda _et, _data: None,
        "_event_handlers": {},
    }
    defaults.update(overrides)
    return defaults


# ── _maybe_compact ────────────────────────────────────────────────────────────

def test_maybe_compact_disabled():
    """When compaction is disabled, _maybe_compact is a no-op."""
    session = _make_session(compaction_enabled=False)
    client = _make_fake_client()
    usage = _FakeUsage(prompt_tokens=DEFAULT_COMPACTION_TRIGGER_TOKENS + 1)
    _maybe_compact(client, "m", session, usage)
    assert len(client.calls) == 0


def test_maybe_compact_under_threshold():
    """When prompt tokens are under the threshold, no compaction happens."""
    session = _make_session()
    client = _make_fake_client()
    usage = _FakeUsage(prompt_tokens=DEFAULT_COMPACTION_TRIGGER_TOKENS - 1)
    _maybe_compact(client, "m", session, usage)
    assert len(client.calls) == 0


def test_maybe_compact_at_threshold():
    """When prompt tokens meet the threshold, compaction is triggered."""
    session = _make_session(messages=[
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ])
    client = _make_fake_client("summary")
    usage = _FakeUsage(prompt_tokens=DEFAULT_COMPACTION_TRIGGER_TOKENS)
    _maybe_compact(client, "m", session, usage)
    assert len(client.calls) == 1


def test_maybe_compact_does_not_retrigger_immediately():
    """After compaction, the same token count must not re-trigger compaction."""
    session = _make_session(messages=[
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ])
    client = _make_fake_client("summary")
    usage = _FakeUsage(prompt_tokens=DEFAULT_COMPACTION_TRIGGER_TOKENS)
    _maybe_compact(client, "m", session, usage)
    assert len(client.calls) == 1
    # Same token count again — should NOT trigger a second compaction.
    _maybe_compact(client, "m", session, usage)
    assert len(client.calls) == 1


def test_maybe_compact_retriggers_after_growth():
    """Compaction fires again only after prompt tokens grow past the last recorded value."""
    session = _make_session(messages=[
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ])
    client = _make_fake_client("summary")
    usage1 = _FakeUsage(prompt_tokens=DEFAULT_COMPACTION_TRIGGER_TOKENS)
    _maybe_compact(client, "m", session, usage1)
    assert len(client.calls) == 1
    # Slightly higher token count — should NOT trigger (needs 25 % hysteresis).
    usage2 = _FakeUsage(prompt_tokens=DEFAULT_COMPACTION_TRIGGER_TOKENS + 1)
    _maybe_compact(client, "m", session, usage2)
    assert len(client.calls) == 1
    # Growth beyond the 25 % hysteresis — should trigger again.
    hysteresis = DEFAULT_COMPACTION_TRIGGER_TOKENS // 4
    usage3 = _FakeUsage(prompt_tokens=DEFAULT_COMPACTION_TRIGGER_TOKENS + hysteresis + 1)
    _maybe_compact(client, "m", session, usage3)
    assert len(client.calls) == 2


# ── _compact_session ──────────────────────────────────────────────────────────

def test_compact_session_keeps_system_and_recent():
    """Compaction keeps system messages and the most recent N turns."""
    session = _make_session(messages=[
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "old assistant"},
        {"role": "user", "content": "recent user"},
        {"role": "assistant", "content": "recent assistant"},
    ])
    client = _make_fake_client("compact summary")
    _compact_session(client, "m", session)

    msgs = session["messages"]
    roles = [m["role"] for m in msgs]
    # system + summary + last 2 turns
    assert roles == ["system", "assistant", "user", "assistant"]
    assert msgs[1].get("compacted_summary") is True
    assert msgs[1]["content"] == "compact summary"


def test_compact_session_keep_zero_turns():
    """With keep_last_turns=0, only system + summary remain."""
    session = _make_session(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ],
        compaction_keep_last_turns=0,
    )
    client = _make_fake_client("summary")
    _compact_session(client, "m", session)

    roles = [m["role"] for m in session["messages"]]
    assert roles == ["system", "assistant"]


def test_compact_session_nothing_to_compact():
    """When there are fewer non-system messages than keep_last_turns, noop."""
    session = _make_session(messages=[
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
    ])
    client = _make_fake_client("summary")
    _compact_session(client, "m", session)
    assert len(client.calls) == 0
    assert len(session["messages"]) == 2


def test_compact_session_empty_summary():
    """An empty summary from the model is ignored."""
    session = _make_session(messages=[
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ])
    client = _make_fake_client("")  # empty summary
    _compact_session(client, "m", session)
    # Messages unchanged because summary was empty.
    assert len(session["messages"]) == 4


def test_compact_session_api_error():
    """An API error during compaction is logged but does not crash."""
    session = _make_session(messages=[
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ])

    class _BrokenClient:
        class _Chat:
            class _Completions:
                def create(self, **kwargs):
                    raise RuntimeError("boom")
            completions = _Completions()
        chat = _Chat()

    _compact_session(_BrokenClient(), "m", session)
    # Messages unchanged after failed compaction.
    assert len(session["messages"]) == 4


def test_compact_session_uses_target_tokens():
    """The compaction call respects compaction_target_tokens."""
    session = _make_session(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
        ],
        compaction_target_tokens=1234,
    )
    client = _make_fake_client("summary")
    _compact_session(client, "m", session)
    assert client.calls[0]["max_tokens"] == 1234


def test_compact_session_summary_metadata():
    """The summary message is tagged with compacted_summary=True."""
    session = _make_session(messages=[
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ])
    client = _make_fake_client("the summary")
    _compact_session(client, "m", session)
    summary_msg = session["messages"][1]
    assert summary_msg.get("compacted_summary") is True
    assert "ts" in summary_msg


def test_compact_session_returns_true_on_success_false_on_noop():
    session = _make_session(messages=[
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ])
    client = _make_fake_client("summary")
    assert compact_session(client, "m", session) is True

    empty = _make_session(messages=[{"role": "system", "content": "sys"}])
    assert compact_session(client, "m", empty) is False


def test_compact_session_min_chars_guard():
    """Compaction is skipped while the compactable text is below the minimum."""
    session = _make_session(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "short"},
            {"role": "assistant", "content": "reply"},
        ],
        compaction_keep_last_turns=0,
        compaction_min_chars=1000,
    )
    client = _make_fake_client("summary")
    assert compact_session(client, "m", session) is False
    assert len(client.calls) == 0

    # Enough text → compaction proceeds.
    session["messages"][1]["content"] = "x" * 2000
    assert compact_session(client, "m", session) is True


def test_compact_session_never_splits_tool_call_pairs():
    """The keep boundary snaps back to a user message, keeping tool pairs intact."""
    session = _make_session(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "tool_calls": [{"id": "t1", "type": "function",
                                                  "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "t1", "content": "result"},
            {"role": "assistant", "content": "final"},
        ],
        compaction_keep_last_turns=2,  # naive split would land on the tool message
    )
    client = _make_fake_client("summary")
    assert compact_session(client, "m", session) is True

    roles = [m["role"] for m in session["messages"]]
    # Snapped back to u2: system + summary + [u2, tool_calls, tool, final].
    assert roles == ["system", "assistant", "user", "assistant", "tool", "assistant"]
    assert session["messages"][2]["content"] == "u2"


def test_compact_session_no_clean_boundary_is_noop():
    """If no user boundary exists inside the compactable range, skip."""
    session = _make_session(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "assistant", "content": "a2"},
        ],
        compaction_keep_last_turns=2,
    )
    client = _make_fake_client("summary")
    # Snapping back from a2 reaches u1, the first non-system message: the
    # prefix would be empty, so nothing is compacted.
    assert compact_session(client, "m", session) is False
    assert len(client.calls) == 0


# ── _apply_compaction_policy ──────────────────────────────────────────────────

_POLICY_MESSAGES = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "u1"},
    {"role": "assistant", "content": "a1"},
    {"role": "user", "content": "u2"},
    {"role": "assistant", "content": "a2"},
]


def test_policy_every_turn_compacts_at_turn_end_unconditionally():
    session = _make_session(messages=list(_POLICY_MESSAGES),
                            compaction_policy="every_turn",
                            compaction_keep_last_turns=0)
    client = _make_fake_client("summary")
    usage = _FakeUsage(prompt_tokens=10)  # far below any threshold
    _apply_compaction_policy(client, "m", session, usage, phase="turn_end")
    assert len(client.calls) == 1
    assert session["messages"][1].get("compacted_summary") is True


def test_policy_every_turn_keeps_threshold_backstop_mid_turn():
    session = _make_session(messages=list(_POLICY_MESSAGES),
                            compaction_policy="every_turn")
    client = _make_fake_client("summary")
    # Below threshold mid-turn: nothing happens.
    _apply_compaction_policy(client, "m", session, _FakeUsage(prompt_tokens=10),
                             phase="mid_turn")
    assert len(client.calls) == 0
    # Above threshold mid-turn: the backstop fires.
    _apply_compaction_policy(client, "m", session,
                             _FakeUsage(prompt_tokens=DEFAULT_COMPACTION_TRIGGER_TOKENS),
                             phase="mid_turn")
    assert len(client.calls) == 1


def test_policy_threshold_is_default_and_ignores_turn_end():
    session = _make_session(messages=list(_POLICY_MESSAGES))
    client = _make_fake_client("summary")
    _apply_compaction_policy(client, "m", session, _FakeUsage(prompt_tokens=10),
                             phase="turn_end")
    assert len(client.calls) == 0


def test_policy_never_disables_all_compaction():
    session = _make_session(messages=list(_POLICY_MESSAGES),
                            compaction_policy="never")
    client = _make_fake_client("summary")
    big = _FakeUsage(prompt_tokens=DEFAULT_COMPACTION_TRIGGER_TOKENS * 2)
    _apply_compaction_policy(client, "m", session, big, phase="mid_turn")
    _apply_compaction_policy(client, "m", session, big, phase="turn_end")
    assert len(client.calls) == 0


def test_policy_callable_receives_phase_and_decides():
    seen: list[str] = []

    def policy(session, usage, phase):
        seen.append(phase)
        return phase == "turn_end"

    session = _make_session(messages=list(_POLICY_MESSAGES),
                            compaction_policy=policy,
                            compaction_keep_last_turns=0)
    client = _make_fake_client("summary")
    _apply_compaction_policy(client, "m", session, None, phase="mid_turn")
    assert len(client.calls) == 0
    _apply_compaction_policy(client, "m", session, None, phase="turn_end")
    assert len(client.calls) == 1
    assert seen == ["mid_turn", "turn_end"]


def test_compact_session_turn_count():
    """Compaction correctly counts compacted turns."""
    session = _make_session(messages=[
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "a3"},
    ])
    client = _make_fake_client("summary")
    _compact_session(client, "m", session)
    # system + summary + 2 kept turns = 4 messages
    assert len(session["messages"]) == 4
    # The compacted portion had 4 non-system messages (u1,a1,u2,a2).
    # Summary replaces them, so we have 1 system + 1 summary + 2 kept = 4.
    assert session["messages"][1].get("compacted_summary") is True
