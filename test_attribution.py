"""Tests that git/PR attribution instructions are injected into the system prompt."""

from __future__ import annotations

from agentknit._core import attribution_block, environment_context, init_session

_MINIMAL_SCHEMA = {
    "model": "test-model",
    "endpoint": "https://example.com",
    "tool_specs": [],
    "behaviour": {"call_delivery_mode": "structured_tool_calls"},
}


def test_attribution_block_contents() -> None:
    block = attribution_block("k3")
    assert "## Attribution" in block
    assert "Co-Authored-By: agentknit+k3 <agentknit+k3@monperrus.com>" in block
    assert "🤖 Generated with [agentknit](https://github.com/monperrus/agentknit)" in block


def test_attribution_uses_model_name() -> None:
    block = attribution_block("kimi-k2.6")
    assert "agentknit+kimi-k2.6 <agentknit+kimi-k2.6@monperrus.com>" in block


def test_attribution_in_environment_context() -> None:
    block = environment_context("test-model")
    assert "## Attribution" in block
    assert "Co-Authored-By: agentknit+test-model" in block


def test_attribution_in_system_prompt() -> None:
    session = init_session(dict(_MINIMAL_SCHEMA))
    msg = session["messages"][0]["content"]
    assert "## Attribution" in msg
    assert "Co-Authored-By: agentknit+test-model <agentknit+test-model@monperrus.com>" in msg
    assert "🤖 Generated with [agentknit](https://github.com/monperrus/agentknit)" in msg
