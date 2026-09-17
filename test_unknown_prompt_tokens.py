"""An unreported input token count renders as `?`, not as a confident 0.

`prompt 0  |  compact 0%` for a whole session is indistinguishable from a
genuinely empty prompt, and hides that compaction can never trigger.
"""

from __future__ import annotations

from agentknit._core import fmt_usage


class _Usage:
    prompt_tokens = 0
    completion_tokens = 42
    total_tokens = 42
    cached_tokens = 0
    cache_creation_tokens = 0


class _RealUsage(_Usage):
    prompt_tokens = 1000
    total_tokens = 1042
    cached_tokens = 400


def test_known_zero_still_prints_zero():
    line = fmt_usage(_Usage())
    assert "prompt 0" in line
    assert "prompt ?" not in line


def test_unknown_prompt_prints_a_question_mark():
    line = fmt_usage(_Usage(), prompt_tokens_known=False)
    assert "prompt ?" in line
    assert "prompt 0" not in line


def test_unknown_prompt_suppresses_the_compaction_percentage():
    """0% of an unknown prompt is not a measurement."""
    known = fmt_usage(_Usage(), compaction_trigger=10_000)
    unknown = fmt_usage(_Usage(), compaction_trigger=10_000, prompt_tokens_known=False)
    assert "compact 0%" in known
    assert "compact" not in unknown


def test_completion_side_is_unaffected():
    line = fmt_usage(_Usage(), prompt_tokens_known=False)
    assert "completion 42" in line
    assert "total 42" in line


def test_real_accounting_is_unchanged():
    line = fmt_usage(_RealUsage(), compaction_trigger=10_000)
    assert "prompt 1,000 (400 cached, 40%)" in line
    assert "compact 10%" in line
