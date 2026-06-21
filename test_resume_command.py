from __future__ import annotations

from unittest.mock import patch

from agentknit._core import _build_resume_cmd


def test_resume_command_defaults_to_program_model_and_session():
    with patch("sys.argv", ["agent-probe"]):
        assert (
            _build_resume_cmd("test/model", "abc123")
            == "agent-probe test/model --session abc123"
        )


def test_resume_command_uses_wrapper_override_without_model():
    with patch.dict(
        "os.environ",
        {"AGENTKNIT_RESUME_COMMAND": "/home/martin/bin/agent-deepseek-v4-flash-zen.py"},
    ):
        assert (
            _build_resume_cmd("/home/martin/bin/opencode-free-deepseek-v4-flash-completions.py", "cb2c8ce8e1c6")
            == "/home/martin/bin/agent-deepseek-v4-flash-zen.py --session cb2c8ce8e1c6"
        )
