"""A `run://` backend can be parameterised by the spec.

Before this, the subprocess was invoked as a bare path with the ambient
environment, so a launcher wanting to choose e.g. a model had to set a
process-wide variable — which also made two models in one process impossible.
"""

from __future__ import annotations

import json
import os
import stat
import sys

import agentknit
from agentknit.openai_compat import DEFAULT_SUBPROCESS_TIMEOUT, SubprocessOpenAI


def _echo_shim(tmp_path, name="shim.py"):
    """A shim that answers with the argv and env it was given."""
    path = tmp_path / name
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "json.load(sys.stdin)\n"
        "report = {'argv': sys.argv[1:], 'model_env': os.environ.get('SHIM_MODEL')}\n"
        "print(json.dumps({'choices': [{'index': 0, 'message': "
        "{'role': 'assistant', 'content': json.dumps(report)}, "
        "'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 1, "
        "'completion_tokens': 1, 'total_tokens': 2}}))\n"
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def _ask(client):
    response = client.chat.completions.create(
        model="x", messages=[{"role": "user", "content": "hi"}]
    )
    return json.loads(response.choices[0].message.content)


def test_command_vector_reaches_the_child(tmp_path):
    shim = _echo_shim(tmp_path)
    client = SubprocessOpenAI([sys.executable, str(shim), "--model", "grok-4.6"])
    assert _ask(client)["argv"] == ["--model", "grok-4.6"]


def test_env_is_added_for_the_child_only(tmp_path):
    shim = _echo_shim(tmp_path)
    client = SubprocessOpenAI([sys.executable, str(shim)], env={"SHIM_MODEL": "composer-2.5"})
    assert _ask(client)["model_env"] == "composer-2.5"
    assert "SHIM_MODEL" not in os.environ


def test_bare_path_still_works(tmp_path):
    """Backward compatibility: a plain string is still a valid backend."""
    shim = _echo_shim(tmp_path)
    shim.write_text("#!" + sys.executable + "\n" + shim.read_text().split("\n", 1)[1])
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    client = SubprocessOpenAI(str(shim))
    assert client._binary_path == str(shim)
    assert _ask(client)["argv"] == []


def test_create_client_passes_command_and_env(tmp_path):
    shim = _echo_shim(tmp_path)
    client = agentknit.create_client({
        "model": str(shim),
        "endpoint": f"run://{shim}",
        "command": [sys.executable, str(shim), "--model", "claude-opus-5"],
        "command_env": {"SHIM_MODEL": "claude-opus-5"},
    })
    report = _ask(client)
    assert report["argv"] == ["--model", "claude-opus-5"]
    assert report["model_env"] == "claude-opus-5"


def test_command_timeout_defaults_and_overrides(tmp_path):
    shim = _echo_shim(tmp_path)
    default = agentknit.create_client({
        "model": str(shim), "endpoint": f"run://{shim}",
    })
    assert (default._timeout or DEFAULT_SUBPROCESS_TIMEOUT) == DEFAULT_SUBPROCESS_TIMEOUT
    slow = agentknit.create_client({
        "model": str(shim), "endpoint": f"run://{shim}", "command_timeout": 900,
    })
    assert slow._timeout == 900
