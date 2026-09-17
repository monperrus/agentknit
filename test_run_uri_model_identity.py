"""A `run://` backend can keep a real model name.

`run://` may be given as the *endpoint*, leaving `model` free to name the model
the subprocess actually talks to. The alternative -- `run://` as the model --
makes the binary path the model identity, so the status line and the log
directory read `_home_user_bin_my-shim.py` and the real model appears nowhere.

Both forms are supported; this pins the distinction so it cannot regress.
"""

from __future__ import annotations

import agentknit
from agentknit._core import DEFAULT_ENDPOINT, load_specification, safe_model_name


def test_run_uri_as_endpoint_keeps_the_model_name(tmp_path):
    shim = tmp_path / "my-shim.py"
    shim.write_text("#!/usr/bin/env python3\n")

    schema = load_specification("composer-2.5", f"run://{shim}")

    assert schema["model"] == "composer-2.5"
    assert schema["endpoint"] == f"run://{shim}"
    # ...and the log directory is named after the model, not the binary.
    assert safe_model_name(schema["model"]) == "composer-2.5"


def test_run_uri_as_model_still_uses_the_path(tmp_path):
    """Backward compatibility: the older form keeps its existing identity."""
    shim = tmp_path / "my-shim.py"
    shim.write_text("#!/usr/bin/env python3\n")

    schema = load_specification(f"run://{shim}", DEFAULT_ENDPOINT)

    assert schema["model"] == str(shim)
    assert schema["endpoint"] == f"run://{shim}"


def test_both_forms_reach_the_same_binary(tmp_path):
    shim = tmp_path / "my-shim.py"
    shim.write_text("#!/usr/bin/env python3\n")

    by_endpoint = agentknit.create_client(load_specification("composer-2.5", f"run://{shim}"))
    by_model = agentknit.create_client(load_specification(f"run://{shim}", DEFAULT_ENDPOINT))

    assert by_endpoint._binary_path == str(shim)
    assert by_model._binary_path == str(shim)
