"""Typing-related guarantees: the package ships inline types via py.typed (PEP 561).

Also guards the project's own ``python -m mypy agentknit/`` gate.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import agentknit

REPO_ROOT = Path(__file__).resolve().parent


def test_py_typed_marker_ships() -> None:
    """The PEP 561 marker must exist inside the installed package directory."""
    pkg_dir = Path(agentknit.__file__).resolve().parent
    assert (pkg_dir / "py.typed").is_file(), f"missing py.typed in {pkg_dir}"


def test_public_api_annotations_are_introspectable() -> None:
    """Public callables must carry real annotations (not implicitly untyped)."""
    import inspect

    for name in ("init_session", "run_agent", "run_turn", "safe_model_name"):
        fn = getattr(agentknit, name, None)
        assert callable(fn), f"agentknit.{name} missing or not callable"
        sig = inspect.signature(fn)
        # At minimum the empty-annotation check must not blow up; every public
        # function in this repo is fully annotated under mypy --strict.
        assert sig.return_annotation is not inspect.Signature.empty, (
            f"agentknit.{name} has no return annotation"
        )


def test_mypy_strict_passes_on_package() -> None:
    """``python -m mypy agentknit/`` must pass under the strict pyproject config."""
    try:
        import mypy  # noqa: F401
    except ImportError:
        import pytest

        pytest.skip("mypy not installed")
    proc = subprocess.run(
        [sys.executable, "-m", "mypy", "agentknit/"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"mypy failed:\n{proc.stdout}\n{proc.stderr}"
