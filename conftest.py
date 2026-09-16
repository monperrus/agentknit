"""Test-wide isolation from the developer's real home directory.

agentknit reads several user-level config layers at runtime — most notably
``~/.agentknit/hooks.json`` and ``~/.claude/CLAUDE.md``.  Without isolation a
test run picks those up, so the suite passes in CI (a bare home) and fails on
a developer machine that happens to have them.  The autouse fixture below
points ``$HOME`` at an empty per-run directory so every test sees the same,
config-free home.
"""

from __future__ import annotations

import site
import sys
from pathlib import Path

import pytest

# Where pip --user installed packages live for the interpreter running the
# suite.  Moving $HOME would hide them from any interpreter a test spawns
# (test_typing.py runs `python -m mypy`), so pin it back explicitly.
_USER_BASE = site.ENABLE_USER_SITE and getattr(site, "USER_BASE", None)


@pytest.fixture(scope="session")
def _isolated_home(tmp_path_factory) -> Path:
    home = tmp_path_factory.mktemp("home")
    if sys.platform != "win32":
        # A fake home with no .config/.cache still needs to be writable for
        # tools that insist on creating them.
        (home / ".config").mkdir(exist_ok=True)
        (home / ".cache").mkdir(exist_ok=True)
    return home


@pytest.fixture(autouse=True)
def isolate_home(monkeypatch, _isolated_home) -> None:
    """Point ``$HOME`` at an empty directory for every test.

    ``Path.home()`` and ``os.path.expanduser`` both honour ``$HOME`` on POSIX,
    so this neutralises every user-level config layer at once.
    """
    monkeypatch.setenv("HOME", str(_isolated_home))
    monkeypatch.setenv("USERPROFILE", str(_isolated_home))  # Windows
    if _USER_BASE:
        monkeypatch.setenv("PYTHONUSERBASE", str(_USER_BASE))
