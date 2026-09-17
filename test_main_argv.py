"""`main()` and `parse_args()` accept an explicit argv.

Launchers used to compose a command line by mutating `sys.argv` with
positional inserts — order-sensitive, and visible to anything else in the
process that reads it.
"""

from __future__ import annotations

import sys

import pytest

import agentknit
from agentknit._core import parse_args


def test_parse_args_reads_the_given_argv():
    args = parse_args(["my-model", "do", "the", "thing"])
    assert args.model == "my-model"
    assert args.task == ["do", "the", "thing"]


def test_parse_args_flags_from_argv():
    args = parse_args(["m", "--context-window", "4096", "--non-interactive"])
    assert args.context_window == 4096
    assert args.non_interactive is True


def test_parse_args_still_defaults_to_sys_argv(monkeypatch):
    """Backward compatibility: no argument means sys.argv, as before."""
    monkeypatch.setattr(sys, "argv", ["agentknit", "from-sys-argv"])
    assert parse_args().model == "from-sys-argv"


def test_main_consumes_argv_without_touching_sys_argv(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["agentknit", "untouched"])
    before = list(sys.argv)

    # --context-window 0 is rejected early, which proves the argv reached the
    # parser without running a whole session.
    with pytest.raises(SystemExit) as excinfo:
        agentknit.main(["some-model", "--context-window", "0"])

    assert "--context-window" in str(excinfo.value)
    assert sys.argv == before
