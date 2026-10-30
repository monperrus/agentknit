"""Read-before-write and stale-read guards for write_file and str_replace."""

from __future__ import annotations

import os
from pathlib import Path

from agentknit.tool_library import t_read, t_update, t_write


def test_write_existing_unread_file_is_refused(tmp_path: Path) -> None:
    p = tmp_path / "f.txt"
    p.write_text("original")
    r, meta = t_write(path=str(p), content="overwritten")
    assert r.startswith("ERROR: File has not been read yet")
    assert meta["ok"] is False
    assert p.read_text() == "original"


def test_write_new_file_needs_no_prior_read(tmp_path: Path) -> None:
    p = tmp_path / "new.txt"
    r, _ = t_write(path=str(p), content="fresh")
    assert r.startswith("OK:")
    assert p.read_text() == "fresh"


def test_edit_existing_unread_file_is_refused(tmp_path: Path) -> None:
    p = tmp_path / "f.txt"
    p.write_text("alpha\nbeta\n")
    r, meta = t_update(path=str(p), old="alpha", new="gamma")
    assert r.startswith("ERROR: File has not been read yet")
    assert meta["ok"] is False
    assert p.read_text() == "alpha\nbeta\n"


def test_read_then_write_succeeds(tmp_path: Path) -> None:
    p = tmp_path / "f.txt"
    p.write_text("one")
    t_read(path=str(p))
    r, _ = t_write(path=str(p), content="two")
    assert r.startswith("OK:")


def test_read_then_edit_succeeds(tmp_path: Path) -> None:
    p = tmp_path / "f.txt"
    p.write_text("alpha\nbeta\n")
    t_read(path=str(p))
    r, _ = t_update(path=str(p), old="alpha", new="gamma")
    assert r.startswith("OK: replaced 1 of 1")
    assert p.read_text() == "gamma\nbeta\n"


def test_edit_after_external_change_is_refused(tmp_path: Path) -> None:
    p = tmp_path / "f.txt"
    p.write_text("alpha\nbeta\n")
    t_read(path=str(p))
    p.write_text("alpha\nbeta\n# changed by a linter\n")
    r, meta = t_update(path=str(p), old="alpha", new="gamma")
    assert r.startswith("ERROR: File has been modified since read")
    assert meta["ok"] is False
    assert p.read_text() == "alpha\nbeta\n# changed by a linter\n"


def test_write_after_external_change_is_refused(tmp_path: Path) -> None:
    p = tmp_path / "f.txt"
    p.write_text("one")
    t_read(path=str(p))
    p.write_text("two")
    r, meta = t_write(path=str(p), content="three")
    assert r.startswith("ERROR: File has been modified since read")
    assert meta["ok"] is False
    assert p.read_text() == "two"


def test_reread_after_change_recovers(tmp_path: Path) -> None:
    p = tmp_path / "f.txt"
    p.write_text("alpha\n")
    t_read(path=str(p))
    p.write_text("beta\n")
    assert t_update(path=str(p), old="beta", new="gamma")[0].startswith("ERROR:")
    t_read(path=str(p))
    r, _ = t_update(path=str(p), old="beta", new="gamma")
    assert r.startswith("OK: replaced 1 of 1")
    assert p.read_text() == "gamma\n"


def test_partial_read_registers_the_whole_file(tmp_path: Path) -> None:
    p = tmp_path / "f.txt"
    p.write_text("".join(f"line {i}\n" for i in range(200)))
    t_read(path=str(p), offset=150, limit=1)
    r, _ = t_update(path=str(p), old="line 3\n", new="LINE 3\n")
    assert r.startswith("OK: replaced 1 of 1")


def test_same_size_change_is_detected(tmp_path: Path) -> None:
    p = tmp_path / "f.txt"
    p.write_text("aaaa")
    t_read(path=str(p))
    p.write_text("bbbb")          # same byte count, different content
    r, _ = t_update(path=str(p), old="bbbb", new="cccc")
    assert r.startswith("ERROR: File has been modified since read")


def test_failed_read_does_not_register_the_file(tmp_path: Path) -> None:
    p = tmp_path / "f.txt"
    assert t_read(path=str(p))[0].startswith("ERROR:")
    p.write_text("created outside the agent")
    r, _ = t_write(path=str(p), content="agent")
    assert r.startswith("ERROR: File has not been read yet")
    assert p.read_text() == "created outside the agent"


def test_own_edit_refreshes_the_record(tmp_path: Path) -> None:
    p = tmp_path / "f.txt"
    p.write_text("a\nb\n")
    t_read(path=str(p))
    assert t_update(path=str(p), old="a", new="c")[0].startswith("OK:")
    r, _ = t_update(path=str(p), old="b", new="d")
    assert r.startswith("OK: replaced 1 of 1")
    assert p.read_text() == "c\nd\n"


def test_relative_and_absolute_paths_share_the_record(tmp_path: Path) -> None:
    p = tmp_path / "f.txt"
    p.write_text("alpha\n")
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        t_read(path="f.txt")
        r, _ = t_update(path=str(p), old="alpha", new="beta")
        assert r.startswith("OK: replaced 1 of 1")
    finally:
        os.chdir(cwd)
