"""Tests for endpoint/model metadata in saved trajectories."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from agentknit._core import (
    _save_messages_snapshot,
    _load_messages_snapshot,
    _find_snapshot_in_other_models,
    safe_model_name,
)


def _make_session(messages, model="test-model", endpoint="https://api.example.com/v1", session_id="sid-123"):
    return {
        "messages": messages,
        "model": model,
        "endpoint": endpoint,
        "session_id": session_id,
    }


def _patched_commit(monkeypatch, value="deadbeef"):
    from agentknit import _core as core_mod
    monkeypatch.setattr(core_mod, "_agentknit_commit", lambda: value)
    return core_mod


def test_save_messages_snapshot_includes_metadata(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        session = _make_session(
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello"},
            ],
            model="my/model",
            endpoint="https://api.example.com/v1",
            session_id="sess-abc",
        )
        # patch LOG_BASE for the test
        from agentknit import _core as core_mod
        orig_log_base = core_mod.LOG_BASE
        core_mod.LOG_BASE = base
        core_mod = _patched_commit(monkeypatch)
        try:
            _save_messages_snapshot(session)
            path = base / safe_model_name("my/model") / "sess-abc_messages.json"
            assert path.exists()
            data = json.loads(path.read_text())
            assert "metadata" in data
            assert data["metadata"]["endpoint"] == "https://api.example.com/v1"
            assert data["metadata"]["model"] == "my/model"
            assert data["metadata"]["session_id"] == "sess-abc"
            assert "messages" in data
            assert len(data["messages"]) == 2

            # Tool provenance: absent "tools" in the session → default set,
            # named explicitly, plus the agentknit commit id.
            assert data["metadata"]["default_tools"] is True
            assert data["metadata"]["tools"] == [
                "read_file", "write_file", "str_replace", "execute_shell_command",
            ]
            assert data["metadata"]["agentknit_commit"] == "deadbeef"
        finally:
            core_mod.LOG_BASE = orig_log_base


def test_load_messages_snapshot_reads_new_format():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        from agentknit import _core as core_mod
        orig_log_base = core_mod.LOG_BASE
        core_mod.LOG_BASE = base
        try:
            model_dir = base / safe_model_name("my/model")
            model_dir.mkdir(parents=True)
            payload = {
                "metadata": {
                    "endpoint": "https://api.example.com/v1",
                    "model": "my/model",
                    "session_id": "sess-abc",
                },
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "hello"},
                ],
            }
            path = model_dir / "sess-abc_messages.json"
            path.write_text(json.dumps(payload))
            loaded = _load_messages_snapshot("my/model", "sess-abc")
            assert loaded == payload["messages"]
        finally:
            core_mod.LOG_BASE = orig_log_base


def test_load_messages_snapshot_reads_legacy_format():
    """Backward compatibility: plain array without metadata wrapper."""
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        from agentknit import _core as core_mod
        orig_log_base = core_mod.LOG_BASE
        core_mod.LOG_BASE = base
        try:
            model_dir = base / safe_model_name("my/model")
            model_dir.mkdir(parents=True)
            legacy = [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello"},
            ]
            path = model_dir / "sess-legacy_messages.json"
            path.write_text(json.dumps(legacy))
            loaded = _load_messages_snapshot("my/model", "sess-legacy")
            assert loaded == legacy
        finally:
            core_mod.LOG_BASE = orig_log_base


def test_find_snapshot_in_other_models_reads_new_format():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        from agentknit import _core as core_mod
        orig_log_base = core_mod.LOG_BASE
        core_mod.LOG_BASE = base
        try:
            other_dir = base / safe_model_name("other-model")
            other_dir.mkdir(parents=True)
            payload = {
                "metadata": {
                    "endpoint": "https://api.other.com/v1",
                    "model": "other-model",
                    "session_id": "sess-shared",
                },
                "messages": [
                    {"role": "user", "content": "hi"},
                ],
            }
            path = other_dir / "sess-shared_messages.json"
            path.write_text(json.dumps(payload))
            loaded, source = _find_snapshot_in_other_models("my/model", "sess-shared")
            assert loaded == payload["messages"]
            assert source == safe_model_name("other-model")
        finally:
            core_mod.LOG_BASE = orig_log_base


def test_find_snapshot_in_other_models_reads_legacy_format():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        from agentknit import _core as core_mod
        orig_log_base = core_mod.LOG_BASE
        core_mod.LOG_BASE = base
        try:
            other_dir = base / safe_model_name("other-model")
            other_dir.mkdir(parents=True)
            legacy = [{"role": "user", "content": "legacy"}]
            path = other_dir / "sess-legacy_messages.json"
            path.write_text(json.dumps(legacy))
            loaded, source = _find_snapshot_in_other_models("my/model", "sess-legacy")
            assert loaded == legacy
            assert source == safe_model_name("other-model")
        finally:
            core_mod.LOG_BASE = orig_log_base


def test_save_snapshot_records_custom_tools(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        core_mod = _patched_commit(monkeypatch, "cafebabe")
        orig_log_base = core_mod.LOG_BASE
        core_mod.LOG_BASE = base
        try:
            session = _make_session(
                messages=[{"role": "user", "content": "hi"}],
                model="m",
                session_id="sess-custom",
            )
            session["tools"] = [
                {"type": "function", "function": {
                    "name": "custom_tool", "description": "d",
                    "parameters": {"type": "object", "properties": {}}}},
            ]
            _save_messages_snapshot(session)
            data = json.loads(
                (base / safe_model_name("m") / "sess-custom_messages.json").read_text())
            assert data["metadata"]["default_tools"] is False
            assert data["metadata"]["tools"] == ["custom_tool"]
            assert data["metadata"]["agentknit_commit"] == "cafebabe"
        finally:
            core_mod.LOG_BASE = orig_log_base


def test_save_skips_when_only_system_messages():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        from agentknit import _core as core_mod
        orig_log_base = core_mod.LOG_BASE
        core_mod.LOG_BASE = base
        try:
            session = _make_session(
                messages=[{"role": "system", "content": "sys"}],
                model="m",
                session_id="s",
            )
            _save_messages_snapshot(session)
            path = base / safe_model_name("m") / "s_messages.json"
            assert not path.exists()
        finally:
            core_mod.LOG_BASE = orig_log_base
