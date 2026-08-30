"""Resuming a session must reuse the endpoint (and key source) it ran on.

The snapshot metadata records the endpoint a session actually used; a
resume that lands on a different endpoint (CLI default, wrapper override)
replays the transcript against the wrong provider — e.g. a z.ai session
resumed via the OpenRouter default fails with a bogus 402 Payment Required.
"""

from __future__ import annotations

import json
from pathlib import Path

import agentknit._core as core
from agentknit import (
    DEFAULT_ENDPOINT,
    load_specification,
    run_task,
    safe_model_name,
)
from agentknit._core import (
    _bind_schema_to_resumed_session,
    _load_snapshot_metadata,
    _port_snapshot_to_model,
)

Z_AI = "https://api.z.ai/api/coding/paas/v4"


def _write_snapshot(base: Path, model: str, session_id: str, *,
                    endpoint: str, auth: dict | None = None) -> None:
    d = base / safe_model_name(model)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{session_id}_messages.json").write_text(json.dumps({
        "metadata": {
            "endpoint": endpoint,
            "model": model,
            "session_id": session_id,
            "auth": auth or {},
        },
        "messages": [{"role": "user", "content": "hi"}],
    }))


def _schema() -> dict:
    return load_specification("test/model", DEFAULT_ENDPOINT)


# ── _load_snapshot_metadata ──────────────────────────────────────────────────

def test_load_snapshot_metadata_prefers_own_model_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "LOG_BASE", tmp_path)
    _write_snapshot(tmp_path, "test/model", "s1", endpoint="https://a.test/v1")
    _write_snapshot(tmp_path, "other/model", "s1", endpoint="https://b.test/v1")
    assert _load_snapshot_metadata("test/model", "s1")["endpoint"] == "https://a.test/v1"


def test_load_snapshot_metadata_ignores_other_models(monkeypatch, tmp_path):
    """A snapshot under another model is a provider switch, not this model's
    history — its endpoint must not leak into the binding."""
    monkeypatch.setattr(core, "LOG_BASE", tmp_path)
    _write_snapshot(tmp_path, "other/model", "s1", endpoint=Z_AI)
    assert _load_snapshot_metadata("test/model", "s1") is None


def test_load_snapshot_metadata_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "LOG_BASE", tmp_path)
    assert _load_snapshot_metadata("test/model", "nope") is None


def test_load_snapshot_metadata_legacy_array(monkeypatch, tmp_path):
    """Snapshots from before metadata existed carry no endpoint → no rebinding."""
    monkeypatch.setattr(core, "LOG_BASE", tmp_path)
    d = tmp_path / safe_model_name("test/model")
    d.mkdir(parents=True)
    (d / "s1_messages.json").write_text(json.dumps([{"role": "user", "content": "x"}]))
    assert _load_snapshot_metadata("test/model", "s1") is None


# ── _load_snapshot_metadata / _session_origin_endpoint ──────────────────────

def test_origin_endpoint_in_logs_wins_over_overwritten_snapshot(monkeypatch, tmp_path):
    """A bad resume rewrites the snapshot with its own endpoint; the logs don't lie."""
    monkeypatch.setattr(core, "LOG_BASE", tmp_path)
    model_dir = tmp_path / "glm-5.3"
    # Original session ran on z.ai…
    day = model_dir / "2026-08-30"
    day.mkdir(parents=True)
    (day / "071859_s1.jsonl").write_text(json.dumps({
        "type": "session_start", "endpoint": Z_AI, "session_id": "s1"}) + "\n")
    # …a later buggy resume logged itself against OpenRouter…
    (day / "075417_s1.jsonl").write_text(json.dumps({
        "type": "session_start", "endpoint": DEFAULT_ENDPOINT, "session_id": "s1"}) + "\n")
    # …and overwrote the snapshot metadata with it.
    _write_snapshot(tmp_path, "glm-5.3", "s1", endpoint=DEFAULT_ENDPOINT)
    meta = _load_snapshot_metadata("glm-5.3", "s1")
    assert meta is not None
    assert meta["endpoint"] == Z_AI


def test_metadata_without_logs_keeps_snapshot_endpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "LOG_BASE", tmp_path)
    _write_snapshot(tmp_path, "test/model", "s1", endpoint=Z_AI)
    meta = _load_snapshot_metadata("test/model", "s1")
    assert meta is not None
    assert meta["endpoint"] == Z_AI


# ── _port_snapshot_to_model ──────────────────────────────────────────────────

def test_port_copies_snapshot_and_restamps_model(monkeypatch, tmp_path, capsys):
    """Cross-model resume copies the file; the original stays untouched."""
    monkeypatch.setattr(core, "LOG_BASE", tmp_path)
    _write_snapshot(tmp_path, "glm-5.3", "s1", endpoint=Z_AI)
    path = _port_snapshot_to_model(
        {"model": "qwen/qwen3.7-max", "endpoint": "https://opencode.ai/zen/v1"}, "s1")
    assert path is not None and path.exists()
    data = json.loads(path.read_text())
    assert data["metadata"]["model"] == "qwen/qwen3.7-max"
    assert data["metadata"]["session_id"] == "s1"
    assert data["metadata"]["endpoint"] == "https://opencode.ai/zen/v1"
    assert data["metadata"]["ported_from"] == {"model": "glm-5.3", "endpoint": Z_AI}
    assert len(data["messages"]) == 1
    # Original file untouched.
    orig = json.loads((tmp_path / safe_model_name("glm-5.3")
                       / "s1_messages.json").read_text())
    assert orig["metadata"]["model"] == "glm-5.3"
    assert orig["metadata"]["endpoint"] == Z_AI
    assert "Ported session s1 from" in capsys.readouterr().out


def test_port_is_idempotent(monkeypatch, tmp_path):
    """An existing snapshot under the target model wins; nothing is copied."""
    monkeypatch.setattr(core, "LOG_BASE", tmp_path)
    _write_snapshot(tmp_path, "test/model", "s1", endpoint="https://mine.test/v1")
    _write_snapshot(tmp_path, "other/model", "s1", endpoint=Z_AI)
    path = _port_snapshot_to_model({"model": "test/model"}, "s1")
    data = json.loads(path.read_text())
    assert data["metadata"]["endpoint"] == "https://mine.test/v1"


def test_port_without_snapshot_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "LOG_BASE", tmp_path)
    assert _port_snapshot_to_model({"model": "test/model"}, "nope") is None


def test_port_skips_legacy_array_snapshots(monkeypatch, tmp_path):
    """A legacy snapshot without a messages wrapper is not portable."""
    monkeypatch.setattr(core, "LOG_BASE", tmp_path)
    d = tmp_path / safe_model_name("other/model")
    d.mkdir(parents=True)
    (d / "s1_messages.json").write_text(json.dumps([{"role": "user", "content": "x"}]))
    assert _port_snapshot_to_model({"model": "test/model"}, "s1") is None


def test_ported_session_binds_to_new_model_not_old_endpoint(monkeypatch, tmp_path, capsys):
    """End to end: cross-model resume ports the snapshot, keeps the caller's
    endpoint (the new provider) and re-saves the ported session under it."""
    from agentknit import init_session, _save_messages_snapshot
    monkeypatch.setattr(core, "LOG_BASE", tmp_path)
    _write_snapshot(tmp_path, "glm-5.3", "s2", endpoint=Z_AI)
    new_endpoint = "https://opencode.ai/zen/v1"
    schema = load_specification("qwen/qwen3.7-max", new_endpoint)
    session = init_session(schema, resumed_from="s2")
    # The new provider is used — no binding back to z.ai…
    assert session["endpoint"] == new_endpoint
    # …and the ported history was loaded.
    assert any(m.get("content") == "hi" for m in session["messages"])
    # The first snapshot the ported session saves records the new endpoint.
    _save_messages_snapshot(session)
    saved = json.loads((tmp_path / safe_model_name("qwen/qwen3.7-max")
                        / "s2_messages.json").read_text())
    assert saved["metadata"]["endpoint"] == new_endpoint
    # The original file is still the z.ai record.
    orig = json.loads((tmp_path / safe_model_name("glm-5.3")
                       / "s2_messages.json").read_text())
    assert orig["metadata"]["endpoint"] == Z_AI
    assert "Ported session s2 from" in capsys.readouterr().out


def test_same_model_resume_still_binds_to_origin(monkeypatch, tmp_path):
    """Same-model resume (the 402 bug) is unaffected by the porting path."""
    from agentknit import init_session
    monkeypatch.setattr(core, "LOG_BASE", tmp_path)
    _write_snapshot(tmp_path, "glm-5.3", "s3", endpoint=Z_AI)
    day = tmp_path / "glm-5.3" / "2026-08-30"
    day.mkdir(parents=True)
    (day / "071859_s3.jsonl").write_text(json.dumps({
        "type": "session_start", "endpoint": Z_AI, "session_id": "s3"}) + "\n")
    session = init_session(load_specification("glm-5.3", DEFAULT_ENDPOINT),
                           resumed_from="s3")
    assert session["endpoint"] == Z_AI


# ── _bind_schema_to_resumed_session ──────────────────────────────────────────

def test_bind_rewrites_endpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "LOG_BASE", tmp_path)
    _write_snapshot(tmp_path, "glm-5.3", "7cd55392612b",
                    endpoint=Z_AI,
                    auth={"keyring_service": "z.ai", "keyring_username": "api_key"})
    schema = _bind_schema_to_resumed_session(
        {"model": "glm-5.3", "endpoint": DEFAULT_ENDPOINT}, "7cd55392612b")
    assert schema["endpoint"] == Z_AI
    assert schema["keyring_service"] == "z.ai"
    assert schema["keyring_username"] == "api_key"


def test_bind_drops_key_source_the_session_never_used(monkeypatch, tmp_path):
    """A key_env that would resolve to the wrong provider is dropped, not merged."""
    monkeypatch.setattr(core, "LOG_BASE", tmp_path)
    _write_snapshot(tmp_path, "glm-5.3", "s1", endpoint=Z_AI,
                    auth={"keyring_service": "z.ai", "keyring_username": "api_key"})
    schema = _bind_schema_to_resumed_session(
        {"model": "glm-5.3", "endpoint": Z_AI, "key_env": "OPENROUTER_API_KEY"},
        "s1")
    assert "key_env" not in schema
    assert schema["keyring_service"] == "z.ai"


def test_bind_does_not_mutate_caller_schema(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "LOG_BASE", tmp_path)
    _write_snapshot(tmp_path, "glm-5.3", "s1", endpoint=Z_AI)
    original = {"model": "glm-5.3", "endpoint": DEFAULT_ENDPOINT}
    _bind_schema_to_resumed_session(original, "s1")
    assert original["endpoint"] == DEFAULT_ENDPOINT


def test_bind_noop_without_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "LOG_BASE", tmp_path)
    schema = _schema()
    assert _bind_schema_to_resumed_session(schema, "missing") is schema


def test_bind_noop_when_endpoint_already_matches(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(core, "LOG_BASE", tmp_path)
    _write_snapshot(tmp_path, "test/model", "s1", endpoint=DEFAULT_ENDPOINT)
    schema = _schema()
    bound = _bind_schema_to_resumed_session(schema, "s1")
    assert bound == schema
    assert "Resuming session" not in capsys.readouterr().out


# ── end to end: run_task builds the client from the resumed endpoint ─────────

class _RecordingClient:
    """Stands in for create_client; captures the bound schema."""

    def __init__(self, reply="ok") -> None:
        from agentknit.openai_compat import _Message, _Choice, _Usage, _Response, _SubprocessChat
        self.requests = []
        self.base_url = type("B", (), {"host": ""})()
        msg = _Message(role="assistant", content=reply, tool_calls=None)
        usage = _Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        self._response = _Response(choices=[_Choice(msg)], usage=usage)
        self.chat = _SubprocessChat(self)

    def _complete(self, model, messages, **kwargs):
        self.requests.append({"model": model, "messages": list(messages), **kwargs})
        return self._response


def test_init_session_binds_resumed_schema(monkeypatch, tmp_path, capsys):
    """The binding must also cover callers that use init_session directly
    (e.g. the TUI) — otherwise their next snapshot rewrites the endpoint."""
    from agentknit import init_session, _save_messages_snapshot
    monkeypatch.setattr(core, "LOG_BASE", tmp_path)
    day = tmp_path / "glm-5.3" / "2026-08-30"
    day.mkdir(parents=True)
    (day / "071859_s1.jsonl").write_text(json.dumps({
        "type": "session_start", "endpoint": Z_AI, "session_id": "s1"}) + "\n")
    session = init_session(
        load_specification("glm-5.3", DEFAULT_ENDPOINT), resumed_from="s1")
    assert session["endpoint"] == Z_AI
    # And the snapshot the resumed session writes records that endpoint.
    session["messages"].append({"role": "user", "content": "hi"})
    _save_messages_snapshot(session)
    saved = json.loads(
        (tmp_path / safe_model_name("glm-5.3") / "s1_messages.json").read_text())
    assert saved["metadata"]["endpoint"] == Z_AI


def test_run_task_resumes_on_recorded_endpoint(monkeypatch, tmp_path):
    from agentknit.openai_compat import _SubprocessCompletions

    def _create(self, *, model, messages, **kwargs):
        return self._client._complete(model, messages, **kwargs)

    monkeypatch.setattr(_SubprocessCompletions, "create", _create)

    monkeypatch.setattr(core, "LOG_BASE", tmp_path)
    _write_snapshot(tmp_path, "test/model", "s2", endpoint=Z_AI,
                    auth={"key_env": "ZAI_API_KEY"})

    captured = {}

    def fake_create_client(schema):
        captured["schema"] = schema
        return _RecordingClient()

    monkeypatch.setattr(core, "create_client", fake_create_client)
    schema = load_specification("test/model", DEFAULT_ENDPOINT)
    run_task(schema, "continue", session_id="s2", strict_cache_proof=False)
    assert captured["schema"]["endpoint"] == Z_AI
    assert captured["schema"]["key_env"] == "ZAI_API_KEY"
