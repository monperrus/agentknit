"""Tests for reading endpoint, key and quirks from an inference-db entry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import agentknit._core as _core
from agentknit import inference_db, load_specification, run_task
from agentknit.exceptions import AgentSpecInvalidError, AuthenticationError
from test_injected_client import RecordingOpenAI, _stub_completions

DB = """
[endpoints.az]
name = "Azure"
url = "https://az.example/openai/deployments/{model}"
type = "chat-completions"
auth = "api-key"
[[endpoints.az.key]]
source = "env"
var = "AGENTKNIT_TEST_AZ_KEY"
[[endpoints.az.key]]
source = "file"
path = "{creds}"
field = "*.token"
[endpoints.az.headers]
x-extra = "1"
[endpoints.az.query]
api-version = "2024-12-01-preview"
[endpoints.az.params]
seed = 7
[endpoints.az.model_params.k3]
temperature = 1

[endpoints.free]
name = "Free"
url = "https://free.example/v1"
type = "chat-completions"
auth = "none"
[[endpoints.free.key]]
source = "none"
"""


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    creds = tmp_path / "creds.json"
    creds.write_text(json.dumps({"app": {"token": "from-file"}}))
    path = tmp_path / "endpoints.toml"
    path.write_text(DB.replace("{creds}", str(creds)))
    monkeypatch.setenv("INFERENCE_DB", str(path))
    monkeypatch.delenv("AGENTKNIT_TEST_AZ_KEY", raising=False)
    return path


def test_load_specification_applies_entry(db: Path) -> None:
    schema = load_specification("k3", "https://ignored.example/v1", inference_db="az")
    assert schema["inference_db"] == "az"
    assert schema["endpoint"] == "https://az.example/openai/deployments/k3?api-version=2024-12-01-preview"
    assert schema["auth_header"] == "api-key"
    assert schema["extra_headers"] == {"x-extra": "1"}
    assert schema["request_params"] == {"seed": 7, "temperature": 1}


def test_unknown_entry_is_a_spec_error(db: Path) -> None:
    with pytest.raises(AgentSpecInvalidError, match="no entry 'nope'"):
        load_specification("m", inference_db="nope")


def test_key_sources_in_order(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    schema = load_specification("k3", inference_db="az")
    assert _core._get_key_for_schema(schema) == "from-file"
    monkeypatch.setenv("AGENTKNIT_TEST_AZ_KEY", "from-env")
    assert _core._get_key_for_schema(schema) == "from-env"


def test_explicit_key_env_beats_entry(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    schema = load_specification("k3", inference_db="az")
    schema["key_env"] = "AGENTKNIT_TEST_PROXY"
    monkeypatch.setenv("AGENTKNIT_TEST_PROXY", "proxy-token")
    assert _core._get_key_for_schema(schema) == "proxy-token"


def test_no_key_found(tmp_path: Path, db: Path) -> None:
    (tmp_path / "creds.json").unlink()
    with pytest.raises(AuthenticationError, match="no key found"):
        _core._get_key_for_schema(load_specification("k3", inference_db="az"))


def test_create_client_uses_entry(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTKNIT_TEST_AZ_KEY", "k")
    # A schema that names the entry after loading (wrapper style) is applied too.
    schema = load_specification("k3", "https://stale.example/v1")
    schema["inference_db"] = "az"
    client = _core.create_client(schema)
    assert client._base_url == "https://az.example/openai/deployments/k3?api-version=2024-12-01-preview"
    assert client._auth_header == "api-key"
    assert client._api_key == "k"
    assert client._extra_headers == {"x-extra": "1"}


def test_endpoint_override_after_load_is_kept(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTKNIT_TEST_AZ_KEY", "k")
    schema = load_specification("k3", inference_db="az")
    schema["endpoint"] = "https://gateway.example/v1"
    client = _core.create_client(schema)
    assert client._base_url == "https://gateway.example/v1"
    assert client._auth_header == "api-key"


def test_auth_none(db: Path) -> None:
    schema = load_specification("m", inference_db="free")
    assert _core._get_key_for_schema(schema) == ""


def test_request_params_reach_the_request(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_completions(monkeypatch)
    schema = load_specification("k3", inference_db="az")
    client = RecordingOpenAI()
    run_task(schema, "hi", client=client, durable=False)
    assert client.requests[0]["temperature"] == 1
    assert client.requests[0]["seed"] == 7


def test_session_records_entry_for_resume(db: Path) -> None:
    session = _core.init_session(load_specification("k3", inference_db="az"), durable=False)
    assert session["auth"]["inference_db"] == "az"


def test_openrouter_entry_keeps_rotation(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_core, "get_api_key", lambda: "rotated")
    schema = {"model": "m", "endpoint": "https://openrouter.ai/api/v1", "inference_db": "free"}
    assert _core._get_key_for_schema(schema) == "rotated"


def test_entry_url_without_query() -> None:
    assert inference_db.entry_url({"url": "https://x.example/v1/"}) == "https://x.example/v1"
