"""Read endpoint, key and request quirks from an inference-db entry.

inference-db (https://github.com/monperrus/inference-db) keeps one TOML table
per inference endpoint in ``~/.config/inference-db/endpoints.toml`` (or
``$INFERENCE_DB``): base URL, API type, where the key lives, and declarative
quirks (auth header style, extra headers, query parameters, forced body
params, per-model params).  This module reads it with the standard library
only, so agentknit gains no dependency.

A schema opts in with ``"inference_db": "<entry id>"`` (or ``--inference-db``
on the CLI); :func:`apply_entry` then fills ``endpoint``, ``auth_header``,
``extra_headers`` and ``request_params``, and :func:`resolve_key` walks the
entry's key sources.
"""

from __future__ import annotations

import json
import os
import subprocess
import tomllib
import urllib.parse
from pathlib import Path
from typing import Any

from .exceptions import AuthenticationError, AgentSpecInvalidError

# inference-db `auth` value -> header name for the OpenAI-compatible client.
AUTH_HEADERS = {"bearer": "Authorization", "x-api-key": "x-api-key", "api-key": "api-key", "none": "Authorization"}


def db_path() -> Path:
    env = os.environ.get("INFERENCE_DB")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".config" / "inference-db" / "endpoints.toml"


def load_entry(entry_id: str, path: Path | None = None) -> dict[str, Any]:
    """Return the raw ``[endpoints.<entry_id>]`` table."""
    path = path or db_path()
    try:
        with open(path, "rb") as f:
            endpoints = tomllib.load(f).get("endpoints", {})
    except FileNotFoundError:
        raise AgentSpecInvalidError(f"inference-db not found at {path}") from None
    if entry_id not in endpoints:
        raise AgentSpecInvalidError(f"inference-db {path} has no entry {entry_id!r}")
    entry = dict(endpoints[entry_id])
    entry["id"] = entry_id
    return entry


def entry_url(entry: dict[str, Any], model: str = "") -> str:
    """Base URL with ``{model}`` filled and the entry's query quirks appended."""
    url = str(entry["url"]).replace("{model}", model).rstrip("/")
    query = entry.get("query") or {}
    if query:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(query)
    return url


def request_params(entry: dict[str, Any], model: str = "") -> dict[str, Any]:
    """Body fields the entry forces: ``params`` then ``model_params[model]``."""
    return {**(entry.get("params") or {}), **((entry.get("model_params") or {}).get(model) or {})}


def apply_entry(schema: dict[str, Any], entry: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a copy of *schema* with the entry's endpoint and quirks applied.

    The entry always decides the endpoint; explicit key settings already in
    the schema (``keyring_*``/``key_env``) are left alone and keep precedence
    over the entry's key sources (see ``_get_key_for_schema``).
    """
    entry = entry or load_entry(str(schema["inference_db"]))
    model = str(schema.get("model") or "")
    out = dict(schema)
    out["inference_db"] = entry["id"]
    out["endpoint"] = entry_url(entry, model)
    out["auth_header"] = AUTH_HEADERS.get(str(entry.get("auth", "bearer")), "Authorization")
    if entry.get("headers"):
        out["extra_headers"] = {**(schema.get("extra_headers") or {}), **entry["headers"]}
    params = request_params(entry, model)
    if params:
        out["request_params"] = {**(schema.get("request_params") or {}), **params}
    return out


def _json_field(data: Any, dotted: str) -> Any:
    for part in dotted.split("."):
        if part == "*":
            data = next(iter(data.values()), None) if isinstance(data, dict) else None
        else:
            data = data.get(part) if isinstance(data, dict) else None
    return data


def _resolve_source(source: dict[str, Any]) -> str | None:
    kind = source.get("source")
    if kind == "env":
        return os.environ.get(str(source["var"])) or None
    if kind == "keyring":
        try:
            import keyring as kr
        except Exception:  # noqa: BLE001 - optional dependency
            return None
        from ._core import _keyring_get_password_with_timeout
        return _keyring_get_password_with_timeout(kr, str(source["service"]), str(source["username"]))
    if kind == "file":
        p = Path(str(source["path"])).expanduser()
        if not p.exists():
            return None
        val = _json_field(json.loads(p.read_text()), str(source["field"]))
        return val if isinstance(val, str) and val else None
    if kind == "command":
        cmd = str(source["command"])
        if "(" in cmd:  # descriptive pointer, not runnable
            return None
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)
        return (r.stdout.strip() or None) if r.returncode == 0 else None
    return None  # none / unknown; plaintext sources are never read


def resolve_key(entry: dict[str, Any]) -> str:
    """Return the first key found among the entry's sources, in order."""
    sources = entry.get("key") or []
    for source in sources:
        if source.get("source") == "none":
            return ""
        val = _resolve_source(source)
        if val:
            return val
    tried = ", ".join(str(s.get("source")) for s in sources) or "none declared"
    raise AuthenticationError(f"inference-db entry {entry['id']!r}: no key found ({tried})")
