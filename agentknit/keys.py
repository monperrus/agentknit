#!/usr/bin/env python3
"""OpenRouter API key management: balance check and automatic key rotation.

If the current key's remaining credit is below LOW_BALANCE_THRESHOLD, a new
key with NEW_KEY_LIMIT dollars is created via OPENROUTER_MGMT_KEY (same
approach as ~/bin/openrouter-keys.py), saved to keyring under
OPENROUTER_API_KEY, and used for the rest of the process.
"""

import datetime
import json
import os
import subprocess
import sys
import urllib.request

from .exceptions import AuthenticationError
import urllib.error

_cached_key: str | None = None

_KEYRING_SERVICE      = "login2"
_KEY_NAME             = "OPENROUTER_API_KEY"
LOW_BALANCE_THRESHOLD = 0.10
NEW_KEY_LIMIT         = 10.0


def _get_raw_key(key_name: str) -> str | None:
    """Read a key from env → password-get, in that order."""
    val = os.environ.get(key_name)
    if val:
        return val
    try:
        out = subprocess.check_output(
            ["password-get", key_name], text=True, stderr=subprocess.DEVNULL
        ).strip()
        return out if out and out != "None" else None
    except Exception:
        return None


def _get_mgmt_key() -> str:
    """Return OPENROUTER_MGMT_KEY from env or keyring (mirrors openrouter-keys.py)."""
    key = _get_raw_key("OPENROUTER_MGMT_KEY")
    if not key:
        raise AuthenticationError(
            "Could not retrieve OPENROUTER_MGMT_KEY from environment or keyring."
        )
    return key


def _check_balance(key: str) -> float | None:
    """Return remaining USD credit on *key*, or None if unlimited / unreachable."""
    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode()).get("data", {})
        limit = data.get("limit")
        if limit is None:
            return None  # unlimited key — no rotation needed
        return float(limit) - float(data.get("usage") or 0)
    except Exception:
        return None


def _create_key(mgmt_key: str, name: str, limit: float) -> str | None:
    """Create a new OpenRouter sub-key (mirrors openrouter-keys.py create_key)."""
    date_str   = datetime.datetime.now().strftime("%Y-%m-%d")
    final_name = f"{name}-{date_str}-${limit:.0f}"
    payload    = json.dumps({"name": final_name, "limit": limit}).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/keys",
        data=payload,
        headers={
            "Authorization": f"Bearer {mgmt_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
        return result["key"]
    except urllib.error.HTTPError as e:
        print(f"Warning: failed to create OpenRouter key: HTTP {e.code} {e.read().decode()}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Warning: failed to create OpenRouter key: {e}", file=sys.stderr)
        return None


def _save_key_to_keyring(key: str) -> None:
    try:
        import keyring as kr
        kr.get_keyring().set_password(_KEYRING_SERVICE, _KEY_NAME, key)
    except Exception as e:
        print(f"Warning: could not save new key to keyring: {e}", file=sys.stderr)


def ensure_api_key() -> str:
    """Return a usable OPENROUTER_API_KEY, rotating to a new $10 key if balance < $0.10.

    Result is cached for the lifetime of the process.
    """
    global _cached_key
    if _cached_key:
        return _cached_key

    key = _get_raw_key(_KEY_NAME)
    if not key:
        raise AuthenticationError(
            "Cannot obtain OPENROUTER_API_KEY from environment or keyring."
        )

    balance = _check_balance(key)
    if balance is not None:
        if balance < LOW_BALANCE_THRESHOLD:
            print(
                f"OpenRouter balance ${balance:.4f} < ${LOW_BALANCE_THRESHOLD:.2f} — "
                f"creating new ${NEW_KEY_LIMIT:.0f} key …"
            )
            mgmt_key = _get_mgmt_key()
            new_key  = _create_key(mgmt_key, "auto", NEW_KEY_LIMIT)
            if new_key:
                _save_key_to_keyring(new_key)
                os.environ[_KEY_NAME] = new_key
                key = new_key
                print("New key created and saved to keyring.")
            else:
                print("Warning: proceeding with existing key.", file=sys.stderr)
        else:
            print(f"OpenRouter balance: ${balance:.4f}")

    _cached_key = key
    return key
