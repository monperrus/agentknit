"""Minimal OpenAI-compatible client backed by requests.

Implements only the surface used by agent_probe.py:
  OpenAI(api_key, base_url)
    .base_url.host
    .chat.completions.create(model, messages, temperature, tools,
                             tool_choice, user, extra_body)
  -> response.choices[0].message
       .content
       .tool_calls[i].id / .function.name / .function.arguments
  -> response.usage.total_tokens

SubprocessOpenAI(binary_path)  — same interface but pipes JSON to a binary's
  stdin and reads the OpenAI-format JSON response from its stdout.
"""

import datetime as _dt
import json
import re
import subprocess
import time
import urllib.parse
import requests
import threading
import collections

from .exceptions import RateLimitError

# Zhipu (Z.ai / BigModel) endpoints never send Retry-After or X-RateLimit-*
# headers on a 429. Instead the reset timestamp is embedded in the JSON body:
#   {"error":{"code":"1308","message":"Usage limit reached for 5 hour.
#    Your limit will reset at 2026-08-16 05:42:29"}}
# (the Anthropic-style /api/anthropic/v1/messages variant wraps the same
# sentence in brackets). The timestamp is Beijing time (UTC+8): a request at
# 18:52 UTC got "reset at 05:42:29", i.e. 21:42 UTC — ~2.8h into a 5-hour
# window. Read as UTC it would be 11h away, which the window forbids.
_ZHIPU_HOSTS = ("z.ai", "bigmodel.cn")
_BEIJING_TZ = _dt.timezone(_dt.timedelta(hours=8))  # no DST in China
_RESET_AT_RE = re.compile(
    r"reset at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", re.IGNORECASE
)
# Zhipu's longest quota window is 5 hours; anything beyond this margin means
# the body parse (or its timezone assumption) went wrong.
_MAX_AUTO_RETRY_SECONDS = 6 * 3600


# ── rate limiter (token-bucket) ───────────────────────────────────────────────

class RateLimiter:
    """Token-bucket rate limiter for API requests.

    Ensures no more than *max_rpm* requests per minute are made.
    Uses a sliding-window approach with per-second granularity.
    """

    def __init__(self, max_rpm: int = 40) -> None:
        self._max_rpm = max_rpm
        self._lock = threading.Lock()
        # ring buffer of timestamps for the last *max_rpm* requests
        self._timestamps: collections.deque[float] = collections.deque(maxlen=max_rpm)

    def acquire(self) -> None:
        """Block until a request slot is available."""
        with self._lock:
            now = time.monotonic()
            # If the ring buffer is full, the oldest entry is at index 0.
            if len(self._timestamps) == self._max_rpm:
                oldest = self._timestamps[0]
                elapsed = now - oldest
                if elapsed < 60.0:
                    wait = 60.0 - elapsed
                    print(
                        f"  [rate-limit] {self._max_rpm} RPM limit reached — "
                        f"waiting {wait:.1f}s …",
                        flush=True,
                    )
                    time.sleep(wait)
                    now = time.monotonic()
            self._timestamps.append(now)


def _retry_delay_for_z_ai(text) -> float | None:  # type: ignore[no-untyped-def]
    """Return the retry delay (seconds) parsed from a Z.ai 429 JSON body, or None.

    Zhipu (Z.ai / BigModel) reports the quota reset time only in the error
    body — ``"Your limit will reset at 2026-08-16 05:42:29"`` — with no
    ``Retry-After`` header. The naive timestamp is Beijing time (UTC+8).
    Returns ``None`` when no timestamp is found or it is implausibly far
    in the future (guards against a misparsed date).
    """
    if not text:
        return None
    match = _RESET_AT_RE.search(text)
    if not match:
        return None
    try:
        reset = _dt.datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    reset = reset.replace(tzinfo=_BEIJING_TZ)
    delay = (reset - _dt.datetime.now(_dt.timezone.utc)).total_seconds()
    if delay > _MAX_AUTO_RETRY_SECONDS:
        return None
    return max(0.0, delay)


def _retry_delay_seconds(headers) -> float | None:  # type: ignore[no-untyped-def]
    """Return the retry delay (seconds) a 429 response tells us to wait, or None.

    Checked in order of precedence: ``retry-after-ms``, ``retry-after``
    (RFC 7231 — either delta-seconds or an HTTP-date), then
    ``x-ratelimit-reset-requests``. Returns ``None`` when no header gives
    a usable, well-formed delay. (Z.ai's body-embedded delay is handled
    separately by :func:`_retry_delay_for_z_ai`.)
    """
    if headers.get("retry-after-ms"):
        try:
            return float(headers["retry-after-ms"]) / 1000
        except (TypeError, ValueError):
            pass
    if headers.get("retry-after"):
        raw = headers["retry-after"]
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            pass
        try:
            from email.utils import parsedate_to_datetime
            import datetime as _dt
            dt = parsedate_to_datetime(raw)
            if dt is not None:
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=_dt.timezone.utc)
                delta = (dt - _dt.datetime.now(_dt.timezone.utc)).total_seconds()
                return max(0.0, delta)
        except (TypeError, ValueError):
            pass
    if headers.get("x-ratelimit-reset-requests"):
        try:
            return max(0.0, float(headers["x-ratelimit-reset-requests"]))
        except (TypeError, ValueError):
            pass
    return None


class _Function:
    __slots__ = ("name", "arguments")

    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _ToolCall:
    __slots__ = ("id", "type", "function")

    def __init__(self, id: str, function: _Function) -> None:
        self.id = id
        self.type = "function"
        self.function = function


class _Message:
    __slots__ = ("role", "content", "tool_calls")

    def __init__(self, role: str, content: str | None,
                 tool_calls: list[_ToolCall] | None) -> None:
        self.role = role
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    __slots__ = ("message",)

    def __init__(self, message: _Message) -> None:
        self.message = message


class _Usage:
    __slots__ = ("total_tokens", "prompt_tokens", "completion_tokens",
                 "cached_tokens", "cache_creation_tokens", "has_cache_proof")

    def __init__(self, total_tokens: int = 0, prompt_tokens: int = 0,
                 completion_tokens: int = 0, cached_tokens: int = 0,
                 cache_creation_tokens: int = 0,
                 has_cache_proof: bool = False) -> None:
        self.total_tokens = total_tokens
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        # cached_tokens = portion of prompt_tokens served from a prefix cache
        # (OpenAI: usage.prompt_tokens_details.cached_tokens; Anthropic cache reads).
        self.cached_tokens = cached_tokens
        # cache_creation_tokens = tokens written to the cache this call
        # (Anthropic-specific; surfaced by claude-haiku-completions.py).
        self.cache_creation_tokens = cache_creation_tokens
        # True when the provider returned an explicit cache-read or cache-write field.
        self.has_cache_proof = has_cache_proof


class _Response:
    __slots__ = ("choices", "usage", "provider", "reasoning")

    def __init__(self, choices: list[_Choice], usage: _Usage,
                 provider: str | None = None) -> None:
        self.choices = choices
        self.usage = usage
        # OpenRouter reports which upstream provider served the request.
        self.provider = provider
        # Assembled reasoning trace from streaming (None for non-streaming responses).
        self.reasoning: str | None = None


class _Completions:
    def __init__(self, client: "OpenAI") -> None:
        self._client = client

    def _build_url_and_headers(self) -> tuple[str, dict]:
        base = self._client._base_url.rstrip("/")
        if "?" in base:
            _path, _qs = base.split("?", 1)
            url = _path + "/chat/completions?" + _qs
        else:
            url = base + "/chat/completions"
        auth_hdr = self._client._auth_header
        auth_val = (self._client._api_key if auth_hdr != "Authorization"
                    else f"Bearer {self._client._api_key}")
        return url, {auth_hdr: auth_val, "Content-Type": "application/json"}

    def _retry_post(self, url: str, headers: dict, payload: dict,
                    stream: bool = False) -> requests.Response:
        while True:
            self._client._rate_limiter.acquire()
            resp = requests.post(url, headers=headers, json=payload,
                                 stream=stream, timeout=300)
            if resp.status_code == 429:
                host = (self._client.base_url.host or "").lower()
                # Reading the body consumes the stream — only do it for the
                # one provider family known to hide the reset time there.
                is_z_ai = any(h in host for h in _ZHIPU_HOSTS)
                delay = _retry_delay_seconds(resp.headers)
                if delay is None and is_z_ai:
                    delay = _retry_delay_for_z_ai(resp.text)
                if delay is None:
                    # No header or body told us when it's safe to retry —
                    # guessing a delay risks looping forever against a hard
                    # block. Surface a typed error instead so the caller
                    # (CLI/TUI) can show a clear message and stop.
                    raise RateLimitError(
                        "Rate limited (HTTP 429) and the server did not "
                        "specify a retry delay (no Retry-After / "
                        "retry-after-ms / x-ratelimit-reset-requests "
                        "header and no reset time in the response body) "
                        "— stopping instead of retrying blindly.",
                        headers=dict(resp.headers),
                    )
                print(f"  [rate-limited] waiting {delay:.1f}s …", flush=True)
                time.sleep(delay)
                continue
            return resp

    def create(self, *, model: str, messages: list, temperature: float = 0,
               tools: list | None = None, tool_choice: str | None = None,
               user: str | None = None, extra_body: dict | None = None,
               max_tokens: int | None = None,
               on_content_delta=None,
               on_reasoning_delta=None) -> _Response:
        payload: dict = {"model": model, "messages": messages,
                         "temperature": temperature}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if user is not None:
            payload["user"] = user
        if extra_body:
            payload.update(extra_body)

        url, headers = self._build_url_and_headers()

        if on_content_delta is not None:
            return self._create_streaming(url, headers, payload,
                                          on_content_delta, on_reasoning_delta)

        resp = self._retry_post(url, headers, payload)
        if not resp.ok:
            print(f"  [HTTP {resp.status_code}] {resp.text[:2000]}", flush=True)
        resp.raise_for_status()
        return _parse_response(resp.json())

    def _create_streaming(self, url: str, headers: dict, payload: dict,
                          on_content_delta, on_reasoning_delta=None) -> _Response:
        payload = {**payload, "stream": True}
        assembled_content = ""
        assembled_reasoning = ""
        assembled_tool_calls: dict[int, dict] = {}
        usage_data: dict = {}

        # _retry_post loops internally while a retryable 429 is seen, and
        # raises RateLimitError for a non-retryable one — it never returns
        # a 429 response here.
        resp = self._retry_post(url, headers, payload, stream=True)
        if not resp.ok:
            print(f"  [HTTP {resp.status_code}] {resp.text[:2000]}", flush=True)
            resp.raise_for_status()

        with resp:
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if line == "data: [DONE]":
                    break
                if not line.startswith("data: "):
                    continue
                try:
                    chunk = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                if chunk.get("usage"):
                    usage_data = chunk["usage"]

                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}

                    # reasoning_content (nvidia/stepfun) or reasoning (OpenRouter)
                    reasoning_piece = (delta.get("reasoning_content")
                                       or delta.get("reasoning") or "")
                    if reasoning_piece and on_reasoning_delta:
                        assembled_reasoning += reasoning_piece
                        on_reasoning_delta(reasoning_piece)

                    content_piece = delta.get("content") or ""
                    if content_piece:
                        assembled_content += content_piece
                        on_content_delta(content_piece)

                    for tc_delta in delta.get("tool_calls") or []:
                        idx = tc_delta.get("index", 0)
                        if idx not in assembled_tool_calls:
                            assembled_tool_calls[idx] = {
                                "id": "", "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        entry = assembled_tool_calls[idx]
                        if tc_delta.get("id"):
                            entry["id"] += tc_delta["id"]
                        fn = tc_delta.get("function") or {}
                        if fn.get("name"):
                            entry["function"]["name"] += fn["name"]
                        if fn.get("arguments"):
                            entry["function"]["arguments"] += fn["arguments"]

        tool_calls_list = [assembled_tool_calls[i]
                           for i in sorted(assembled_tool_calls)] or None
        data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": assembled_content or None,
                    "tool_calls": tool_calls_list,
                }
            }],
            "usage": usage_data,
        }
        resp = _parse_response(data)
        resp.reasoning = assembled_reasoning or None
        return resp


def _parse_response(data: dict) -> _Response:
    """Build a _Response from a parsed OpenAI-format dict (shared by HTTP and subprocess)."""
    choices = []
    for c in data.get("choices", []):
        m = c.get("message", {})
        raw_tcs = m.get("tool_calls")
        tool_calls = None
        if raw_tcs:
            tool_calls = [
                _ToolCall(
                    id=tc.get("id", ""),
                    function=_Function(
                        name=(tc.get("function") or {}).get("name", ""),
                        arguments=(tc.get("function") or {}).get("arguments", "{}"),
                    ),
                )
                for tc in raw_tcs
            ]
        choices.append(_Choice(_Message(
            role=m.get("role", "assistant"),
            content=m.get("content"),
            tool_calls=tool_calls,
        )))
    u = data.get("usage") or {}
    details = u.get("prompt_tokens_details") or {}
    cached_candidates = [
        details.get("cached_tokens"),
        u.get("cache_read_input_tokens"),
        u.get("cache_read_tokens"),
        details.get("cache_read_tokens"),
        details.get("cache_read_input_tokens"),
    ]
    cache_creation_candidates = [
        u.get("cache_creation_input_tokens"),
        u.get("cache_write_tokens"),
        details.get("cache_write_tokens"),
        details.get("cache_creation_input_tokens"),
    ]
    cached_tokens = next((int(v) for v in cached_candidates if v is not None), 0)
    cache_creation = next((int(v) for v in cache_creation_candidates if v is not None), 0)
    has_cache_proof = any(v is not None for v in [*cached_candidates, *cache_creation_candidates])
    return _Response(choices, _Usage(
        total_tokens=u.get("total_tokens", 0) or 0,
        prompt_tokens=u.get("prompt_tokens", 0) or 0,
        completion_tokens=u.get("completion_tokens", 0) or 0,
        cached_tokens=cached_tokens,
        cache_creation_tokens=cache_creation,
        has_cache_proof=has_cache_proof,
    ), provider=data.get("provider"))


class _Chat:
    def __init__(self, client: "OpenAI") -> None:
        self.completions = _Completions(client)


class _BaseURL:
    """Exposes .host so agent_probe can check for openrouter.ai."""

    def __init__(self, url: str) -> None:
        self.host = urllib.parse.urlparse(url).hostname or ""


class OpenAI:
    # Class-level rate limiter shared across all instances targeting the same
    # base URL.  NVIDIA NIM free-tier endpoints are limited to 40 RPM.
    _rate_limiters: dict[str, RateLimiter] = {}
    _rate_limiters_lock = threading.Lock()

    @classmethod
    def _get_rate_limiter(cls, base_url: str, max_rpm: int = 40) -> RateLimiter:
        """Return a per-base-url RateLimiter (shared across instances)."""
        with cls._rate_limiters_lock:
            if base_url not in cls._rate_limiters:
                cls._rate_limiters[base_url] = RateLimiter(max_rpm)
            return cls._rate_limiters[base_url]

    def __init__(self, *, api_key: str, base_url: str, auth_header: str = "Authorization",
                 max_rpm: int = 40) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._auth_header = auth_header
        self.base_url = _BaseURL(base_url)
        self.chat = _Chat(self)
        # Acquire a per-base-url rate limiter (default 40 RPM for NVIDIA NIM).
        self._rate_limiter = self._get_rate_limiter(base_url, max_rpm)


# ── subprocess backend ────────────────────────────────────────────────────────

class _SubprocessCompletions:
    def __init__(self, client: "SubprocessOpenAI") -> None:
        self._client = client

    def create(self, *, model: str, messages: list, temperature: float = 0,
               tools: list | None = None, tool_choice: str | None = None,
               user: str | None = None, extra_body: dict | None = None,
               max_tokens: int | None = None,
               on_content_delta=None,
               on_reasoning_delta=None) -> _Response:
        payload: dict = {"messages": messages, "temperature": temperature}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        # Strip run:// model identifiers so the binary uses its own default.
        if not model.startswith("run://"):
            payload["model"] = model
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if user is not None:
            payload["user"] = user
        if extra_body:
            payload.update(extra_body)

        proc = subprocess.run(
            [self._client._binary_path],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if proc.stderr:
            print(f"  [subprocess stderr] {proc.stderr}", flush=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Binary {self._client._binary_path!r} exited {proc.returncode}: "
                f"{proc.stderr}"
            )
        data = json.loads(proc.stdout)
        return _parse_response(data)


class _SubprocessChat:
    def __init__(self, client: "SubprocessOpenAI") -> None:
        self.completions = _SubprocessCompletions(client)


class SubprocessOpenAI:
    """Client that routes completions to a local binary via stdin/stdout."""

    def __init__(self, binary_path: str) -> None:
        self._binary_path = binary_path
        self.base_url = _BaseURL("")   # empty host → no openrouter-specific headers
        self.chat = _SubprocessChat(self)
