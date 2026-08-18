# Agent Spec JSON — Field Reference

An agent spec is a JSON file that describes how agentknit should connect to a model, dispatch tools, and enforce usage limits.  Files are conventionally named `agent_spec_<identifier>.json` and are produced by [llmprobe](https://github.com/monperrus/llmprobe) or written by hand.

---

## Top-level fields

### Identity

| Field | Type | Required | Description |
|---|---|---|---|
| `model` | string | yes | Model identifier sent to the API (e.g. `"qwen/qwen3-8b"`).  For subprocess endpoints this is the path to the binary. |
| `endpoint` | string | no | Base URL of the OpenAI-compatible API (default: OpenRouter).  Use a `run://` URI (e.g. `"run:///path/to/binary"`) to invoke a local subprocess instead of an HTTP endpoint. |
| `status` | string | no | Informational label (e.g. `"default"`, `"experimental"`).  Not used by the runtime. |
| `comment` | string | no | Human-readable note.  Shown as the error message when `disabled` is `true`. |
| `disabled` | boolean | no | If `true`, the agent refuses to start and raises `AgentSpecDisabledError`. |

### Tool schema

| Field | Type | Required | Description |
|---|---|---|---|
| `tool_specs` | array | yes | OpenAI-compatible tool definitions — the list of `{"type": "function", "function": {...}}` objects sent to the model.  Entries of type `"custom"` (custom/freeform tools, see [Custom tool calls](#custom-tool-calls)) are forwarded verbatim. |
| `tools` | array of strings | no | Ordered list of Python function names from `tool_library.TOOL_LIBRARY`. Each entry is paired with the tool spec at the same index. |
| `tool_dispatch` | object | no | Legacy explicit dispatch mapping (see [Tool dispatch](#tool-dispatch) below).  Prefer `tools` for new specs. |
| `aliases` | object | no | Maps alias names to canonical tool names already present in `tool_dispatch`.  Both the tool schema and dispatch table are expanded at session start so aliases behave identically to the original tool. |

### Behaviour

| Field | Type | Required | Description |
|---|---|---|---|
| `behaviour` | object | no | Agent behaviour knobs (see [Behaviour object](#behaviour-object)). |
| `options` | array of strings | no | Extra feature flags (see [Options](#options)). |
| `max_output_tokens` | integer | no | Override the default `max_tokens` value sent with every API request. |
| `provider` | string | no | OpenRouter provider hint pinned for the session. |
| `provider_api_support` | object | no | Capability map written by llmprobe; `provider_api_support.streaming.supported` enables streaming. |

### Authentication

| Field | Type | Required | Description |
|---|---|---|---|
| `auth` | string | no | Authentication scheme.  `"opencode-github-copilot"` uses the OpenCode GitHub Copilot token flow; omit for standard API key auth. |
| `key_env` | string | no | Name of the environment variable that holds the API key (e.g. `"MISTRAL_API_KEY"`). |
| `keyring_service` | string | no | Keyring service name for retrieving the API key via the system keyring. |
| `keyring_username` | string | no | Keyring username.  Also used (uppercased, `-` → `_`) as a fallback env var name when the keyring lookup fails. |

Key resolution order: `keyring_service`+`keyring_username` → `key_env` → `OPENROUTER_API_KEY`.  Specify one key source only (`key_env` **xor** `keyring_service`+`keyring_username`); the first matching source wins, and session snapshots record only that single source.  A configured source that yields no key raises `AuthenticationError` instead of falling through.  The `OPENROUTER_API_KEY` fallback runs the balance-check/rotation flow (`agentknit.keys`) only for `openrouter.ai` endpoints; for any other endpoint it is read as a plain key, and if absent the spec must define `key_env` or `keyring_service`+`keyring_username`.

### Pricing limits

| Field | Type | Required | Description |
|---|---|---|---|
| `max_input_token_price_per_million` | number | no | Abort if the live input price (fetched from OpenRouter or Azure) exceeds this value in USD. |
| `max_output_token_price_per_million` | number | no | Abort if the live output price exceeds this value in USD. |

### Rate limiting

| Field | Type | Required | Description |
|---|---|---|---|
| `max_rpm` | integer | no | Client-side requests-per-minute cap passed to the OpenAI client constructor (e.g. `40` for NVIDIA NIM free tier). |

### Durability

| Field | Type | Required | Description |
|---|---|---|---|
| `durable` | boolean | no | Enable the write-ahead journal (default `true`). Every message append, tool call and tool result inside a turn is fsync'd to `<session_id>_journal.jsonl` as it happens, so a crashed session recovers to the exact point of failure: completed tool results are re-injected instead of re-run, and in-flight tool calls with unknown side effects are flagged for verification. `false` falls back to turn-boundary snapshots only. |

---

## Behaviour object

```json
"behaviour": {
  "call_delivery_mode": "structured_tool_calls"
}
```

| Key | Values | Description |
|---|---|---|
| `call_delivery_mode` | `"structured_tool_calls"` (default) / `"inline"` | `"structured_tool_calls"` uses the API's native function-calling mechanism.  `"inline"` injects a text-based tool schema into the system prompt and parses tool calls from the model's plain-text output. |
| `resume_rejects_stale_tool_call_ids` | boolean, default `false` | Set `true` only for providers that reject tool-call IDs minted in a previous API session on resume (HTTP 400 "Upstream request failed" — seen on opencode.ai / deepseek-v4-flash-free). When `true`, resumed history has its `tool_calls` / tool results flattened into a neutral `prior tool use: name(args) -> ok` summary line instead of being replayed as structured messages. Leave `false` (the default) so resumed history keeps real structured tool calls — flattening unconditionally teaches the model that tool results are plain text it can write itself, which it will then fabricate (see [issue #25](https://github.com/monperrus/agentknit/issues/25)). |

---

## Tool dispatch

Each key in `tool_dispatch` is a tool name matching an entry in `tool_specs`.

```json
"tool_dispatch": {
  "read_file": {
    "python_function": "t_read",
    "param_map": {}
  },
  "str_replace": {
    "python_function": "t_update",
    "param_map": { "old_str": "old", "new_str": "new" }
  }
}
```

| Key | Type | Description |
|---|---|---|
| `python_function` | string | Name of the Python callable in `tool_library.TOOL_LIBRARY` to invoke. |
| `param_map` | object | Maps model-facing argument names to Python keyword argument names.  Use `{}` for identity. |

### Custom tool calls

Responses may carry custom tool calls
(`{"type": "custom", "custom": {"name": ..., "input": "<raw text>"}}` —
OpenAI custom tools / grammar-constrained output). These are dispatched
**without JSON decoding**: the Python function receives a single `input: str`
keyword argument holding the raw text. Function calls (`type == "function"`)
are unaffected.

```python
def t_grammar(input: str) -> tuple[str, dict]:
    ...
```

Custom calls are re-serialized in their original shape in the assistant
history, so multi-turn conversations with grammar tools round-trip.

### Declaring custom tools in Python

Custom tools can be declared with `Tool(custom_format=...)` instead of
hand-writing specs. No JSON Schema is synthesized; the format object is
forwarded verbatim (both the flat Responses form and the chat-completions
`{"type": "grammar", "grammar": {...}}` nesting work):

```python
from agentknit import Tool, build_tool_spec, register_tools_in_library

tools = [Tool(
    name="apply_patch",
    description="Apply a patch that adds, updates, moves or deletes files.",
    fn=t_apply_patch,                      # fn(input: str) -> (str, dict)
    custom_format={"type": "grammar", "syntax": "lark", "definition": GRAMMAR},
)]
register_tools_in_library(tools)
schema, dispatch = build_tool_spec(tools)
# schema[0] == {"type": "custom", "name": "apply_patch", "description": ...,
#               "format": {...}}
```

With `format: {"type": "grammar"}` the endpoint does constrained decoding —
malformed tool calls become structurally impossible.

---

## Options

String flags in the `options` array enable provider-specific workarounds:

| Value | Effect |
|---|---|
| `"exclude-prompt_cache_key"` | Omits the `prompt_cache_key` field from `extra_body` in every API request.  Required for providers (e.g. NVIDIA NIM) that reject unknown `extra_body` fields. |

---

## Minimal example

```json
{
  "model": "qwen/qwen3-8b",
  "endpoint": "https://openrouter.ai/api/v1",
  "tool_specs": [
    {
      "type": "function",
      "function": {
        "name": "execute_shell_command",
        "description": "Execute a shell command and return stdout, stderr, and exit code.",
        "parameters": {
          "type": "object",
          "properties": { "command": { "type": "string" } },
          "required": ["command"]
        }
      }
    }
  ],
  "tools": ["t_run"],
  "behaviour": { "call_delivery_mode": "structured_tool_calls" }
}
```
