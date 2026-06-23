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
| `inferred_tool_schema` | array | yes | OpenAI-compatible tool definitions — the list of `{"type": "function", "function": {...}}` objects sent to the model. |
| `tool_dispatch` | object | no | Maps each tool name to a Python function entry (see [Tool dispatch](#tool-dispatch) below).  Omit to use the framework's built-in dispatch. |
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

Key resolution order: `keyring_service`+`keyring_username` → `key_env` → `OPENROUTER_API_KEY`.

### Pricing limits

| Field | Type | Required | Description |
|---|---|---|---|
| `max_input_token_price_per_million` | number | no | Abort if the live input price (fetched from OpenRouter or Azure) exceeds this value in USD. |
| `max_output_token_price_per_million` | number | no | Abort if the live output price exceeds this value in USD. |

### Rate limiting

| Field | Type | Required | Description |
|---|---|---|---|
| `max_rpm` | integer | no | Client-side requests-per-minute cap passed to the OpenAI client constructor (e.g. `40` for NVIDIA NIM free tier). |

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

---

## Tool dispatch

Each key in `tool_dispatch` is a tool name matching an entry in `inferred_tool_schema`.

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
  "inferred_tool_schema": [
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
  "behaviour": { "call_delivery_mode": "structured_tool_calls" }
}
```

