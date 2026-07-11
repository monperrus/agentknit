# agentknit

Spec-driven coding agent framework for any OpenAI-compatible endpoint.

Reads a [JSON spec](specification.md) runs an interactive coding agent that dispatches tool calls (read_file, write_file, execute_bash, …) to Python implementations.

## Install

```
pip install agentknit
```

### Programmatic

```python
from agentknit import load_or_probe, run_task

schema = load_or_probe("qwen/qwen3-8b", "https://openrouter.ai/api/v1", force=False)
result = run_task(schema, "List the files in /tmp")
print(result.final_reply)
```

## Strict Cache Proof

`agentknit` now runs in strict cache-proof mode by default.

After the first LLM call in a session, every later call must return explicit
server-side cache accounting with a nonzero cache hit. If the provider does
not expose cache-proof fields, or reports `cached_tokens = 0`, the run aborts.

This is fail-closed by design: models that cannot prove cache reuse should be
treated as unsupported for cache-sensitive workloads.

The usage layer normalizes several provider response shapes into one check,
including:

- `usage.prompt_tokens_details.cached_tokens`
- `usage.cache_read_input_tokens`
- `usage.cache_read_tokens`
- cache-write fields such as `cache_creation_input_tokens` and `cache_write_tokens`

Programmatic calls accept `strict_cache_proof=True` by default:

```python
result = run_task(schema, "List the files in /tmp", strict_cache_proof=True)
```

CLI usage is also strict by default. To opt out explicitly:

```bash
agent-probe <model> --no-strict-cache-proof
```

### Defining tools with `Tool` & `build_tool_spec`

Declare tools using the `Tool` dataclass and convert them into the schema/dispatch
pair that the agent loop expects with `build_tool_spec`:

```python
from agentknit import Tool, build_tool_spec, register_tools_in_library
from agentknit.tool_library import TOOL_LIBRARY


def t_read(path: str) -> tuple[str, dict]:
    """Read and return the contents of a file."""
    return Path(path).read_text(), {"result": "ok"}


def t_write(path: str, content: str) -> tuple[str, dict]:
    """Write content to a file."""
    Path(path).write_text(content)
    return f"wrote {len(content)} bytes", {"result": "ok"}


tools = [
    Tool("read_file", "Read a file", t_read,
         parameters={"type": "object",
                     "properties": {"path": {"type": "string"}},
                     "required": ["path"]}),
    Tool("write_file", "Write a file", t_write,
         parameters={"type": "object",
                     "properties": {"path": {"type": "string"},
                                    "content": {"type": "string"}},
                     "required": ["path", "content"]}),
]

# Build the OpenAI-compatible schema and dispatch dict
schema, dispatch = build_tool_spec(tools)

# Register the callables so the dispatch loop can find them
register_tools_in_library(tools)

# Now schema and dispatch can be used with init_session / run_task
```

The `Tool` dataclass also supports `param_map` for translating model-facing
argument names to Python keyword argument names:

```python
tools = [
    Tool("read_file", "Read a file", t_read,
         param_map={"file_path": "path"}),
]
```

### Tool library

The framework ships with a built-in set of tools (`read_file`, `write_file`,
`str_replace`, `execute_shell_command`). 

### Sandboxed tool execution (Linux)

Direct local tool dispatch remains the default. For untrusted replay workloads,
pass a `BubblewrapToolExecutor` to keep file and shell tools in an isolated
workspace while the controller retains the model credential and network access:

```python
from pathlib import Path
from agentknit import BubblewrapToolExecutor, SandboxPolicy, run_task

executor = BubblewrapToolExecutor(SandboxPolicy(
    workspace=Path("/tmp/replay-worktree"),
    network="none",
    environment={"PATH": "/usr/bin:/bin"},
))
result = run_task(schema, task, tool_executor=executor)
```

The Bubblewrap executor supports the built-in file tools and synchronous shell
commands. It rejects custom Python and asynchronous tools unless they provide a
sandbox adapter; it never falls back to local execution. Paths are restricted
to the workspace and the selected sandbox policy is recorded in the session log.


## Event System

agentknit emits events during agent execution so you can build logging
frameworks, GUI/TUI integrations, streaming dashboards, or custom monitoring
on top of the framework.

### Subscribing to events

Use `subscribe(session, event_type, handler)` to register a handler for a
specific event type:

```python
from agentknit import subscribe, init_session, run_turn, create_client

schema = load_or_probe("qwen/qwen3-8b", "https://openrouter.ai/api/v1", False)
client = create_client(schema)
session = init_session(schema)

# Log every tool call
subscribe(session, "tool_call", lambda event_type, data: print(f"[tool] {data['name']}"))

# Stream content deltas in real-time
subscribe(session, "content_delta", lambda event_type, data: print(data.get("text", ""), end=""))

# Track token usage
subscribe(session, "usage", lambda event_type, data: print(f"[tokens] {data}"))

# React to errors
subscribe(session, "error", lambda event_type, data: print(f"[error] {data['text']}"))
```

The `on` function is a convenience alias for `subscribe`:

```python
from agentknit import on

on(session, "tool_call", my_handler)
```

Multiple handlers can be registered for the same event type; they are called
in registration order.

### Unsubscribing

```python
from agentknit import unsubscribe

unsubscribe(session, "tool_call", my_handler)
```

### Generic handler

The lower-level `EventCallback` can be passed to `init_session()` via the
`on_event` keyword and receives *all* events:

```python
from agentknit import EventCallback

def my_handler(event_type: str, data: dict) -> None:
    print(f"[{event_type}] {data.get('fmt', data)}")

session = init_session(schema, on_event=my_handler)
```

Per-event-type handlers registered via `subscribe` are called *before* the
generic `on_event` handler.

### Full list of event types

| Event type | When it fires | Data keys |
|---|---|---|
| `tool_call` | Before dispatching a tool | `name`, `args`, `fmt` |
| `tool_result` | After receiving tool result | `name`, `result`, `streamed`, `fmt` |
| `content_delta` | Streaming text chunk from the model | `text`, `first`, `no_newline`, `fmt` |
| `reasoning_delta` | Streaming reasoning trace | `text`, `first`, `no_newline`, `fmt` |
| `content_stream_end` | End of a streaming content sequence | `no_newline`, `fmt` |
| `reasoning_stream_end` | End of a streaming reasoning sequence | `no_newline`, `fmt` |
| `usage` | Per-turn token usage report | `prompt`, `completion`, `total`, `cached`, `cache_write`, `fmt` |
| `session_usage` | Cumulative session usage at final answer | `prompt`, `completion`, `total`, `cached`, `cache_write`, `fmt` |
| `error` | API or dispatch error | `text`, `fmt` |
| `final_answer` | Agent produces its final reply | `text`, `fmt` |
| `token_limit` | Token budget exceeded | `used`, `limit`, `fmt` |
| `session_resumed` | Session history was loaded from disk | `session_id`, `messages_loaded`, `fmt` |
| `provider_pinned` | OpenRouter provider was locked for the session | `provider`, `fmt` |
| `compaction` | Context was compacted into a summary | `summary`, `compacted_turns`, `fmt` |

Every event data dict includes a `"fmt"` key containing a pre-formatted ANSI
string suitable for direct printing to a terminal — this is what the default
handler uses.  Custom handlers may ignore `"fmt"` and use the other keys
instead.

## Context Compaction

Long sessions automatically compact when the prompt token budget is exceeded.
The oldest portion of the conversation is summarized by the model into a
continuation-oriented summary that preserves coding state (objectives, files
touched, errors, next steps). The summary replaces the compacted prefix, while
the most recent turns remain in raw form.

Compaction is **enabled by default** and configured via the agent spec or
programmatic arguments:

```python
from agentknit import run_task

result = run_task(
    schema,
    "Implement feature X",
    compaction_enabled=True,
    compaction_trigger_tokens=100_000,   # trigger when prompt tokens reach this
    compaction_target_tokens=20_000,     # max tokens for the summary call
    compaction_keep_last_turns=2,        # raw turns to keep after compaction
)
```

Or in the agent spec JSON:

```json
{
  "model": "...",
  "compaction_enabled": true,
  "compaction_trigger_tokens": 100000,
  "compaction_target_tokens": 20000,
  "compaction_keep_last_turns": 2
}
```

The summary message is tagged with `"compacted_summary": true` so consumers
can distinguish compacted state from raw conversation turns. Compaction events
are emitted as `"compaction"` events and logged to the session trace.

## rtk Integration (optional token savings)

[rtk](https://github.com/rtk-ai/rtk) is a CLI proxy that rewrites shell
command output for 60–90% token savings. When installed, you can opt in by
calling `enable_rtk_rewrite()` once before `run_task()`:

```python
from agentknit import enable_rtk_rewrite, run_task

enable_rtk_rewrite()   # no-op if rtk is not in PATH

result = run_task(schema, "List the files in /tmp")
```

This patches `t_run` and `t_execute_async` in the tool library so every shell
command passes through `rtk rewrite` before execution. It is off by default.
