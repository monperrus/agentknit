# agentknit

Coding agent framework for any `/chat/completion` endpoint.

Features:
* Reads a [JSON spec](specification.md) to dispatches tool calls (read_file, write_file, execute_bash, …) to tool implementations.
* Grammar-constrained custom tools for endpoint-side constrained decoding
* Fail-closed strict cache-proof mode for saving your $$$$$
* Rich event system (`tool_call`, `content_delta`, `usage`, …) for logging/TUI/dashboards
* Automatic context compaction keeps long sessions inside the token budget
* Bubblewrap-sandboxed tool execution for untrusted replay workloads
* `rtk` integration cuts shell tool output tokens by 60–90%

## Install

```
pip install agentknit
```

### Programmatic

```python
from agentknit import load_specification, run_task

schema = load_specification("qwen3-8b.json", "https://openrouter.ai/api/v1")
result = run_task(schema, "List the files in /tmp")
print(result.final_reply)
```

To load a pre-written spec file directly (no name-based lookup, no probing), pass
`spec_path=`:

```python
schema = load_specification(
    "deepseek-v4-flash-free",
    "run:///home/user/bin/completions.py",
    spec_path="/home/user/.config/agentknit/agent_spec.json",
)
```

### Quick scripts with direct tools

For small agents, use `run_agent()` to provide the model connection and
`Tool` definitions directly. It builds the internal schema and registers the
tool callables for you:

```python
from pathlib import Path

from agentknit import Tool, run_agent


def list_files(path: str) -> tuple[str, dict]:
    return ", ".join(p.name for p in Path(path).iterdir()), {"result": "ok"}


result = run_agent(
    task="List the files in /tmp",
    model="deepseek-v4-flash-free",
    endpoint="https://opencode.ai/zen/v1",
    auth="opencode-github-copilot",
    tools=[Tool("list_files", "List a directory", list_files)],
)
print(result.final_reply)
```

### Injecting a custom client

All entry points — `run_task`, `run_agent`, `run` and `run_repl`/`run_async_repl` — accept
an optional `client=`. Pass your own `OpenAI`- or `SubprocessOpenAI`-compatible
object (sandbox client, wrapper, instrumented subclass) instead of the one
`create_client` builds from the spec:

```python
from agentknit import load_specification, run_repl
from mybridge import GrammarOpenAI   # subclass of SubprocessOpenAI

client = GrammarOpenAI("./completions.sh")
agentknit.run_repl(load_specification("run://grammar"), client=client)
```

The same client can be reused for `init_session` + `run_turn` one-shot calls
in the same script — no monkey-patching of `create_client`.


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

#### Provider minimum cacheable prompt size

Some providers cache nothing below a documented minimum prompt size —
e.g. Anthropic Claude Haiku requires roughly 4096 input tokens before any
caching kicks in, while GPT-5.6-class models cache from roughly 1024 tokens.
A short prompt that legitimately misses the cache would otherwise trip
strict cache-proof mode and abort the run with `CacheProofError`.

Set `min_cacheable_tokens` (prompt-token floor) to tell agentknit that a
zero-cache-hit response is *expected*, not a failure, whenever the current
call's prompt is below that floor:

```python
result = run_task(schema, "Hello", min_cacheable_tokens=4096)
```

Or in the agent spec JSON:

```json
{
  "model": "...",
  "min_cacheable_tokens": 4096
}
```

Or from the CLI:

```bash
agent-probe <model> --min-cacheable-tokens 4096
```

Defaults to `0` (no minimum) — any zero-cache-hit call after the first is
still treated as a genuine cache miss unless you configure this.

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

#### Custom (grammar-constrained) tools

`Tool(custom_format=...)` declares an OpenAI **custom tool**: the argument is
raw text, optionally constrained by a grammar for endpoint-side constrained
decoding — no JSON Schema is synthesized:

```python
tools = [Tool(
    "apply_patch",
    "Apply a patch that adds, updates, moves or deletes files.",
    t_apply_patch,                # fn(input: str) -> (str, dict)
    custom_format={"type": "grammar", "syntax": "lark", "definition": GRAMMAR},
)]
schema, dispatch = build_tool_spec(tools)
# schema[0] == {"type": "custom", "name": "apply_patch", ..., "format": {...}}
```

The model's raw text arrives as the single `input` keyword argument.

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

schema = load_specification("qwen/qwen3-8b", "https://openrouter.ai/api/v1", False)
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

`init_session()` returns a `Session` — a `TypedDict`, i.e. a plain `dict` at
runtime, with typed, IDE-completable keys (`session["messages"]`,
`session["usage_totals"]`, …). Import it from the package when you want the
annotations in your own code:

```python
from agentknit import Session, init_session

session: Session = init_session(schema)
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
| `tool_result` | After receiving tool result | `name`, `result`, `streamed`, `files`, `diff_summary`, `fmt` |
| `content_delta` | Streaming text chunk from the model | `text`, `first`, `no_newline`, `fmt` |
| `reasoning_delta` | Streaming reasoning trace | `text`, `first`, `no_newline`, `fmt` |
| `content_stream_end` | End of a streaming content sequence | `no_newline`, `fmt` |
| `reasoning_stream_end` | End of a streaming reasoning sequence | `no_newline`, `fmt` |
| `usage` | Per-turn token usage report | `prompt`, `completion`, `total`, `cached`, `cache_write`, `fmt` |
| `session_usage` | Cumulative session usage at final answer | `prompt`, `completion`, `total`, `cached`, `cache_write`, `fmt` |
| `error` | API or dispatch error | `text`, `error_class`, `http_status`, `error_code`, `error_message`, `elapsed_s`, `adapter`, `fmt` |
| `final_answer` | Agent produces its final reply | `text`, `fmt` |
| `token_limit` | Token budget exceeded | `used`, `limit`, `fmt` |
| `session_resumed` | Session history was loaded from disk | `session_id`, `messages_loaded`, `fmt` |
| `provider_pinned` | OpenRouter provider was locked for the session | `provider`, `fmt` |
| `compaction` | Context was compacted into a summary | `summary`, `compacted_turns`, `fmt` |
| `cache_cold` | Resumed turn missed the (expired) prefix cache | `age`, `fmt` |
| `journal_recovered` | A resumed session was rebuilt from the durable journal | `entries_replayed`, `messages_loaded`, `pending`, `mid_turn`, `fmt` |

Every event data dict includes a `"fmt"` key containing a pre-formatted ANSI
string suitable for direct printing to a terminal — this is what the default
handler uses.  Custom handlers may ignore `"fmt"` and use the other keys
instead.

The `tool_result` event includes additional metadata for file-writing tools:

- **`files`** — a list of file paths that were created or modified by the tool
  call (e.g. `["src/main.py"]`).  `None` for tools that don't touch files.
- **`diff_summary`** — a dict with `path`, `added` (lines added), and `removed`
  (lines removed) so consumers can display summaries like `+5 -2 src/main.py`
  without re-reading the file.  `None` for non-file tools.

Example::

    subscribe(session, "tool_result", lambda et, data: print(
        f"Files changed: {data.get('files')}  "
        f"Diff: {data.get('diff_summary')}"
    ))

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

## Durable Recovery

Snapshots alone only persist the conversation at *turn boundaries* — a crash
mid-turn loses the whole turn from the transcript, including tool calls whose
side effects (file writes, shell commands, deploys) already happened. Resuming
from the snapshot then makes the model blindly re-run those tools.

By default (`durable=True`) agentknit keeps an append-only, fsync-per-record
write-ahead journal next to the snapshot:

```
~/.local/share/agent_probe/<model>/<session_id>_journal.jsonl
```

Every state transition inside a turn is journaled *as it happens*:

| Record | Written | Meaning on recovery |
|---|---|---|
| `message` | when a message joins the history | the conversation is rebuildable past the last snapshot |
| `tool_start` | **before** a tool executes | crash before `tool_end` → side effects unknown |
| `tool_end` | **after** the tool returns | result is known; never re-run it |
| `turn_start` / `turn_end` | around each turn | `turn_start` without `turn_end` → crashed mid-turn |
| `reset_messages` | compaction, `/clear` | history replacement is replayed, not lost |

On resume (`--session <id>`) the journal is replayed and takes precedence over
a stale snapshot:

- messages the snapshot never saw are recovered;
- a tool that finished but whose result the model never saw is **not re-run** —
  the recorded result is re-injected into the conversation as a recovery note;
- a tool that was in flight when the process died has *unknown* side effects —
  a recovery note tells the model to verify state (inspect files, read-only
  checks) before re-running anything.

A torn tail write (crash mid-line) is ignored: an incomplete record was never
acknowledged.

Disable with `durable=False` (programmatic), `"durable": false` in the agent
spec JSON, or `--no-durable` on the CLI:

```python
from agentknit import run_task

result = run_task(schema, task, durable=False)
```

A `journal_recovered` event is emitted whenever a resume rebuilt state from
the journal.

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
