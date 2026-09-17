# agentknit

Coding agent framework for any `/chat/completion` endpoint.

Features:
* Reads a [JSON spec](specification.md) to dispatches tool calls (read_file, write_file, execute_bash, …) to tool implementations.
* Grammar-constrained custom tools for endpoint-side constrained decoding
* Fail-closed strict cache-proof mode for saving your $$$$$
* Rich event system (`tool_call`, `content_delta`, `usage`, …) for logging/TUI/dashboards
* Automatic context compaction keeps long sessions inside the token budget
* Model-facing [token awareness](https://www.monperrus.net/martin/token-awareness): a true countdown in the model's context, with a checkpoint reminder before compaction
* Model-facing time awareness: every turn opens with `session elapsed · last tool · wall since your previous message`, and every tool result is stamped with its ISO-8601 start/end/duration
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

The **first** LLM call of a session must return explicit server-side cache
accounting (a cache-proof field or a cache write). If it does not, caching
cannot work at all and the run aborts — cheaply, before anything else has
been paid for. This is fail-closed by design: models that cannot prove cache
reuse should be treated as unsupported for cache-sensitive workloads.

After the first call, a response with no cache accounting (or no cache hit
on a prompt above the provider's minimum cacheable size) no longer aborts:
the turn's tokens are already paid, so the run continues automatically and
emits a temporary `cache_proof_missing` warning instead. UIs can surface it
in a status bar via `session["_cache_status"]` (`"missing"` while the
warning is active, back to `"ok"` on the next observed cache hit).

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
`str_replace`, `exec_shell`). The previous name `execute_shell_command` is
still accepted as an alias. `str_replace` replaces the first occurrence of
`old_str` by default; pass `replace_all: true` to replace every occurrence.

`tool_library` also carries `list_dir`, `glob` and `search_files`, plus a few
older functions kept for compatibility. `CANONICAL_TOOLS` maps each tool name
to the implementation to use, and `SUPERSEDED_TOOLS` maps the rest to their
replacement:

```python
from agentknit import CANONICAL_TOOLS, SUPERSEDED_TOOLS

CANONICAL_TOOLS["search_files"]     # 't_search'
SUPERSEDED_TOOLS["t_find_files"]    # 't_glob'
```

### Using the tool runtime from another host

The sections above have agentknit drive the model. The opposite arrangement
works too: something else runs the conversation — Claude Code over MCP, an
editor over ACP, your own loop — and agentknit supplies only the tools.

Two public functions cover it. `default_tool_spec()` returns the built-in
tools as a `(schema, dispatch)` pair, and `dispatch()` runs one call:

```python
from agentknit import default_tool_spec, dispatch

schema, tool_dispatch = default_tool_spec()   # publish schema to your host

text, meta = dispatch("read_file", {"path": "README.md"}, tool_dispatch)
if not meta.get("ok", True):
    ...                                       # meta["error"] has the message
```

`text` is written for the model and starts with `ERROR: ` on failure; code
should test `meta.get("ok", True)` rather than that prefix. To publish more
than the default four, read their descriptions straight from the docstrings
instead of keeping a second copy:

```python
from agentknit import extract_tool_specs_from_module, tool_spec_to_schema, tool_library

specs = extract_tool_specs_from_module(tool_library)
schema = [tool_spec_to_schema(spec) for spec in specs.values()]
tool_dispatch = {
    spec["name"]: {"python_function": fn_name, "param_map": spec["param_map"]}
    for fn_name, spec in specs.items()
}
```

One thing to set up front: `exec_shell` and `search_files` echo their
subprocess output live, to stdout by default. If your host speaks a protocol
on stdout, move that stream before the first tool call:

```python
import sys
from agentknit import set_tool_output_stream

set_tool_output_stream(sys.stderr)
```

A worked example is [agentknit-over-mcp](https://github.com/monperrus/agentknit-over-mcp),
which publishes this runtime as an MCP server.

### Runtime tool management (`/tool`)

In the REPL the toolset is not fixed at startup — `/tool` lists and mutates it:

```
/tool                    # or /tool list — active ✓, inactive ✗, available
/tool activate <name>    # add a tool back (removed, library function or alias)
/tool remove <name>      # drop a tool for the rest of the session
```

`/tool remove` also retires the tool's dispatch entry, so the model can no
longer call a tool it cannot see; the spec is parked and restored verbatim by
`/tool activate`. `activate` also brings in any `TOOL_LIBRARY` function (e.g.
`/tool activate t_glob` — schema inferred from the signature when no
`Tool spec:` docstring exists) and resolves legacy aliases
(`execute_shell_command` → `exec_shell`). Interactive tools such as
`t_ask_user` are hidden in `--non-interactive` sessions.

### Background shell tools (`nohup` / `nohup_query` / `nohup_wait`)

`agentknit.async_toolkit` provides bounded background execution on top of the
low-level `t_execute_async` / `t_query_exec` primitives: stdout/stderr are
captured to files, stdin is exposed as a FIFO, and execution is wrapped in
`timeout(1)`. The tool definitions and implementations both live there — add
them to a spec with one call:

```python
import agentknit
from agentknit.async_toolkit import enable_nohup

schema = agentknit.load_specification(MODEL, ENDPOINT)
enable_nohup(schema)   # appends nohup + nohup_query + nohup_wait specs and dispatch entries
```

It is idempotent and supports both schema shapes (`tools` list or pre-built
`tool_dispatch`). The default bound is 10 minutes (`NOHUP_TIMEOUT_MIN`).

Async tools always come with three:

| Tool | Role |
|---|---|
| `nohup` | Start a shell command in the background, bounded by `timeout` minutes (default 10). Returns `tool_exec_id`, the `pid`, and the local stdin (FIFO)/stdout/stderr paths. `wait_before_s` starts the command N seconds later (the `timeout` budget counts from the start, not from the delay). |
| `nohup_query` | Poll one execution by `tool_exec_id`; inlines stdout/stderr when both fit in 4 KiB, otherwise reports file sizes. |
| `nohup_wait` | `nohup_wait(tool_exec_id, howmuch, unit)` waits for **one** background execution instead of busy-polling; `howmuch` × `unit` (`s`/`m`/`h`/`d`) is an optional budget. Returns as soon as the exec completes (returncode, output paths, last lines of stdout/stderr, inline output when small). If the budget expires first, it reports `completed: false` plus the CPU and I/O activity of the still-running process so you can tell progress from a hang. Execs finishing meanwhile are listed under `also_completed`. |

`nohup_wait` is the tool to call right after `nohup`: it blocks until the given
`tool_exec_id` finishes (or the optional `howmuch` budget expires) and returns
the result, so short commands need no `nohup_query` round trip. Waits are
capped at `WAIT_FOR_MAX_SECONDS` (3600) per call; split longer waits into
several calls.

To run a command later — e.g. re-check CI in 5 minutes — pass
`wait_before_s=300`: the `nohup` call returns immediately with the
`tool_exec_id`, the command starts after the delay, the `timeout(1)` budget
applies only once it runs, and `nohup_wait` reports it as usual.

A long `nohup_wait` never freezes the conversation. In the REPL, typing a line
while the agent is waiting cuts the wait short: the call comes back with
`completed: false` and `interrupted_by: "user_input"`, the model ends its turn,
and your message runs as the next turn while the command keeps running in the
background. When that command finishes, the model is pinged — the completion is
delivered as a `[background]` notice, either prepended to your next message or,
if you are idle at the prompt, as a turn of its own. Programmatic embedders get
the same two halves via `async_toolkit.wait_interrupt_hook` (a predicate that
cuts waits short) and `async_toolkit.drain_completions()` /
`completion_notice()` (what to feed the model afterwards).

Two consecutive `nohup_query` calls for the same still-running
`tool_exec_id` get a response pointing at
`nohup_wait(tool_exec_id, howmuch, unit)`, which returns as soon as the
execution finishes. The redirect lifts as
soon as anything else happens — another exec is polled, a new command is
started, or the execution completes.

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
| `reasoning_stream_end` | End of a streaming reasoning sequence; emitted *before* `content_stream_end` when both streamed, since reasoning precedes content in the SSE stream | `no_newline`, `fmt` |
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
| `rate_limit_wait` | Before sleeping through a retryable HTTP 429 | `delay`, `resume_at`, `fmt` |

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

## Hooks (Claude Code / Codex compatible)

agentknit implements the **common core of the Claude Code and Codex CLI hook
conventions**: same `hooks.json` config shape, same matcher semantics, same
JSON-on-stdin input, same exit-code + JSON-on-stdout output contract. A
`hooks.json` written for either tool works unmodified — drop it in
`~/.agentknit/hooks.json` (user), `<git-root>/.agentknit/hooks.json`
(project), the spec's `behaviour.hooks`, the `hooks=` kwarg of
`init_session`/`run_task`/`run`/`run_agent`, or the CLI's `--hooks PATH`
(repeatable; layers merge additively).

```json
{"hooks": {"PreToolUse": [{"matcher": "Bash",
  "hooks": [{"type": "command", "command": "~/.local/bin/guard.sh",
             "timeout": 30}]}]}}
```

### Hooks directory — presence is the registration

No JSON is required at all. A `hooks/` directory next to `hooks.json` in either
layer (`~/.agentknit/hooks/`, `<git-root>/.agentknit/hooks/`) is scanned, and an
executable script found there *is* a registered hook — there is nothing else to
keep in sync:

```
~/.agentknit/hooks/notify-stop.py           # event inferred from the name → Stop
~/.agentknit/hooks/Stop/notify.py           # explicit event directory
~/.agentknit/hooks/PreToolUse/Bash/guard.sh # … with a matcher
```

The event comes from an exact (case- and separator-insensitive) name match, else
the longest event name the file name ends with — `notify-subagent-stop.py` is a
`SubagentStop` hook, not a `Stop` one. Scripts run in exec form (no shell), so a
path with spaces is fine. Hidden files, `*.disabled` and `__pycache__` are
ignored; a file whose event cannot be inferred, or that is not executable, is
reported as a startup warning and **not** registered, so it can never fail at
dispatch time with exit 127. Directory and `hooks.json` layers merge additively,
and a script named by both is registered once: two command hooks on the same
event and matcher whose commands resolve to the same file are one hook (shell
one-liners and Python hooks are never deduplicated).

Any path passed as a hooks source — `--hooks PATH`, the `hooks=` kwarg — is
scanned this way when it is a directory, or parsed as JSON when it is a file.
The scan is available on its own too:

```python
from agentknit import discover_hook_dir

entries, warnings = discover_hook_dir("~/.agentknit/hooks")
```

Events fired: `SessionStart`, `SessionEnd`, `UserPromptSubmit`,
`PreToolUse`, `PostToolUse`, `Stop`, `PreCompact`, `PostCompact`,
`Interrupt`. Claude-only events (`PermissionRequest`, `SubagentStart`,
`SubagentStop`, `Notification`) load without errors but never fire until the
matching lifecycle point exists. Tool names are reported with their
Claude-canonical aliases (`exec_shell`→`Bash`, `str_replace`→`Edit`,
`read_file`→`Read`, …) and matchers are tested against both names;
`updatedInput` accepts Claude argument names (`file_path`, `old_string`,
…) and translates them to agentknit's.

Output contract (both ecosystems): exit 0 silent = no decision; exit 0 +
JSON on stdout = structured control (`permissionDecision` allow/deny/ask,
`updatedInput`, `additionalContext`, `decision: "block"` + `reason`,
`continue: false`, `systemMessage`); **exit 2 = blocking** with stderr as
the reason; any other exit code, timeout, or invalid output = non-blocking
error and the operation proceeds (fail open). `PreToolUse` deny/exit-2
blocks the call (the reason becomes the tool result); `"ask"` pauses for
user confirmation in the REPL and degrades to deny under `--non-interactive`;
`PostToolUse` block replaces the tool result; `Stop` block continues the
turn with the reason as a new user message (guarded by `stop_hook_active`);
`UserPromptSubmit` block rejects the prompt. Model-visible hook text is
capped at 10,000 chars with spill-to-disk (`additionalContextLimit`
configurable per handler).

**Strict script ≡ Python symmetry**: both front-ends go through one
normalizer, so a hook script and the same logic as a Python function are
proven equivalent by tests. Register Python hooks with:

```python
from agentknit import register_hook, load_hooks, HookBlock

register_hook(session, "PreToolUse",
              lambda p: {"hookSpecificOutput": {
                  "hookEventName": "PreToolUse",
                  "permissionDecision": "deny",
                  "permissionDecisionReason": "read-only session"}},
              matcher="Bash")

def stop_hook(payload):
    raise HookBlock("run the tests first")   # exit-2 equivalent

register_hook(session, "Stop", stop_hook)
```

A Python hook returns `None` (silent), a `dict` (JSON outcome) or a `str`
(plain stdout); raising `HookBlock` is exit 2; any other exception is a
non-blocking error. Hooks are the **control plane**; the `subscribe` event
system remains the **observation plane** (hooks run first and may
rewrite/block; the `tool_call` event then fires with post-rewrite args).
New events: `hook_warning` (systemMessage + errors), `hook_status`
(`statusMessage` while a hook runs). `/hooks` in the REPL lists configured
hooks and their sources; `--no-hooks` / `hooks_enabled: false` disables
everything.

Deliberate deviations from the upstream defaults: handler timeout defaults
to 60 s (not Codex's 600 s), no trust/hash review flow (agentknit is a
library — trust belongs to the embedder), and `mcp_tool`/`prompt`/`agent`
handler types are parsed but skipped with a warning.

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

### Explicit session directories and durable sinks

Pass `session_dir` to keep one session's journal, snapshot, and event log in
an explicit folder. This enables complete lifecycle capture: initial prompts,
messages, model requests and responses, streaming frames, tool boundaries, and
runtime events are committed before they are sent to a model, tool, renderer,
or subscriber.

```python
from agentknit import run_task

result = run_task(schema, task, session_dir="/srv/agent-sessions/job-42")
```

`durable_sink` accepts an object with a synchronous `append(record)` method.
It receives the same ordered records after the built-in filesystem journal has
been fsync'd and before downstream consumers run. Raising from `append` stops
the producing operation, so queueing a background write does not satisfy the
contract.

```python
class AuditSink:
    def append(self, record: dict) -> None:
        store_and_fsync(record)

result = run_task(schema, task,
                  session_dir="/srv/agent-sessions/job-42",
                  durable_sink=AuditSink())
```

A `journal_recovered` event is emitted whenever a resume rebuilt state from
the journal.

## Resuming sessions

A session is resumed with ``--session <id>`` (CLI) or ``session_id=`` (SDK).
Resume always continues on the endpoint the session was **created** on, not on
whatever the current `--endpoint` / CLI default resolves to: the endpoint is
recovered from the session's append-only logs (the earliest `session_start`
record), which also protects against a snapshot overwritten by an earlier
resume against a different provider. The recorded key source
(`keyring_service`+`keyring_username` or `key_env`) is restored with it, and
key sources the resumed session never used are dropped. When the resolved
endpoint differs from the requested one, a one-line notice is printed.

### Resuming with another provider/model

`agentknit <other-model> --session <id>` deliberately switches provider. The
session file is **copied** into the new model's directory and re-stamped —
`metadata.model` / `endpoint` / `auth` describe the new provider, and the old
one is kept in `metadata.ported_from` — so the transcript continues on the
endpoint/key of the model you named, and a later resume of that session binds
to the new provider. The original file is left untouched as the record of
where the history came from.

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
