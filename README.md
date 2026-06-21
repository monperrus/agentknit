# agentknit

Spec-driven coding agent framework for any OpenAI-compatible endpoint.

Reads a JSON spec produced by [llmprobe](https://github.com/monperrus/llmprobe) and runs an interactive coding agent that dispatches tool calls (read_file, write_file, execute_bash, …) to Python implementations.

## Install

```
pip install agentknit
```

## Usage

```
agentknit qwen/qwen3-8b "list the files in /tmp"
agentknit qwen/qwen3-8b              # interactive REPL
```

### Programmatic

```python
from agentknit import load_or_probe, run_task

schema = load_or_probe("qwen/qwen3-8b", "https://openrouter.ai/api/v1", force=False)
result = run_task(schema, "List the files in /tmp")
print(result.final_reply)
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

Parameters can also be **inferred automatically** from the function signature:

```python
tools = [
    Tool("read_file", "Read a file", t_read),   # parameters inferred from def t_read(path: str)
    Tool("write_file", "Write a file", t_write), # parameters inferred from def t_write(path: str, content: str)
]
schema, dispatch = build_tool_spec(tools)
```

The `Tool` dataclass also supports `param_map` for translating model-facing
argument names to Python keyword argument names:

```python
tools = [
    Tool("read_file", "Read a file", t_read,
         param_map={"file_path": "path"}),
]
```

### Replacing the default tool definitions

The framework ships with a built-in set of tools (`read_file`, `write_file`,
`str_replace`, `execute_shell_command`).  To replace them with your own, pass
`schema` and `dispatch` to `init_session`:

```python
from agentknit import init_session, run_turn, create_client

schema_data = {
    "model": "my-model",
    "endpoint": "https://api.example.com/v1",
    "inferred_tool_schema": schema,   # from build_tool_spec
    "behaviour": {"call_delivery_mode": "structured_tool_calls"},
    "tool_dispatch": dispatch,        # from build_tool_spec
}
client = create_client(schema_data)
session = init_session(schema_data)
result = run_turn(client, "my-model", session, "Do something")
```

## API Reference

### `Tool` dataclass

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Tool name the model uses (e.g. `"read_file"`) |
| `description` | `str` | Human-readable description |
| `fn` | `Callable` | Python implementation; must return `(str, dict)` |
| `parameters` | `dict \| None` | JSON Schema object, or `None` to infer from `fn` signature |
| `param_map` | `dict \| None` | Model→Python arg name mapping, or `None` for identity |

### `build_tool_spec(tools) -> (list, dict)`

Convert a list of `Tool` objects into the OpenAI-compatible schema list and
dispatch dict.  The dispatch dict uses `fn.__name__` as the `python_function`
value — call `register_tools_in_library(tools)` to register the callables
under those names.

### `register_tools_in_library(tools)`

Register each tool's `fn` in `tool_library.TOOL_LIBRARY` keyed by
`fn.__name__`.  Required for the dispatch loop to resolve string-based
`python_function` entries.
