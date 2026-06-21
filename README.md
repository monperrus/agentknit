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
