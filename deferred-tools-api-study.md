# Deferred tool loading in agentknit: what the API needs

Prototype branch `proto/deferred-tools`. It implements two provider-native mechanisms behind one switch:

- **`tool_loading="kimi"`:** a found tool is revealed by appending `{"role":"system","tools":[…]}` (Kimi Chat Completions, `k3`).
- **`tool_loading="glm"`:** tools are declared with `function.defer_loading`, and the `search_tools` result is a list of `[{"type":"tool_reference","name":…}]` blocks (Z.ai Chat Completions, `glm-5.3`).

Background: provider survey and measurements in `deferred-tool-experiments/docs/`.

## Public API changes (all additive, default behaviour unchanged)

- **`Tool(..., deferred: bool = False)`.** `build_tool_spec` writes `"defer_loading": true` inside `function` for deferred tools.
  - This one marker works for both strategies, and a spec JSON can carry it too.
- **`init_session` / `run_task` / `run_agent`** gain:
  - `tool_loading: "eager" | "kimi" | "glm" | None` (also read from `schema["tool_loading"]`).
  - `tool_search: Callable[[query, catalog_specs], list[name]] | None`. The default is a keyword search over name + description, top 3.
- **Default (`None` / `"eager"`):** the marker is stripped and every tool is sent up front, exactly as today. Sessions without deferred tools are not touched.
- **`search_tools` is a framework tool.** It is added automatically when the mode is not eager and at least one tool is deferred. It is answered inside the agent loop, not via `Tool.fn`.

## Internal touch points (and why)

- **`_run_turn` / `_run_one`:** `search_tools` is intercepted before `_handle_tool_call`.
  - Its result is not always a string (GLM: a block list).
  - Kimi needs an **extra message** after *all* the step's tool results. Inserting it between a `tool_call` and its result would break pairing.
  - Hence a `step_extra` list, flushed after the parallel or sequential dispatch.
- **`_with_pending_ta`:** the time-awareness, token-awareness and hook text normally appended to a tool result cannot ride on a GLM reference list, because the server silently drops text blocks there. The prototype keeps it pending until the next string result.
- **`_compact_once`:**
  - *Before* the summary call (sent with no `tools`), references in the summarized prefix are rewritten to text. Measured: GLM answers HTTP 400 code 1210 otherwise.
  - *After* compaction, GLM tools revealed only by now-summarized messages are un-deferred in `tools`. That costs nothing extra, since compaction already resets the cache.
  - Kimi needs nothing: compaction already keeps every `role:system` message.
- **`_normalise_for_resume`** assumed string content and raised `TypeError` on a GLM reference list. Fixed in `812fd1c`.

## Found along the way (not specific to deferral)

- **Compaction ignored `request_params`:** it hard-coded `temperature=0`, so every compaction on Kimi `k3` failed with HTTP 400. Fixed in `728b75f`.
- **`bare=True` sends an empty system message;** Kimi rejects it (`role 'system' must not be empty`).
- **`load_specification(..., inference_db=…)` returns `tool_specs`, which wins over `inferred_tool_schema`.** A caller that injects its own tools must override `tool_specs`, which is easy to get wrong.

## Measured (live, 30-tool catalogue with 2 useful tools, 2 turns, 3 runs each, `demo_deferred_tools.py`)

- **kimi:** ✅ 3/3.
  - The model searches on its own, uses the tools, and reuses `get_weather` on turn 2 without searching again.
  - First request: 252 prompt tokens, vs 3663 eager.
- **glm:** ✅ 3/3. First request: 234 prompt tokens, vs 3822 eager.
- **With `--compact`** (whole history summarized between the turns): ✅ 3/3 on both.
  - Kimi keeps the reveal messages.
  - GLM un-defers the 6 revealed tools.
- Cached tokens keep growing across requests in deferred mode, like the eager control, so reveals do not break the prefix cache.

## Open issues before merging

1. **Mode selection belongs in inference-db.**
   - Today the caller must know that `kimi-coding` means `"kimi"` and `zai-coding` means `"glm"`.
   - A `tool_loading` quirk per entry (and per model: Kimi documents it only for `kimi-k3`) would make it automatic.
   - Unknown endpoints should fall back to a portable `"grow"` mode (append the definition to `tools`), which works everywhere but costs a cache miss.
2. **Resume is only partly covered.**
   - The catalogue is rebuilt from the schema, and the reveal messages survive in the snapshot.
   - But the GLM un-deferred set and `_revealed_tools` are not persisted. A GLM tool revealed before a compaction is hidden again after resume.
   - These two keys need to go into the snapshot.
3. **Moving a session to another provider breaks it** (`_port_snapshot_to_model`).
   - A Kimi system-tools message or a GLM reference list is invalid elsewhere: Kimi rejects `tool_reference` and GLM drops system tools.
   - Porting must rewrite reveals to the target's mechanism, or to plain eager `tools`.
4. **Step reducers see the reveal messages** as part of the step. A reducer that drops them silently hides the tools.
   - Contract: reducers must keep `role:system` messages carrying `tools`. Alternatively, reveals could be applied after the reducer.
5. **Hooks:** `search_tools` bypasses `PreToolUse`/`PostToolUse` and the durable journal's `tool_start`/`tool_end`, but still emits `tool_call`/`tool_result` events.
   - Decide whether hooks should see it.
6. **Inline mode** (non-structured tool calls) is not supported. Deferred tools stay invisible there.
7. **Events:** a `tools_revealed` event (names, mode) would let UIs show what the model loaded. The prototype only logs the search result.
8. **Search quality:** the keyword default reveals decoys (top 3), about +150 tokens each.
   - Embedding search or `limit=1` with a confidence threshold would cut that.
   - The `tool_search` hook already allows plugging either in.
