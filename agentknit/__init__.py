"""agentknit — spec-driven coding agent framework for any OpenAI-compatible endpoint."""

__version__ = "0.1.0"

from .exceptions import (
    AgentProbeError,
    AgentSpecDisabledError,
    AgentSpecInvalidError,
    PricingLimitExceededError,
    AuthenticationError,
    CacheProofError,
    ContextWindowExceededError,
    RateLimitError,
)

from ._core import (
    main,
    run,
    validate_schema,
    create_client,
    run_task,
    run_agent,
    run_repl,
    run_async_repl,
    SessionResult,
    Session,
    CancelToken,
    EventCallback,
    _default_event_handler,
    enable_osc8_hyperlinks,
    subscribe,
    unsubscribe,
    on,
    run_turn,
    init_session,
    dispatch,
    load_specification,
    check_and_display_pricing,
    extract_inline_calls,
    schema_props,
    fmt_usage,
    fmt_call,
    fmt_result,
    safe_model_name,
    inline_system_prompt,
    FatalToolDispatchError,
    LOG_BASE,
    DEFAULT_ENDPOINT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_COMPACTION_TRIGGER_TOKENS,
    DEFAULT_COMPACTION_TARGET_TOKENS,
    DEFAULT_COMPACTION_KEEP_LAST_TURNS,
    _parse_run_uri,
    _expand_aliases,
    _open_log,
    _save_messages_snapshot,
    _load_messages_snapshot,
    _find_snapshot_in_other_models,
    _load_snapshot_metadata,
    _bind_schema_to_resumed_session,
    _port_snapshot_to_model,
    _handle_tool_call,
    _complete,
    compact_session,
    _compact_session,
    _maybe_compact,
    _is_context_window_error,
    _journal_path,
)

from ._journal import (
    DurableSink,
    SessionJournal,
    JournalState,
    PendingToolCall,
    KnownToolResult,
    new_call_id,
    replay_journal,
)

from .hooks import (
    HookBlock,
    HookDecision,
    HookEntry,
    HookHandler,
    RawHookResult,
    dedupe_entries,
    discover_hook_dir,
    load_hooks,
    register_hook,
)

from .tool import (
    Tool,
    build_tool_spec,
    default_tool_spec,
    register_tools_in_library,
)

from .sandbox import (
    ToolExecutor,
    ToolSessionContext,
    LocalToolExecutor,
    SandboxPolicy,
    BubblewrapToolExecutor,
)

from ._tool_spec import (
    parse_tool_spec_from_docstring,
    extract_tool_specs_from_module,
    tool_spec_to_schema,
)

from .tool_library import (
    t_execute_async,
    t_query_exec,
    ASYNC_EXEC_DIR,
    ASYNC_FAST_THRESHOLD_S,
    ASYNC_INLINE_MAX_BYTES,
    async_completion_queue,
    enable_rtk_rewrite,
    get_tool_output_stream,
    set_tool_output_stream,
    CANONICAL_TOOLS,
    SUPERSEDED_TOOLS,
)

from .async_toolkit import (
    NOHUP_TIMEOUT_MIN,
    WAIT_FOR_MAX_SECONDS,
    WAIT_FOR_UNIT_SECONDS,
    enable_nohup,
    nohup_tool_specs,
    t_nohup,
    t_nohup_wait,
)

from .slash_commands import (
    SlashCommand,
    SlashCommandRegistry,
    REGISTRY as slash_registry,
    t_slash_command,
    slash_tool_ctx,
    SLASH_COMMAND_TOOL,
)

__all__ = [
    "parse_tool_spec_from_docstring",
    "extract_tool_specs_from_module",
    "tool_spec_to_schema",
    "AgentProbeError", "AgentSpecDisabledError", "AgentSpecInvalidError",
    "PricingLimitExceededError", "AuthenticationError", "CacheProofError",
    "ContextWindowExceededError", "RateLimitError",
    "main", "run",
    "validate_schema", "create_client", "run_task", "run_agent", "run_repl", "run_async_repl",
    "SessionResult", "Session", "CancelToken", "EventCallback", "_default_event_handler",
    "enable_osc8_hyperlinks",
    "subscribe", "unsubscribe", "on",
    "run_turn", "init_session", "dispatch", "load_specification",
    "check_and_display_pricing", "extract_inline_calls", "schema_props",
    "fmt_usage", "fmt_call", "fmt_result", "safe_model_name",
    "inline_system_prompt", "FatalToolDispatchError",
    "LOG_BASE", "DEFAULT_ENDPOINT", "DEFAULT_MAX_TOKENS",
    "DEFAULT_COMPACTION_TRIGGER_TOKENS", "DEFAULT_COMPACTION_TARGET_TOKENS",
    "DEFAULT_COMPACTION_KEEP_LAST_TURNS",
    "_parse_run_uri", "_expand_aliases", "_open_log",
    "_save_messages_snapshot", "_load_messages_snapshot",
    "_find_snapshot_in_other_models", "_load_snapshot_metadata",
    "_bind_schema_to_resumed_session", "_port_snapshot_to_model",
    "_handle_tool_call", "_complete",
    "compact_session", "_compact_session", "_maybe_compact",
    "_is_context_window_error",
    "_journal_path",
    # durable recovery
    "DurableSink", "SessionJournal", "JournalState", "PendingToolCall", "KnownToolResult",
    "new_call_id", "replay_journal",
    # hooks (Claude Code / Codex-compatible)
    "HookBlock", "HookDecision", "HookEntry", "HookHandler",
    "RawHookResult", "load_hooks", "register_hook", "discover_hook_dir", "dedupe_entries",
    "Tool",
    "build_tool_spec",
    "default_tool_spec",
    "register_tools_in_library",
    "ToolExecutor", "ToolSessionContext", "LocalToolExecutor",
    "SandboxPolicy", "BubblewrapToolExecutor",
    "SlashCommand",
    "SlashCommandRegistry",
    "slash_registry",
    "t_slash_command",
    "slash_tool_ctx",
    "SLASH_COMMAND_TOOL",
    # async shell tools
    "t_execute_async", "t_query_exec", "async_completion_queue",
    "ASYNC_EXEC_DIR", "ASYNC_FAST_THRESHOLD_S", "ASYNC_INLINE_MAX_BYTES",
    "enable_rtk_rewrite",
    # where live tool output goes
    "get_tool_output_stream", "set_tool_output_stream",
    # which tool implementation to use
    "CANONICAL_TOOLS", "SUPERSEDED_TOOLS",
    # nohup tools (async_toolkit)
    "NOHUP_TIMEOUT_MIN", "WAIT_FOR_MAX_SECONDS", "WAIT_FOR_UNIT_SECONDS",
    "enable_nohup", "nohup_tool_specs", "t_nohup", "t_nohup_wait",
]
