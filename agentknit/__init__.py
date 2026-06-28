"""agentknit — spec-driven coding agent framework for any OpenAI-compatible endpoint."""

__version__ = "0.1.0"

from .exceptions import (
    AgentProbeError,
    AgentSpecDisabledError,
    AgentSpecInvalidError,
    PricingLimitExceededError,
    AuthenticationError,
    CacheProofError,
)

from ._core import (
    main,
    run,
    validate_schema,
    create_client,
    run_task,
    run_repl,
    run_async_repl,
    SessionResult,
    CancelToken,
    EventCallback,
    _default_event_handler,
    subscribe,
    unsubscribe,
    on,
    run_turn,
    init_session,
    dispatch,
    load_or_probe,
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
    _handle_tool_call,
    _complete,
    _compact_session,
    _maybe_compact,
)

from .tool import (
    Tool,
    build_tool_spec,
    register_tools_in_library,
)

from ._tool_spec import (
    parse_tool_spec_from_docstring,
    extract_tool_specs_from_module,
)

from .tool_library import (
    t_execute_async,
    t_query_exec,
    ASYNC_EXEC_DIR,
    ASYNC_FAST_THRESHOLD_S,
    ASYNC_INLINE_MAX_BYTES,
    async_completion_queue,
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
    "AgentProbeError", "AgentSpecDisabledError", "AgentSpecInvalidError",
    "PricingLimitExceededError", "AuthenticationError", "CacheProofError",
    "main", "run",
    "validate_schema", "create_client", "run_task", "run_repl", "run_async_repl",
    "SessionResult", "CancelToken", "EventCallback", "_default_event_handler",
    "subscribe", "unsubscribe", "on",
    "run_turn", "init_session", "dispatch", "load_or_probe",
    "check_and_display_pricing", "extract_inline_calls", "schema_props",
    "fmt_usage", "fmt_call", "fmt_result", "safe_model_name",
    "inline_system_prompt", "FatalToolDispatchError",
    "LOG_BASE", "DEFAULT_ENDPOINT", "DEFAULT_MAX_TOKENS",
    "DEFAULT_COMPACTION_TRIGGER_TOKENS", "DEFAULT_COMPACTION_TARGET_TOKENS",
    "DEFAULT_COMPACTION_KEEP_LAST_TURNS",
    "_parse_run_uri", "_expand_aliases", "_open_log",
    "_save_messages_snapshot", "_load_messages_snapshot",
    "_find_snapshot_in_other_models", "_handle_tool_call", "_complete",
    "_compact_session", "_maybe_compact",
    "Tool",
    "build_tool_spec",
    "register_tools_in_library",
    "SlashCommand",
    "SlashCommandRegistry",
    "slash_registry",
    "t_slash_command",
    "slash_tool_ctx",
    "SLASH_COMMAND_TOOL",
    # async shell tools
    "t_execute_async", "t_query_exec", "async_completion_queue",
    "ASYNC_EXEC_DIR", "ASYNC_FAST_THRESHOLD_S", "ASYNC_INLINE_MAX_BYTES",
]
