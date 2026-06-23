"""agentknit — spec-driven coding agent framework for any OpenAI-compatible endpoint."""

__version__ = "0.1.0"

from .exceptions import (
    AgentProbeError,
    AgentSpecDisabledError,
    AgentSpecInvalidError,
    PricingLimitExceededError,
    AuthenticationError,
)

from ._core import (
    main,
    validate_schema,
    create_client,
    run_task,
    run_repl,
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
    _parse_run_uri,
    _expand_aliases,
    _open_log,
    _save_messages_snapshot,
    _load_messages_snapshot,
    _find_snapshot_in_other_models,
    _handle_tool_call,
    _complete,
)

from .tool import (
    Tool,
    build_tool_spec,
    register_tools_in_library,
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
    "AgentProbeError", "AgentSpecDisabledError", "AgentSpecInvalidError",
    "PricingLimitExceededError", "AuthenticationError",
    "main", "validate_schema", "create_client", "run_task", "run_repl",
    "SessionResult", "CancelToken", "EventCallback", "_default_event_handler",
    "subscribe", "unsubscribe", "on",
    "run_turn", "init_session", "dispatch", "load_or_probe",
    "check_and_display_pricing", "extract_inline_calls", "schema_props",
    "fmt_usage", "fmt_call", "fmt_result", "safe_model_name",
    "inline_system_prompt", "FatalToolDispatchError",
    "LOG_BASE", "DEFAULT_ENDPOINT", "DEFAULT_MAX_TOKENS",
    "_parse_run_uri", "_expand_aliases", "_open_log",
    "_save_messages_snapshot", "_load_messages_snapshot",
    "_find_snapshot_in_other_models", "_handle_tool_call", "_complete",
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
