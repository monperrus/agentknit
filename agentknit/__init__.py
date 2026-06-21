"""agentknit — spec-driven coding agent framework for any OpenAI-compatible endpoint."""

__version__ = "0.1.0"

from .exceptions import (
    AgentSpecDisabledError,
    AgentSpecInvalidError,
    PricingLimitExceededError,
    AuthenticationError,
)
from llmprobe.exceptions import AgentProbeError

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

__all__ = [
    "AgentProbeError", "AgentSpecDisabledError", "AgentSpecInvalidError",
    "PricingLimitExceededError", "AuthenticationError",
    "main", "validate_schema", "create_client", "run_task", "run_repl",
    "SessionResult", "CancelToken", "EventCallback", "_default_event_handler",
    "run_turn", "init_session", "dispatch", "load_or_probe",
    "check_and_display_pricing", "extract_inline_calls", "schema_props",
    "fmt_usage", "fmt_call", "fmt_result", "safe_model_name",
    "inline_system_prompt", "FatalToolDispatchError",
    "LOG_BASE", "DEFAULT_ENDPOINT", "DEFAULT_MAX_TOKENS",
    "_parse_run_uri", "_expand_aliases", "_open_log",
    "_save_messages_snapshot", "_load_messages_snapshot",
    "_find_snapshot_in_other_models", "_handle_tool_call", "_complete",
]
