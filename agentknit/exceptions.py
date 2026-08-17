"""Typed exceptions for agentknit."""


class AgentProbeError(Exception):
    """Base class for all agentknit exceptions."""


class AuthenticationError(AgentProbeError):
    """Raised when an API key or auth token cannot be obtained."""


class AgentSpecDisabledError(AgentProbeError):
    """Raised when an agent spec has ``disabled: true`` set."""

    def __init__(self, message: str, *, comment: str = "") -> None:
        super().__init__(message)
        self.comment = comment


class AgentSpecInvalidError(AgentProbeError):
    """Raised when a spec is missing required fields (e.g. inferred_tool_schema)."""

    def __init__(self, message: str, *, model: str = "") -> None:
        super().__init__(message)
        self.model = model


class PricingLimitExceededError(AgentProbeError):
    """Raised when the live model price exceeds the schema-defined limit."""

    def __init__(self, message: str, *, model: str, direction: str,
                 current_price: float, limit: float) -> None:
        super().__init__(message)
        self.model         = model
        self.direction     = direction
        self.current_price = current_price
        self.limit         = limit


class CacheProofError(AgentProbeError):
    """Raised when strict cache-proof mode does not observe a cache hit."""


class RateLimitError(AgentProbeError):
    """Raised when the server returns HTTP 429 with no usable retry delay.

    A retry is only attempted automatically when the response tells us
    *when* it's safe to retry (``Retry-After``, ``retry-after-ms``, or
    ``x-ratelimit-reset-requests``). Without one of those headers we have
    no basis for guessing a delay, so the call is aborted instead of
    looping forever.
    """

    def __init__(self, message: str, *, status_code: int = 429,
                 headers: dict | None = None, error_code: str | None = None,
                 error_message: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.headers = headers or {}
        # Provider business-error details, when the response exposes them.
        self.error_code = error_code
        self.error_message = error_message
