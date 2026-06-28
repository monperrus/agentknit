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
