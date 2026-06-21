"""
Tool dataclass and helper to eliminate schema/dispatch boilerplate.

Provides :class:`Tool` — a declarative way to define a tool — and
:func:`build_tool_spec` which converts a list of :class:`Tool` objects into
the OpenAI-compatible JSON schema list and dispatch dict that the agent loop
expects.

Typical usage::

    from agentknit import Tool, build_tool_spec

    tools = [
        Tool("read_file", "Read a file", t_read,
             parameters={"path": {"type": "string", "description": "Path to the file."}}),
        Tool("write_file", "Write a file", t_write,
             parameters={"path": {"type": "string"},
                         "content": {"type": "string"}}),
    ]
    schema, dispatch = build_tool_spec(tools)
"""

from __future__ import annotations

import inspect
import typing
from dataclasses import dataclass
from typing import Callable


@dataclass
class Tool:
    """Declarative tool definition.

    Fields
    ------
    name
        The tool name as the model will see it (e.g. ``"read_file"``).
    description
        A human-readable description of what the tool does.
    fn
        The Python callable that implements the tool. It must return
        ``(str, dict)`` — the text result and a metadata dict with at least
        a ``"result"`` key.
    parameters
        The JSON Schema object describing the tool's parameters, *or* ``None``
        to infer the schema from *fn*'s signature.  When inferred, parameter
        types are mapped as ``str → "string"``, ``int → "integer"``,
        ``float → "number"``, ``bool → "boolean"`` — all others default to
        ``"string"``.
    param_map
        An optional mapping from model-facing argument names to the Python
        keyword argument names of *fn*.  When ``None`` (the default) the
        identity mapping is used (model names match Python names exactly).
    """

    name: str
    description: str
    fn: Callable
    parameters: dict | None = None
    param_map: dict | None = None

    @property
    def python_function_name(self) -> str:
        """A safe identifier derived from *fn* for use in ``TOOL_LIBRARY``."""
        return self.fn.__name__

    @property
    def resolved_parameters(self) -> dict:
        """Return *parameters* if set, otherwise infer from *fn*'s signature."""
        if self.parameters is not None:
            return self.parameters
        return _infer_parameters(self.fn)

    @property
    def resolved_param_map(self) -> dict:
        """Return *param_map* if set, otherwise the identity mapping."""
        if self.param_map is not None:
            return self.param_map
        # Identity: each parameter name maps to itself.
        params = self.resolved_parameters
        props = params.get("properties", {})
        return {k: k for k in props}


def _infer_parameters(fn: Callable) -> dict:
    """Infer a JSON Schema ``parameters`` object from *fn*'s signature.

    Only positional-or-keyword parameters are included (``*args``, ``**kwargs``
    and ``self``/``cls`` are skipped).
    """
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        # Cannot inspect (e.g. built-in) → return an empty schema.
        return {"type": "object", "properties": {}, "required": []}

    # Resolve PEP 563 (from __future__ import annotations) string annotations
    # to actual types.  Falls back to param.annotation on failure (e.g. for
    # functions defined in test fixtures where the annotation may be a string
    # that cannot be resolved via get_type_hints).
    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        hints = {}

    _TYPE_MAP: dict[object, str] = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }

    properties: dict[str, dict] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        # Skip self, cls, *args, **kwargs
        if name in ("self", "cls"):
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        # Resolve the type — prefer get_type_hints, fall back to annotation.
        raw_type = hints.get(name, param.annotation)
        # If raw_type is still a string (e.g. 'int' without import), map it.
        if isinstance(raw_type, str):
            _STR_TYPE_MAP: dict[str, str] = {
                "str": "string",
                "int": "integer",
                "float": "number",
                "bool": "boolean",
            }
            json_type = _STR_TYPE_MAP.get(raw_type, raw_type)
        else:
            json_type = _TYPE_MAP.get(raw_type, "string")

        prop: dict[str, object] = {"type": json_type}

        properties[name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def build_tool_spec(
    tools: list[Tool],
) -> tuple[list[dict], dict[str, dict]]:
    """Convert a list of :class:`Tool` objects into the schema + dispatch pair.

    Parameters
    ----------
    tools
        One or more :class:`Tool` instances.

    Returns
    -------
    schema
        OpenAI-compatible tool schema list (suitable for the ``"tools"`` key in
        a chat completion request).
    dispatch
        Dispatch dict mapping tool name → ``{"python_function": …, "param_map": …}``
        (suitable for the ``"tool_dispatch"`` key in an agent spec).

    The returned dispatch dict uses the string ``fn.__name__`` as the
    ``python_function`` value.  If you use :func:`build_tool_spec` together
    with the rest of agentknit, you must ensure the callable has been
    registered in :data:`~agentknit.tool_library.TOOL_LIBRARY` under that name
    (or use :func:`register_tools_in_library` for convenience).

    Example
    -------
    >>> schema, dispatch = build_tool_spec([
    ...     Tool("read_file", "Read a file", t_read,
    ...          parameters={"path": {"type": "string"}}),
    ... ])
    >>> schema[0]["function"]["name"]
    'read_file'
    >>> dispatch["read_file"]["python_function"]
    't_read'
    """
    schema: list[dict] = []
    dispatch: dict[str, dict] = {}

    for tool in tools:
        # ── schema entry ─────────────────────────────────────────────────
        params = tool.resolved_parameters
        schema.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": params,
            },
        })

        # ── dispatch entry ───────────────────────────────────────────────
        dispatch[tool.name] = {
            "python_function": tool.python_function_name,
            "param_map": tool.resolved_param_map,
        }

    return schema, dispatch


def register_tools_in_library(tools: list[Tool]) -> None:
    """Register each tool's callable in :data:`agentknit.tool_library.TOOL_LIBRARY`.

    The key used is ``fn.__name__`` (which is also the value written to
    ``dispatch["python_function"]`` by :func:`build_tool_spec`).

    This is a convenience so callers do not have to manually maintain the
    library dict.
    """
    from .tool_library import TOOL_LIBRARY

    for tool in tools:
        TOOL_LIBRARY[tool.python_function_name] = tool.fn
