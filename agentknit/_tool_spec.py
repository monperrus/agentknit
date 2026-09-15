"""Extract tool specifications from docstrings at runtime."""

from __future__ import annotations

import re
from typing import Any


def parse_tool_spec_from_docstring(doc: str) -> dict[str, Any] | None:
    """Parse a YAML-like tool spec block from a function's docstring.

    The block must start with ``Tool spec:`` on its own line, followed by
    indented YAML-like key-value pairs.

    Returns a dict with keys ``name``, ``description``, ``parameters``
    (a dict of parameter name → ``{"type": str, "description": str}``) and
    ``required`` (the parameter names marked ``required: true``, in declaration
    order), or ``None`` if no ``Tool spec:`` block is found.
    """
    if not doc:
        return None

    # Find the "Tool spec:" marker
    lines = doc.splitlines()
    spec_start = None
    for i, line in enumerate(lines):
        if line.strip() == "Tool spec:":
            spec_start = i
            break

    if spec_start is None:
        return None

    spec_lines = lines[spec_start + 1:]

    result: dict[str, Any] = {
        "name": "",
        "description": "",
        "parameters": {},
        "required": [],
    }

    # Determine base indent from first non-empty line
    base_indent = ""
    for line in spec_lines:
        stripped = line.strip()
        if stripped:
            base_indent = line[:len(line) - len(line.lstrip())]
            break

    # Determine parameter indent (one level deeper than base)
    param_indent = None
    for line in spec_lines:
        stripped = line.strip()
        if stripped and line.startswith(base_indent):
            content = line[len(base_indent):]
            if content and not content.startswith(" "):
                continue
            if content.strip() and content[0] == " ":
                # This is indented further than base
                inner = line[:len(line) - len(line.lstrip())]
                if len(inner) > len(base_indent):
                    param_indent = inner
                    break

    current_param: str | None = None
    in_params = False

    for line in spec_lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Check if we're still inside the spec block (same or greater indent)
        if line and not line.startswith(base_indent) and line.rstrip() != "":
            break

        # Remove base indent
        content = line[len(base_indent):] if line.startswith(base_indent) else line.strip()

        if content == "parameters:":
            in_params = True
            continue

        if in_params:
            # Determine the relative indent of this line within parameters
            if param_indent and line.startswith(param_indent):
                relative = line[len(param_indent):]
            else:
                relative = content.lstrip()

            stripped_relative = relative.lstrip()

            # Check for a parameter name line (e.g. "path:" at param_indent level)
            param_match = re.match(r"^(\w+):$", stripped_relative)
            if param_match:
                current_param = param_match.group(1)
                result["parameters"][current_param] = {}
                continue

            # Sub-keys like "type:" or "description:" (further indented)
            if current_param:
                sub_match = re.match(r"^(\w+):\s*(.*)", stripped_relative)
                if sub_match:
                    key = sub_match.group(1)
                    value = sub_match.group(2).strip()
                    if key == "type":
                        result["parameters"][current_param]["type"] = value
                    elif key == "description":
                        result["parameters"][current_param]["description"] = value
                    elif key == "required":
                        if value.strip().lower() in ("true", "yes", "1"):
                            if current_param not in result["required"]:
                                result["required"].append(current_param)
        else:
            # Top-level keys
            top_match = re.match(r"^(\w+):\s*(.*)", content)
            if top_match:
                key = top_match.group(1)
                value = top_match.group(2).strip()
                if key == "name":
                    result["name"] = value
                elif key == "description":
                    result["description"] = value

    return result


def extract_tool_specs_from_module(module: object) -> dict[str, dict[str, Any]]:
    """Extract tool specs from all functions in a module that have a ``Tool spec:`` block.

    Returns a dict mapping function name → parsed spec dict.
    """
    import inspect

    specs: dict[str, dict[str, Any]] = {}
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        doc = inspect.getdoc(obj)
        spec = parse_tool_spec_from_docstring(doc or "")
        if spec is not None:
            specs[name] = spec
    return specs


def tool_spec_to_schema(spec: dict[str, Any]) -> dict[str, Any]:
    """Convert a parsed ``Tool spec:`` block into one OpenAI schema entry.

    The result is what an endpoint's ``"tools"`` list expects, and what
    :func:`~agentknit.tool.build_tool_spec` produces for a :class:`~agentknit.Tool`::

        >>> spec = parse_tool_spec_from_docstring('''
        ...     Tool spec:
        ...         name: glob
        ...         description: Match paths.
        ...         parameters:
        ...             pattern:
        ...                 type: string
        ...                 description: Glob pattern.
        ...                 required: true
        ... ''')
        >>> tool_spec_to_schema(spec)["function"]["parameters"]["required"]
        ['pattern']

    Together with :func:`extract_tool_specs_from_module`, this turns a module of
    documented tool functions into a schema a host can publish, with no second
    copy of the parameter descriptions to keep in sync.
    """
    properties = {
        name: {k: v for k, v in param.items() if k != "required"}
        for name, param in spec.get("parameters", {}).items()
    }
    return {
        "type": "function",
        "function": {
            "name": spec.get("name", ""),
            "description": spec.get("description", ""),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(spec.get("required", [])),
            },
        },
    }