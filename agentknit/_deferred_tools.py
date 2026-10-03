"""Deferred tool loading: show the model a few tools, reveal the rest on demand.

A tool marked deferred (``Tool(..., deferred=True)``, or ``"defer_loading": true``
inside its ``function`` spec) is hidden from the model until it calls the
framework tool ``search_tools`` and the search returns it.  How a found tool is
revealed is provider-specific; each strategy below uses the endpoint's native
mechanism so the request prefix (``tools`` + earlier messages) never changes
and the prompt cache survives:

``kimi``  Kimi Chat Completions (``k3``): deferred tools are not sent in
          ``tools``; a reveal appends ``{"role": "system", "tools": [...]}``
          after the step's tool results.  See Kimi "Dynamically Loaded Tools".
``glm``   Z.ai GLM Chat Completions: deferred tools stay in ``tools`` with
          ``function.defer_loading``; the ``search_tools`` result is a list of
          ``{"type": "tool_reference", "name": ...}`` blocks that the server
          expands into the full definitions (GLM-5.1 chat template).
``eager`` (default) everything is sent up front, the marker is stripped.

A revealed tool stays available on later turns because the revealing message
stays in the history; compaction must therefore keep or re-create it
(:func:`before_compaction` / :func:`after_compaction`).
"""
from __future__ import annotations

import copy
import re
from typing import Any, Callable

SEARCH_TOOL_NAME = "search_tools"
MODES = ("eager", "kimi", "glm")

ToolSearch = Callable[[str, list[dict[str, Any]]], list[str]]

SEARCH_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": SEARCH_TOOL_NAME,
        "description": (
            "Search the catalog of additional tools by keywords (what you want to do, "
            "e.g. 'convert currency'). Matching tools become callable right away. "
            "Call this whenever none of your visible tools fits the task."),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Keywords describing the needed capability."}},
            "required": ["query"],
        },
    },
}


def _fn(spec: dict[str, Any]) -> dict[str, Any]:
    return spec.get("function") or spec


def tool_name(spec: dict[str, Any]) -> str:
    return str(_fn(spec).get("name", ""))


def is_deferred(spec: dict[str, Any]) -> bool:
    return bool(_fn(spec).get("defer_loading"))


def _strip_marker(spec: dict[str, Any]) -> dict[str, Any]:
    spec = copy.deepcopy(spec)
    _fn(spec).pop("defer_loading", None)
    return spec


def keyword_search(query: str, catalog: list[dict[str, Any]], limit: int = 3) -> list[str]:
    """Default search: rank catalog tools by query words found in name + description."""
    words = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 2}
    scored = []
    for spec in catalog:
        f = _fn(spec)
        haystack = f"{f.get('name', '')} {f.get('description', '')}".lower().replace("_", " ")
        score = sum(w in haystack for w in words)
        if score:
            scored.append((-score, tool_name(spec)))
    return [name for _, name in sorted(scored)[:limit]]


def prepare(session: dict[str, Any], mode: str, search: ToolSearch | None = None) -> None:
    """Rewrite ``session["tools"]`` for *mode* and record the deferred catalog."""
    if mode not in MODES:
        raise ValueError(f"tool_loading must be one of {MODES}, got {mode!r}")
    tools: list[dict[str, Any]] = session["tools"]
    deferred = [t for t in tools if is_deferred(t)]
    session["tool_loading"] = mode
    if mode == "eager" or not deferred:
        session["tools"] = [_strip_marker(t) if is_deferred(t) else t for t in tools]
        return
    session["_tool_catalog"] = [_strip_marker(t) for t in deferred]
    session["_tool_search"] = search or keyword_search
    session.setdefault("_revealed_tools", [])
    eager = [t for t in tools if not is_deferred(t) and tool_name(t) != SEARCH_TOOL_NAME]
    if mode == "kimi":
        session["tools"] = [SEARCH_TOOL_SPEC, *eager]
    else:  # glm: catalog stays declared, hidden by function.defer_loading
        session["tools"] = [SEARCH_TOOL_SPEC, *eager, *deferred]
    # search_tools is answered by the framework, but hooks/logging look it up.
    session["tool_dispatch"] = {**session["tool_dispatch"], SEARCH_TOOL_NAME: {"python_function": SEARCH_TOOL_NAME}}


def active(session: dict[str, Any]) -> bool:
    return session.get("tool_loading") in ("kimi", "glm") and bool(session.get("_tool_catalog"))


def handle_search(session: dict[str, Any], args: dict[str, Any]) -> tuple[Any, list[dict[str, Any]], str]:
    """Answer one ``search_tools`` call.

    Returns ``(tool_message_content, messages_to_append_after_the_step, text)``
    where *text* is a human/log rendering of the result.
    """
    catalog: list[dict[str, Any]] = session["_tool_catalog"]
    found = session["_tool_search"](str(args.get("query", "")), catalog)
    by_name = {tool_name(t): t for t in catalog}
    found = [n for n in found if n in by_name]
    if not found:
        return "No matching tools found. Try other keywords.", [], "no match"
    revealed: list[str] = session["_revealed_tools"]
    new = [n for n in found if n not in revealed]
    revealed.extend(new)
    text = "Found and loaded: " + ", ".join(found)
    if session["tool_loading"] == "glm":
        # GLM resolves each reference against `tools` on every request.
        return [{"type": "tool_reference", "name": n} for n in found], [], text
    extra = [{"role": "system", "tools": [by_name[n] for n in new]}] if new else []
    return text + ". They are now callable.", extra, text


def before_compaction(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Make *messages* safe for a request sent without ``tools``.

    GLM rejects a history containing ``tool_reference`` blocks when the request
    declares no tools (HTTP 400 code 1210), which is how the summary call goes out.
    """
    out = []
    for m in messages:
        content = m.get("content")
        if m.get("role") == "tool" and isinstance(content, list) and content \
                and isinstance(content[0], dict) and content[0].get("type") == "tool_reference":
            names = ", ".join(str(b.get("name")) for b in content if isinstance(b, dict))
            m = {**m, "content": f"Found and loaded: {names}"}
        out.append(m)
    return out


def after_compaction(session: dict[str, Any]) -> None:
    """Keep revealed tools available once their revealing messages are summarized.

    kimi: compaction keeps every system message, so nothing is lost.
    glm: the tool_reference results are gone; un-defer the revealed tools in
    ``tools`` (compaction already reset the cached prefix, so this costs nothing extra).
    """
    if session.get("tool_loading") != "glm":
        return
    referenced = {
        str(b.get("name"))
        for m in session["messages"] if m.get("role") == "tool" and isinstance(m.get("content"), list)
        for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_reference"
    }
    lost = set(session.get("_revealed_tools") or []) - referenced
    session["tools"] = [_strip_marker(t) if is_deferred(t) and tool_name(t) in lost else t
                        for t in session["tools"]]
