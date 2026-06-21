"""Tests for the public event subscription API — subscribe, unsubscribe, on."""

from __future__ import annotations

from agentknit._core import subscribe, unsubscribe, on, _emit, _default_event_handler


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_session() -> dict:
    return {
        "on_event": _default_event_handler,
        "_event_handlers": {},
    }


# ── subscribe / _emit ─────────────────────────────────────────────────────────

def test_subscribe_calls_handler():
    """subscribe registers a handler that _emit invokes."""
    session = _make_session()
    calls = []

    def handler(et: str, data: dict) -> None:
        calls.append((et, data))

    subscribe(session, "test_event", handler)
    _emit(session, "test_event", foo=1, fmt="test")

    assert len(calls) == 1
    assert calls[0][0] == "test_event"
    assert calls[0][1]["foo"] == 1


def test_subscribe_multiple_handlers():
    """Multiple handlers for the same event type are called in order."""
    session = _make_session()
    order = []

    def h1(et: str, data: dict) -> None:
        order.append("h1")

    def h2(et: str, data: dict) -> None:
        order.append("h2")

    subscribe(session, "test_event", h1)
    subscribe(session, "test_event", h2)
    _emit(session, "test_event", fmt="test")

    assert order == ["h1", "h2"]


def test_subscribe_different_events():
    """Handlers for different event types are isolated."""
    session = _make_session()
    calls_a = []
    calls_b = []

    def handler_a(et: str, data: dict) -> None:
        calls_a.append(et)

    def handler_b(et: str, data: dict) -> None:
        calls_b.append(et)

    subscribe(session, "event_a", handler_a)
    subscribe(session, "event_b", handler_b)

    _emit(session, "event_a", fmt="a")
    assert calls_a == ["event_a"]
    assert calls_b == []

    _emit(session, "event_b", fmt="b")
    assert calls_a == ["event_a"]
    assert calls_b == ["event_b"]


def test_subscribe_no_handler_for_event():
    """Emitting an event with no specific handler does not error."""
    session = _make_session()
    # Should not raise
    _emit(session, "unknown_event", fmt="test")


def test_subscribe_also_calls_generic_handler():
    """Per-event-type handlers are called before the generic on_event handler."""
    session = _make_session()
    order = []

    def specific(et: str, data: dict) -> None:
        order.append("specific")

    def generic(et: str, data: dict) -> None:
        order.append("generic")

    session["on_event"] = generic
    subscribe(session, "test_event", specific)
    _emit(session, "test_event", fmt="test")

    assert order == ["specific", "generic"]


# ── on alias ──────────────────────────────────────────────────────────────────

def test_on_alias():
    """on is an alias for subscribe."""
    session = _make_session()
    calls = []

    def handler(et: str, data: dict) -> None:
        calls.append(et)

    on(session, "test_event", handler)
    _emit(session, "test_event", fmt="test")

    assert calls == ["test_event"]


# ── unsubscribe ───────────────────────────────────────────────────────────────

def test_unsubscribe_removes_handler():
    """unsubscribe stops a handler from being called."""
    session = _make_session()
    calls = []

    def handler(et: str, data: dict) -> None:
        calls.append(et)

    subscribe(session, "test_event", handler)
    _emit(session, "test_event", fmt="1")
    assert len(calls) == 1

    unsubscribe(session, "test_event", handler)
    _emit(session, "test_event", fmt="2")
    assert len(calls) == 1  # still 1 — handler was not called again


def test_unsubscribe_unknown_handler():
    """unsubscribe on a handler not registered does nothing."""
    session = _make_session()

    def handler(et: str, data: dict) -> None:
        pass

    # Should not raise
    unsubscribe(session, "test_event", handler)


def test_unsubscribe_unknown_event():
    """unsubscribe on a never-registered event type does nothing."""
    session = _make_session()

    def handler(et: str, data: dict) -> None:
        pass

    unsubscribe(session, "nonexistent", handler)


# ── integration with generic on_event ─────────────────────────────────────────

def test_generic_handler_still_works():
    """The generic on_event handler still receives all events."""
    session = _make_session()
    calls = []

    def generic(et: str, data: dict) -> None:
        calls.append((et, data.get("foo")))

    session["on_event"] = generic
    _emit(session, "ev1", foo=10, fmt="")
    _emit(session, "ev2", foo=20, fmt="")

    assert calls == [("ev1", 10), ("ev2", 20)]


def test_default_handler_used_when_no_on_event():
    """Default _default_event_handler is used when session has no on_event."""
    session = {"_event_handlers": {}}  # no on_event key
    # Should not raise; _default_event_handler prints to stdout/stderr
    import io, sys
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        _emit(session, "test_event", fmt="hello world")
        # no exception = default handler worked
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
