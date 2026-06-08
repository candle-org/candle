"""Lightweight FSDP phase profiling helpers.

The profiler is intentionally opt-in and records host wall-clock time around
existing FSDP phases. It does not change collective semantics or tensor device
placement.
"""
import time
from contextlib import contextmanager

_ENABLED = False
_EVENTS = {}


def reset(enabled=False):
    """Enable/disable profiling and clear accumulated events."""
    global _ENABLED, _EVENTS
    _ENABLED = bool(enabled)
    _EVENTS = {}


def enabled():
    """Return whether FSDP phase profiling is active."""
    return _ENABLED


@contextmanager
def record(name):
    """Record elapsed host time for one named FSDP phase when enabled."""
    if not _ENABLED:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        event = _EVENTS.setdefault(name, {"count": 0, "total_ms": 0.0})
        event["count"] += 1
        event["total_ms"] += elapsed_ms


def summary():
    """Return a JSON-serializable summary of recorded FSDP phase timings."""
    result = {}
    for name, event in sorted(_EVENTS.items()):
        count = event["count"]
        total_ms = event["total_ms"]
        result[name] = {
            "count": count,
            "total_ms": total_ms,
            "avg_ms": total_ms / count if count else 0.0,
        }
    return result
