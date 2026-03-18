"""Checkpoint recomputation context tracking for autograd compatibility."""

from __future__ import annotations

import contextlib
import threading


_STATE = threading.local()


def _get_state():
    state = getattr(_STATE, "state", None)
    if state is None:
        state = {"early_stop": True, "saved_counts": []}
        _STATE.state = state
    return state


def save_expected_count(count: int) -> None:
    _get_state()["saved_counts"].append(int(count))


def pop_expected_count() -> int | None:
    saved = _get_state()["saved_counts"]
    if not saved:
        return None
    return saved.pop()


def early_stop_enabled() -> bool:
    return bool(_get_state()["early_stop"])


def set_early_stop_enabled(enabled: bool | None) -> None:
    if enabled is None:
        return
    _get_state()["early_stop"] = bool(enabled)


@contextlib.contextmanager
def checkpoint_context():
    try:
        yield None
    finally:
        return
