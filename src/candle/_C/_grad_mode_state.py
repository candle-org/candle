"""Pure-Python thread-local storage for grad-mode state.

Intentionally has no Cython dependency so the threading.local object
lives in a plain module dict (same pattern as _hooks_state.py).
"""
import threading

_STATE = threading.local()


def get_enabled():
    return getattr(_STATE, 'enabled', True)


def set_enabled(value):
    _STATE.enabled = bool(value)


def get_creation_mode():
    return getattr(_STATE, 'creation_mode', None)


def set_creation_mode(mode):
    _STATE.creation_mode = mode
