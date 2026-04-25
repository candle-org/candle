"""Pure-Python thread-local storage for saved-tensors hooks.

This module intentionally has NO Cython dependency so that the
threading.local object lives in a plain Python module dict and is
never subject to Cython's __pyx_mstate_global offset bugs.
"""
import threading

_STATE = threading.local()


def get_stack():
    """Return (creating if needed) the per-thread hooks stack list."""
    stack = getattr(_STATE, 'hooks', None)
    if stack is None:
        stack = []
        _STATE.hooks = stack
    return stack
