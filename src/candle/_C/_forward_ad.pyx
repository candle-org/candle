# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython-owned forward-mode AD state and JVP registry."""

import threading

_STATE = threading.local()
_JVP_RULES = {}


def _level_stack():
    stack = getattr(_STATE, "levels", None)
    if stack is None:
        stack = []
        _STATE.levels = stack
    return stack


def _current_level():
    stack = _level_stack()
    if not stack:
        return -1
    return stack[len(stack) - 1]


def _disabled_levels():
    disabled = getattr(_STATE, "disabled", None)
    if disabled is None:
        disabled = set()
        _STATE.disabled = disabled
    return disabled


class temporarily_disable:
    def __init__(self, level):
        self.level = level

    def __enter__(self):
        _disabled_levels().add(self.level)
        return self

    def __exit__(self, exc_type, exc, tb):
        _disabled_levels().discard(self.level)
        return False


def is_level_disabled(level):
    return level in _disabled_levels()


def get_tangent(tensor, level):
    if is_level_disabled(level):
        return None
    return tensor._fw_get(level)


def register_jvp(op_name, fn):
    _JVP_RULES[op_name] = fn


def get_jvp(op_name):
    return _JVP_RULES.get(op_name)


def enter_dual_level():
    stack = _level_stack()
    level = stack[len(stack) - 1] + 1 if stack else 0
    stack.append(level)
    return level


def exit_dual_level(*, level=None):
    stack = _level_stack()
    if not stack:
        raise RuntimeError(
            "Trying to exit a forward AD level but no level is currently active."
        )
    current = stack[len(stack) - 1]
    target = current if level is None else level
    if target != current:
        raise RuntimeError(
            "Trying to exit a forward AD level that was not the last one that was created."
        )
    stack.pop()


class dual_level:
    def __enter__(self):
        self.level = enter_dual_level()
        return self.level

    def __exit__(self, exc_type, exc, tb):
        exit_dual_level(level=self.level)
        return False
