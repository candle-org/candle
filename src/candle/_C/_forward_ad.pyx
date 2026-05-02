# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython-owned forward-mode AD state and JVP registry."""

import threading

_STATE = threading.local()
_JVP_RULES = {}
_UNPACKED_DUAL_TYPE = None


def set_unpacked_dual_type(unpacked_dual_type):
    global _UNPACKED_DUAL_TYPE
    _UNPACKED_DUAL_TYPE = unpacked_dual_type


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


def _validate_tangent(tensor, tangent):
    if not (tensor.is_floating_point() or tensor.is_complex()):
        raise ValueError(
            f"Expected primal to be floating point or complex, but got: {tensor.dtype}"
        )
    if not (tangent.is_floating_point() or tangent.is_complex()):
        raise ValueError(
            f"Expected tangent to be floating point or complex, but got: {tangent.dtype}"
        )
    if tensor.shape != tangent.shape:
        raise RuntimeError(
            f"Expected tangent to have the same shape as primal, but got: {tangent.shape}"
        )
    if tensor.dtype != tangent.dtype:
        raise RuntimeError(
            f"Expected tangent to have the same dtype as primal, but got: {tangent.dtype}"
        )


def make_dual(tensor, tangent, *, level=None):
    if level is None:
        level = _current_level()
    if level < 0:
        raise RuntimeError(
            "Trying to create a dual Tensor for forward AD but no level exists, make sure to enter_dual_level() first."
        )
    _validate_tangent(tensor, tangent)
    tensor._fw_set(level, tangent)
    return tensor


def unpack_dual(tensor, *, level=None):
    unpacked_dual_type = _UNPACKED_DUAL_TYPE
    if unpacked_dual_type is None:
        raise RuntimeError("UnpackedDualTensor type has not been registered")
    if level is None:
        level = _current_level()
    if level < 0:
        return unpacked_dual_type(tensor, None)
    return unpacked_dual_type(tensor, get_tangent(tensor, level))


def _tangent_or_zero(tangent, like):
    if tangent is None:
        from candle._functional import zeros_like
        return zeros_like(like)
    return tangent


def _unary_tangent(x, _tangents):
    return _tangent_or_zero(_tangents[0], x)


def _sum_jvp(x, *, _tangents, **kwargs):
    tangent = _tangent_or_zero(_tangents[0], x)
    dim = kwargs.get("dim")
    keepdim = kwargs.get("keepdim", False)
    if dim is None:
        return tangent.sum()
    return tangent.sum(dim=dim, keepdim=keepdim)


def _mean_jvp(x, *, _tangents, **kwargs):
    tangent = _tangent_or_zero(_tangents[0], x)
    dim = kwargs.get("dim")
    keepdim = kwargs.get("keepdim", False)
    if dim is None:
        return tangent.mean()
    return tangent.mean(dim=dim, keepdim=keepdim)


def _view_jvp(x, shape, *, _tangents):
    tangent = _tangent_or_zero(_tangents[0], x)
    return tangent.view(shape)


def _reshape_jvp(x, shape, *, _tangents):
    tangent = _tangent_or_zero(_tangents[0], x)
    return tangent.reshape(shape)


def _register_default_jvps():
    register_jvp(
        "add",
        lambda x, y, *, _tangents: _tangent_or_zero(_tangents[0], x)
        + _tangent_or_zero(_tangents[1], y),
    )
    register_jvp(
        "mul",
        lambda x, y, *, _tangents: _tangent_or_zero(_tangents[0], x) * y
        + x * _tangent_or_zero(_tangents[1], y),
    )
    register_jvp("sum", _sum_jvp)
    register_jvp("neg", lambda x, *, _tangents: -_unary_tangent(x, _tangents))
    register_jvp("exp", lambda x, *, _tangents: _unary_tangent(x, _tangents) * x.exp())
    register_jvp("sin", lambda x, *, _tangents: _unary_tangent(x, _tangents) * x.cos())
    register_jvp("cos", lambda x, *, _tangents: -_unary_tangent(x, _tangents) * x.sin())
    register_jvp(
        "tanh",
        lambda x, *, _tangents: _unary_tangent(x, _tangents)
        * (x._ones_like() - x.tanh() * x.tanh()),
    )
    register_jvp("mean", _mean_jvp)
    register_jvp("view", _view_jvp)
    register_jvp("reshape", _reshape_jvp)


_register_default_jvps()
