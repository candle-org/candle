from collections import namedtuple

from .._C._forward_ad import (  # pylint: disable=import-error,no-name-in-module
    _STATE,
    _JVP_RULES,
    _current_level,
    _disabled_levels,
    _level_stack,
    dual_level,
    enter_dual_level,
    exit_dual_level,
    get_jvp,
    get_tangent,
    is_level_disabled,
    make_dual,
    register_jvp,
    set_unpacked_dual_type,
    temporarily_disable,
    unpack_dual,
)


_UnpackedDualTensor = namedtuple("_UnpackedDualTensor", ["primal", "tangent"])


class UnpackedDualTensor(_UnpackedDualTensor):
    pass


def _tangent_or_zero(tangent, like):
    if tangent is None:
        from .._functional import zeros_like
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


set_unpacked_dual_type(UnpackedDualTensor)


__all__ = [
    "UnpackedDualTensor",
    "enter_dual_level",
    "exit_dual_level",
    "make_dual",
    "unpack_dual",
    "dual_level",
    "register_jvp",
    "get_jvp",
    "temporarily_disable",
    "get_tangent",
]


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
