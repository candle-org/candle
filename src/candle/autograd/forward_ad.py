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
