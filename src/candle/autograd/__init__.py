"""Public shell for ``candle.autograd``.

Mirrors ``torch.autograd``'s public surface; runtime/mechanism lives in
``candle._C``.  This module only re-exports public names so that the typical
``import candle.autograd as autograd`` user gets the same API torch users
expect.
"""

from .._C._autograd_engine import (  # pylint: disable=import-error,no-name-in-module
    Variable,
    _calculate_shape,
    kineto_available,
)
from .anomaly_mode import (
    detect_anomaly,
    is_anomaly_check_nan_enabled,
    is_anomaly_enabled,
    set_detect_anomaly,
)
from .engine import backward, grad
from .function import Function
from . import _functions, forward_ad, functional, graph, profiler, profiler_util


__all__ = [
    "Function",
    "Variable",
    "_calculate_shape",
    "_functions",
    "backward",
    "detect_anomaly",
    "forward_ad",
    "functional",
    "grad",
    "graph",
    "is_anomaly_check_nan_enabled",
    "is_anomaly_enabled",
    "kineto_available",
    "profiler",
    "profiler_util",
    "set_detect_anomaly",
]
