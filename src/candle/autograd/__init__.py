from .function import Function
from .engine import backward, grad
from . import graph
from . import _functions
from . import forward_ad
from . import profiler
from . import profiler_util
from . import functional
from .anomaly_mode import (
    detect_anomaly,
    set_detect_anomaly,
    is_anomaly_enabled,
    is_anomaly_check_nan_enabled,
)


def _calculate_shape(output, grad, is_grads_batched):
    if isinstance(output, graph.GradientEdge):
        if is_grads_batched:
            raise RuntimeError("Batched grads are not supported with GradientEdge")
        out_shape = output.node._input_metadata[output.output_nr].shape
        return out_shape, grad.shape
    if is_grads_batched:
        return output.shape, grad.shape[1:]
    return output.shape, grad.shape


def kineto_available():
    return False


def Variable(*args, **kwargs):
    from .._tensor import Tensor

    if len(args) == 1 and isinstance(args[0], Tensor):
        return args[0]
    raise NotImplementedError("candle.autograd.Variable only supports Tensor passthrough")

__all__ = [
    "Function",
    "backward",
    "grad",
    "graph",
    "_functions",
    "forward_ad",
    "profiler",
    "profiler_util",
    "functional",
    "_calculate_shape",
    "detect_anomaly",
    "set_detect_anomaly",
    "is_anomaly_enabled",
    "is_anomaly_check_nan_enabled",
    "kineto_available",
    "Variable",
]
