from .function import Function
from .engine import grad
from . import graph
from . import _functions
from . import forward_ad


def _calculate_shape(output, grad, is_grads_batched):
    if isinstance(output, graph.GradientEdge):
        if is_grads_batched:
            raise RuntimeError("Batched grads are not supported with GradientEdge")
        out_shape = output.node._input_metadata[output.output_nr].shape
        return out_shape, grad.shape
    if is_grads_batched:
        return output.shape, grad.shape[1:]
    return output.shape, grad.shape

__all__ = ["Function", "grad", "graph", "_functions", "forward_ad", "_calculate_shape"]
