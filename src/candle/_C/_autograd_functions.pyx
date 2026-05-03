# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython-owned built-in autograd Function classes.

Mirrors torch/autograd/_functions/tensor.py in the compiled boundary.

Currently houses ``Resize``, the autograd Function backing
``Tensor.resize`` / ``Tensor.resize_as`` non-inplace variants.

Keeping the class definition (and its forward/backward staticmethods) inside a
real C extension matches the repository-wide rule: Python public surface mirrors
torch 1:1, runtime/mechanism lives in Cython ``_C``.
"""

import operator
from functools import reduce


def _import_function():
    """Lazy import to avoid a circular import at module load time.

    ``Function`` is a Python-defined class living in
    ``candle.autograd.function`` (its public shell) but with the metaclass and
    apply path owned by ``candle._C._autograd_function``.  We pull it in lazily
    because ``candle.autograd.function`` imports from this layer transitively
    via ``candle.autograd._functions`` re-exports.
    """
    from candle.autograd.function import Function  # pylint: disable=import-outside-toplevel
    return Function


def _make_resize():
    Function = _import_function()

    class Resize(Function):
        @staticmethod
        def forward(ctx, tensor, sizes):
            ctx.sizes = tuple(sizes)
            ctx.numel = reduce(operator.mul, ctx.sizes, 1)
            if tensor.numel() != ctx.numel:
                raise RuntimeError(
                    (
                        "requested resize to {} ({} elements in total), "
                        "but the given tensor has a size of {} ({} elements). "
                        "autograd's resize can only change the shape of a given "
                        "tensor, while preserving the number of elements. "
                    ).format(
                        "x".join(map(str, ctx.sizes)),
                        ctx.numel,
                        "x".join(map(str, tensor.size())),
                        tensor.numel(),
                    )
                )
            ctx.input_sizes = tensor.size()
            return tensor.contiguous().view(*ctx.sizes)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output.contiguous().view(ctx.input_sizes), None

    Resize.__module__ = "candle._C._autograd_functions"
    Resize.forward.__module__ = "candle._C._autograd_functions"
    Resize.backward.__module__ = "candle._C._autograd_functions"
    return Resize


Resize = _make_resize()


__all__ = ["Resize"]
