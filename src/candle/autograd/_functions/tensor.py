import operator
from functools import reduce

from ..function import Function


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
