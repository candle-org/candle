"""candle.utils.dlpack — compatibility shim for torch.utils.dlpack."""

from typing import Any

from .._tensor import Tensor


def to_dlpack(tensor: Any):
    if not isinstance(tensor, Tensor):
        raise TypeError("to_dlpack expects a candle.Tensor")
    if tensor.device.type != "cpu":
        raise RuntimeError("to_dlpack currently only supports CPU tensors")
    return tensor.numpy().__dlpack__()


__all__ = ["to_dlpack"]
