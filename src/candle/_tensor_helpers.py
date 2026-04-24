"""Helper functions and classes for tensor implementation."""
import numpy as np


class _StrideTuple(tuple):
    """A tuple subclass that is also callable, matching PyTorch's stride() API."""

    def __call__(self, dim=None):
        if dim is None:
            return tuple(self)
        if dim < 0:
            dim += len(self)
        return self[dim]


def _compute_strides(shape):
    stride = []
    acc = 1
    for d in reversed(shape):
        stride.append(acc)
        acc *= d
    return _StrideTuple(reversed(stride))


def _bf16_to_f32(arr):
    """Convert bfloat16 (stored as uint16) to float32."""
    u32 = arr.astype(np.uint32) << 16
    return u32.view(np.float32)


def _f32_to_bf16(arr):
    """Convert float32 to bfloat16 (stored as uint16), round-to-nearest-even."""
    u32 = arr.view(np.uint32)
    rounding_bias = (u32 >> 16) & 1
    u32 = u32 + 0x7FFF + rounding_bias
    return (u32 >> 16).astype(np.uint16)


class _HookHandle:
    _next_id = 0

    def __init__(self, hooks):
        self._hooks = hooks
        self.id = _HookHandle._next_id
        _HookHandle._next_id += 1

    def remove(self):
        if self._hooks is None:
            return
        self._hooks.pop(self.id, None)
        self._hooks = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove()
