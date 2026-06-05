# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython-owned autograd helper utilities."""

import numpy as np

from candle._C import typed_storage_from_numpy
from candle._C._tensor_impl import cy_make_tensor_from_storage
from candle._dtype import bfloat16 as _bfloat16_dtype


def _f32_to_bf16_bits(arr):
    u32 = arr.astype(np.float32, copy=False).view(np.uint32)
    rounding_bias = (u32 >> 16) & 1
    u32 = u32 + 0x7FFF + rounding_bias
    return (u32 >> 16).astype(np.uint16)


def _numeric_to_storage_array(arr, dtype):
    if dtype == _bfloat16_dtype:
        return _f32_to_bf16_bits(arr)
    return arr


def reduce_grad(grad, shape):
    """Reduce gradient to match the target shape by summing broadcast dimensions."""
    if grad.shape == shape:
        return grad

    if grad.device.type != "cpu":
        return _reduce_grad_dispatch(grad, shape)

    arr = np.asarray(grad.numpy())
    while arr.ndim > len(shape):
        arr = arr.sum(axis=0)
    for i, (g_dim, s_dim) in enumerate(zip(arr.shape, shape)):
        if s_dim == 1 and g_dim != 1:
            arr = arr.sum(axis=i, keepdims=True)
    arr = _numeric_to_storage_array(arr, grad.dtype)
    storage = typed_storage_from_numpy(arr, grad.dtype)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return cy_make_tensor_from_storage(storage, arr.shape, stride, 0, False)


def _reduce_grad_dispatch(grad, shape):
    """reduce_grad using dispatched ops for non-CPU devices."""
    from candle._functional import sum as torch_sum
    from candle.autograd.grad_mode import no_grad

    result = grad
    cdef int rank_diff
    with no_grad():
        # Squash leading broadcast dims (where grad has more dims than target)
        # in a single multi-dim sum rather than looping one dim at a time.
        # Cuts dispatch overhead when grad is rank > target rank by more than 1.
        rank_diff = len(result.shape) - len(shape)
        if rank_diff > 0:
            if rank_diff == 1:
                result = torch_sum(result, dim=0)
            else:
                result = torch_sum(result, dim=tuple(range(rank_diff)))
        # Reduce broadcast-stretched dims (size 1 in target, larger in grad)
        # in a single multi-dim sum with keepdim=True. Same motivation.
        reduce_axes = [i for i, (g_dim, s_dim) in enumerate(zip(result.shape, shape))
                       if s_dim == 1 and g_dim != 1]
        if reduce_axes:
            if len(reduce_axes) == 1:
                result = torch_sum(result, dim=reduce_axes[0], keepdim=True)
            else:
                result = torch_sum(result, dim=tuple(reduce_axes), keepdim=True)
    return result
