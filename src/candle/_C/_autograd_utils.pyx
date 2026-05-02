# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython-owned autograd helper utilities."""

import numpy as np

from candle._C import typed_storage_from_numpy
from candle._C._tensor_impl import cy_make_tensor_from_storage


def reduce_grad(grad, shape):
    """Reduce gradient to match the target shape by summing broadcast dimensions."""
    if grad.shape == shape:
        return grad

    if grad.device.type != "cpu":
        return _reduce_grad_dispatch(grad, shape)

    arr = np.array(grad.storage().data).reshape(grad.shape)
    while arr.ndim > len(shape):
        arr = arr.sum(axis=0)
    for i, (g_dim, s_dim) in enumerate(zip(arr.shape, shape)):
        if s_dim == 1 and g_dim != 1:
            arr = arr.sum(axis=i, keepdims=True)
    storage = typed_storage_from_numpy(arr, grad.dtype)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return cy_make_tensor_from_storage(storage, arr.shape, stride, 0, False)


def _reduce_grad_dispatch(grad, shape):
    """reduce_grad using dispatched ops for non-CPU devices."""
    from candle._functional import sum as torch_sum
    from candle.autograd.grad_mode import no_grad

    result = grad
    with no_grad():
        while len(result.shape) > len(shape):
            result = torch_sum(result, dim=0)
        for i, (g_dim, s_dim) in enumerate(zip(result.shape, shape)):
            if s_dim == 1 and g_dim != 1:
                result = torch_sum(result, dim=i, keepdim=True)
    return result
