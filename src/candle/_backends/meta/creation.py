import numpy as np

from ..._C._tensor_impl import cy_make_tensor_from_storage  # pylint: disable=import-error,no-name-in-module
from ..._dtype import to_numpy_dtype
from ..._C import meta_typed_storage_from_shape


def _contiguous_stride(shape):
    stride = []
    acc = 1
    for d in reversed(shape):
        stride.append(acc)
        acc *= d
    return tuple(reversed(stride))


def _channels_last_stride(shape):
    if len(shape) != 4:
        raise RuntimeError("required rank 4 tensor to use channels_last format")
    _, c, h, w = shape
    return (c * h * w, 1, w * c, c)


def _resolve_stride(shape, memory_format):
    name = getattr(memory_format, "_name", None)
    if name in (None, "contiguous_format"):
        return _contiguous_stride(shape)
    if name == "channels_last":
        return _channels_last_stride(shape)
    raise RuntimeError(f"unsupported memory format {memory_format}")


def tensor_create_meta(data, dtype=None, device=None, requires_grad=False):
    arr = np.array(data, dtype=to_numpy_dtype(dtype))
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    storage = meta_typed_storage_from_shape(arr.shape, dtype, device=device)
    return cy_make_tensor_from_storage(storage, arr.shape, stride, 0, requires_grad)


def zeros_create_meta(shape, dtype=None, device=None, requires_grad=False, memory_format=None):
    shape = tuple(shape)
    stride = _resolve_stride(shape, memory_format)
    storage = meta_typed_storage_from_shape(shape, dtype, device=device)
    return cy_make_tensor_from_storage(storage, shape, stride, 0, requires_grad)


def ones_create_meta(shape, dtype=None, device=None, requires_grad=False, memory_format=None):
    shape = tuple(shape)
    stride = _resolve_stride(shape, memory_format)
    storage = meta_typed_storage_from_shape(shape, dtype, device=device)
    return cy_make_tensor_from_storage(storage, shape, stride, 0, requires_grad)


def empty_create_meta(shape, dtype=None, device=None, requires_grad=False, memory_format=None):
    shape = tuple(shape)
    stride = _resolve_stride(shape, memory_format)
    storage = meta_typed_storage_from_shape(shape, dtype, device=device)
    return cy_make_tensor_from_storage(storage, shape, stride, 0, requires_grad)


def arange_create_meta(start, end, step=1, dtype=None, device=None):
    arr = np.arange(start, end, step, dtype=to_numpy_dtype(dtype))
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    storage = meta_typed_storage_from_shape(arr.shape, dtype, device=device)
    return cy_make_tensor_from_storage(storage, arr.shape, stride, 0, False)


def linspace_create_meta(start, end, steps, dtype=None, device=None):
    arr = np.linspace(start, end, steps, dtype=to_numpy_dtype(dtype))
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    storage = meta_typed_storage_from_shape(arr.shape, dtype, device=device)
    return cy_make_tensor_from_storage(storage, arr.shape, stride, 0, False)


def full_create_meta(shape, fill_value, dtype=None, device=None, memory_format=None):
    shape = tuple(shape)
    stride = _resolve_stride(shape, memory_format)
    storage = meta_typed_storage_from_shape(shape, dtype, device=device)
    return cy_make_tensor_from_storage(storage, shape, stride, 0, False)


def logspace_create_meta(start, end, steps, dtype=None, device=None):
    arr = np.logspace(start, end, steps, dtype=to_numpy_dtype(dtype))
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    storage = meta_typed_storage_from_shape(arr.shape, dtype, device=device)
    return cy_make_tensor_from_storage(storage, arr.shape, stride, 0, False)


def eye_create_meta(n, m=None, dtype=None, device=None, out=None):
    if m is None:
        m = n
    if out is not None:
        return out
    arr = np.eye(n, m, dtype=to_numpy_dtype(dtype))
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    storage = meta_typed_storage_from_shape(arr.shape, dtype, device=device)
    return cy_make_tensor_from_storage(storage, arr.shape, stride, 0, False)


def range_create_meta(start, end, step=1, dtype=None, device=None):
    arr = np.arange(start, end + step, step, dtype=to_numpy_dtype(dtype))
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    storage = meta_typed_storage_from_shape(arr.shape, dtype, device=device)
    return cy_make_tensor_from_storage(storage, arr.shape, stride, 0, False)


def randn_create_meta(shape, dtype=None, device=None, requires_grad=False, memory_format=None, generator=None):
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    stride = _resolve_stride(shape, memory_format)
    storage = meta_typed_storage_from_shape(shape, dtype, device=device)
    return cy_make_tensor_from_storage(storage, shape, stride, 0, requires_grad)


__all__ = [
    "tensor_create_meta",
    "zeros_create_meta",
    "ones_create_meta",
    "empty_create_meta",
    "arange_create_meta",
    "linspace_create_meta",
    "full_create_meta",
    "logspace_create_meta",
    "eye_create_meta",
    "range_create_meta",
    "randn_create_meta",
]
