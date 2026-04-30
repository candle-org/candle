import numpy as np

from ..._dtype import to_numpy_dtype
from ..._C import typed_storage_from_numpy
from ..._C._tensor_impl import cy_make_tensor_from_storage  # pylint: disable=import-error,no-name-in-module


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


def tensor_create(data, dtype=None, device=None, requires_grad=False, memory_format=None):
    arr = np.array(data, dtype=to_numpy_dtype(dtype))
    storage = typed_storage_from_numpy(arr, dtype, device=device)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return cy_make_tensor_from_storage(storage, arr.shape, stride, 0, requires_grad)


def zeros_create(shape, dtype=None, device=None, requires_grad=False, memory_format=None):
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    storage = typed_storage_from_numpy(np.zeros(shape, dtype=to_numpy_dtype(dtype)), dtype, device=device)
    stride = _resolve_stride(shape, memory_format)
    return cy_make_tensor_from_storage(storage, shape, stride, 0, requires_grad)


def ones_create(shape, dtype=None, device=None, requires_grad=False, memory_format=None):
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    storage = typed_storage_from_numpy(np.ones(shape, dtype=to_numpy_dtype(dtype)), dtype, device=device)
    stride = _resolve_stride(shape, memory_format)
    return cy_make_tensor_from_storage(storage, shape, stride, 0, requires_grad)


def empty_create(shape, dtype=None, device=None, requires_grad=False, memory_format=None):
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    storage = typed_storage_from_numpy(np.empty(shape, dtype=to_numpy_dtype(dtype)), dtype, device=device)
    stride = _resolve_stride(shape, memory_format)
    return cy_make_tensor_from_storage(storage, shape, stride, 0, requires_grad)


def arange_create(start, end, step=1, dtype=None, device=None):
    arr = np.arange(start, end, step, dtype=to_numpy_dtype(dtype))
    storage = typed_storage_from_numpy(arr, dtype, device=device)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return cy_make_tensor_from_storage(storage, arr.shape, stride, 0, False)


def linspace_create(start, end, steps, dtype=None, device=None):
    arr = np.linspace(start, end, steps, dtype=to_numpy_dtype(dtype))
    storage = typed_storage_from_numpy(arr, dtype, device=device)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return cy_make_tensor_from_storage(storage, arr.shape, stride, 0, False)


def full_create(shape, fill_value, dtype=None, device=None, memory_format=None):
    shape = tuple(shape)
    storage = typed_storage_from_numpy(
        np.full(shape, fill_value, dtype=to_numpy_dtype(dtype)),
        dtype,
        device=device,
    )
    stride = _resolve_stride(shape, memory_format)
    return cy_make_tensor_from_storage(storage, shape, stride, 0, False)


def logspace_create(start, end, steps, dtype=None, device=None):
    arr = np.logspace(start, end, steps, dtype=to_numpy_dtype(dtype))
    storage = typed_storage_from_numpy(arr, dtype, device=device)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return cy_make_tensor_from_storage(storage, arr.shape, stride, 0, False)


def eye_create(n, m=None, dtype=None, device=None, out=None):
    if m is None:
        m = n
    arr = np.eye(n, m, dtype=to_numpy_dtype(dtype))
    if out is not None:
        out_arr = out._numpy_view()
        out_arr[:] = arr.astype(out_arr.dtype)
        return out
    storage = typed_storage_from_numpy(arr, dtype, device=device)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return cy_make_tensor_from_storage(storage, arr.shape, stride, 0, False)


def range_create(start, end, step=1, dtype=None, device=None):
    arr = np.arange(start, end + step, step, dtype=to_numpy_dtype(dtype))
    storage = typed_storage_from_numpy(arr, dtype, device=device)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return cy_make_tensor_from_storage(storage, arr.shape, stride, 0, False)


def randn_create(shape, dtype=None, device=None, requires_grad=False, memory_format=None, generator=None):
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    from ..._random import _get_cpu_rng
    rng = generator._rng if (generator is not None and hasattr(generator, '_rng') and generator._rng is not None) else _get_cpu_rng()
    arr = rng.randn(*shape)
    if not shape:
        # numpy RandomState.randn() returns a Python float for empty shape.
        arr = np.array(arr)
    arr = arr.astype(to_numpy_dtype(dtype))
    storage = typed_storage_from_numpy(arr, dtype, device=device)
    stride = _resolve_stride(shape, memory_format)
    return cy_make_tensor_from_storage(storage, shape, stride, 0, requires_grad)


def rand_create(shape, dtype=None, device=None, requires_grad=False, memory_format=None, generator=None):
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    from ..._random import _get_cpu_rng
    rng = generator._rng if (generator is not None and hasattr(generator, '_rng') and generator._rng is not None) else _get_cpu_rng()
    arr = rng.random_sample(shape)
    if not shape:
        # numpy RandomState.random_sample(()) returns a scalar for empty shape.
        arr = np.array(arr)
    arr = arr.astype(to_numpy_dtype(dtype))
    storage = typed_storage_from_numpy(arr, dtype, device=device)
    stride = _resolve_stride(shape, memory_format)
    return cy_make_tensor_from_storage(storage, shape, stride, 0, requires_grad)


def randint_create(low, high=None, size=None, dtype=None, device=None, requires_grad=False, generator=None, memory_format=None, **kwargs):
    """torch.randint(low=0, high, size, ...) — fills with random integers from [low, high)."""
    from ..._dtype import int64 as int64_dtype
    if high is None:
        low, high = 0, low
    if size is None:
        raise ValueError("size is required for randint")
    if isinstance(size, int):
        size = (size,)
    size = tuple(size)
    from ..._random import _get_cpu_rng
    rng = generator._rng if (generator is not None and hasattr(generator, '_rng') and generator._rng is not None) else _get_cpu_rng()
    arr = rng.randint(int(low), int(high), size=size).astype(np.int64)
    out_dtype = dtype if dtype is not None else int64_dtype
    storage = typed_storage_from_numpy(arr, out_dtype, device=device)
    stride = _resolve_stride(size, memory_format)
    return cy_make_tensor_from_storage(storage, size, stride, 0, requires_grad)


def randperm_create(n, dtype=None, device=None, requires_grad=False, generator=None, **kwargs):
    """torch.randperm(n) — random permutation of integers 0..n-1."""
    from ..._dtype import int64 as int64_dtype
    from ..._random import _get_cpu_rng
    rng = generator._rng if (generator is not None and hasattr(generator, '_rng') and generator._rng is not None) else _get_cpu_rng()
    arr = rng.permutation(int(n)).astype(np.int64)
    out_dtype = dtype if dtype is not None else int64_dtype
    storage = typed_storage_from_numpy(arr, out_dtype, device=device)
    stride = _contiguous_stride(arr.shape)
    return cy_make_tensor_from_storage(storage, arr.shape, stride, 0, requires_grad)
