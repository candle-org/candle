import numpy as np

from ..._cython._tensor_impl import cy_make_tensor_from_storage  # pylint: disable=import-error,no-name-in-module
from ..._dtype import float32 as float32_dtype
from ..._dtype import float64 as float64_dtype
from ..._dtype import to_numpy_dtype
from ..._C import npu_typed_storage_from_ptr
from . import runtime as npu_runtime
from . import aclnn
from . import state as npu_state
from . import ops_soc


def _wrap_tensor(storage, shape, stride, requires_grad):
    return cy_make_tensor_from_storage(storage, tuple(shape), stride, 0, requires_grad)


def _require_inplace_one_zero():
    if not aclnn.ones_zero_symbols_ok():
        raise RuntimeError("aclnnInplaceOne/Zero not available")


def tensor_create(data, dtype=None, device=None, requires_grad=False, memory_format=None):
    arr = np.array(data, dtype=npu_runtime._dtype_to_numpy(dtype))
    runtime = npu_runtime.get_runtime((device.index if hasattr(device, "index") else None) or 0)
    stream = npu_state.current_stream((device.index if hasattr(device, "index") else None) or 0)
    ptr, _ = npu_runtime._copy_cpu_to_npu(arr, runtime=runtime)
    storage = npu_typed_storage_from_ptr(ptr, arr.size, dtype, device=device)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return _wrap_tensor(storage, arr.shape, stride, requires_grad)


def zeros_create(shape, dtype=None, device=None, requires_grad=False, memory_format=None):
    runtime = npu_runtime.get_runtime((device.index if hasattr(device, "index") else None) or 0)
    stream = npu_state.current_stream((device.index if hasattr(device, "index") else None) or 0)
    _require_inplace_one_zero()
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    size = int(np.prod(shape))
    out_size = size * np.dtype(npu_runtime._dtype_to_numpy(dtype)).itemsize
    ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    stride = npu_runtime._contiguous_stride(shape)
    aclnn.inplace_zero(ptr, shape, stride, dtype, runtime, stream=stream.stream)
    storage = npu_typed_storage_from_ptr(ptr, size, dtype, device=device)
    return _wrap_tensor(storage, shape, stride, requires_grad)


def ones_create(shape, dtype=None, device=None, requires_grad=False, memory_format=None):
    runtime = npu_runtime.get_runtime((device.index if hasattr(device, "index") else None) or 0)
    stream = npu_state.current_stream((device.index if hasattr(device, "index") else None) or 0)
    _require_inplace_one_zero()
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    size = int(np.prod(shape))
    out_size = size * np.dtype(npu_runtime._dtype_to_numpy(dtype)).itemsize
    ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    stride = npu_runtime._contiguous_stride(shape)
    aclnn.inplace_one(ptr, shape, stride, dtype, runtime, stream=stream.stream)
    storage = npu_typed_storage_from_ptr(ptr, size, dtype, device=device)
    return _wrap_tensor(storage, shape, stride, requires_grad)


def empty_create(shape, dtype=None, device=None, requires_grad=False, memory_format=None):
    runtime = npu_runtime.get_runtime((device.index if hasattr(device, "index") else None) or 0)
    stream = npu_state.current_stream((device.index if hasattr(device, "index") else None) or 0)
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    size = int(np.prod(shape))
    out_size = size * np.dtype(npu_runtime._dtype_to_numpy(dtype)).itemsize
    ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    storage = npu_typed_storage_from_ptr(ptr, size, dtype, device=device)
    stride = npu_runtime._contiguous_stride(shape)
    return _wrap_tensor(storage, shape, stride, requires_grad)


def randn_create(shape, dtype=None, device=None, requires_grad=False, memory_format=None, generator=None):
    """Create a tensor filled with random numbers from N(0,1) on NPU."""
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    # Create empty NPU tensor, then fill with normal_ (uses ACLNN kernel)
    t = empty_create(shape, dtype=dtype, device=device, requires_grad=requires_grad,
                     memory_format=memory_format)
    from .ops import normal_
    normal_(t, mean=0.0, std=1.0, generator=generator)
    return t


def rand_create(shape, dtype=None, device=None, requires_grad=False, memory_format=None, generator=None):
    """Create a tensor filled with random numbers from U(0,1) on NPU."""
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    t = empty_create(shape, dtype=dtype, device=device, requires_grad=requires_grad,
                     memory_format=memory_format)
    from .ops import uniform_
    uniform_(t, low=0.0, high=1.0, generator=generator)
    return t


def _resolve_dtype(dtype):
    return float32_dtype if dtype is None else dtype


def _contiguous_stride(shape):
    stride = []
    acc = 1
    for d in reversed(shape):
        stride.append(acc)
        acc *= d
    return tuple(reversed(stride))


def _device_index(device):
    return (device.index if hasattr(device, "index") else None) or 0



def _linspace_fallback_npu(start, end, steps, dtype, device):
    from . import ops as npu_ops

    steps = int(steps)
    shape = (steps,)
    if steps == 0:
        return empty_create(shape, dtype=dtype, device=device, requires_grad=False)
    if steps == 1:
        return full_create(shape, start, dtype=dtype, device=device)

    # Build on NPU without calling aclnn.arange (keeps linspace single-op test semantics).
    one = ones_create(shape, dtype=dtype, device=device)
    idx = npu_ops.cumsum(one, dim=0)
    idx = npu_ops.sub(idx, 1)
    step = (float(end) - float(start)) / float(steps - 1)
    idx = npu_ops.mul(idx, step)
    return npu_ops.add(idx, float(start))


def arange_create(start, end, step=1, dtype=None, device=None):
    dtype = _resolve_dtype(dtype)
    if step == 0:
        raise ValueError("step must be nonzero")
    if not aclnn.arange_symbols_ok():
        raise RuntimeError("aclnnArange not available")

    arr = np.arange(start, end, step, dtype=to_numpy_dtype(dtype))
    shape = tuple(arr.shape)
    stride = _contiguous_stride(shape)
    runtime = npu_runtime.get_runtime(_device_index(device))
    stream = npu_state.current_stream(_device_index(device))
    size = int(arr.size)
    itemsize = np.dtype(npu_runtime._dtype_to_numpy(dtype)).itemsize
    ptr = npu_runtime._alloc_device(max(size, 1) * itemsize, runtime=runtime)
    aclnn.arange(start, end, step, ptr, shape, stride, dtype, runtime, stream=stream.stream)
    storage = npu_typed_storage_from_ptr(ptr, max(size, 1), dtype, device=device)
    return _wrap_tensor(storage, shape, stride, requires_grad=False)


def full_create(shape, fill_value, dtype=None, device=None):
    dtype = _resolve_dtype(dtype)
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    base = zeros_create(shape, dtype=dtype, device=device)
    from . import ops as npu_ops

    return npu_ops.add(base, fill_value)


def linspace_create(start, end, steps, dtype=None, device=None):
    if steps < 0:
        raise ValueError("number of steps must be non-negative")
    dtype = _resolve_dtype(dtype)

    if ops_soc.use_smallop_linspace():
        return _linspace_fallback_npu(start, end, steps, dtype=dtype, device=device)

    if not aclnn.linspace_symbols_ok():
        raise RuntimeError("aclnnLinspace not available")

    shape = (int(steps),)
    stride = _contiguous_stride(shape)
    runtime = npu_runtime.get_runtime(_device_index(device))
    stream = npu_state.current_stream(_device_index(device))
    size = int(steps)
    itemsize = np.dtype(npu_runtime._dtype_to_numpy(dtype)).itemsize
    ptr = npu_runtime._alloc_device(max(size, 1) * itemsize, runtime=runtime)
    aclnn.linspace(start, end, steps, ptr, shape, stride, dtype, runtime, stream=stream.stream)
    storage = npu_typed_storage_from_ptr(ptr, max(size, 1), dtype, device=device)
    return _wrap_tensor(storage, shape, stride, requires_grad=False)


def logspace_create(start, end, steps, dtype=None, device=None):
    dtype = _resolve_dtype(dtype)
    linear = linspace_create(start, end, steps, dtype=dtype, device=device)
    from . import ops as npu_ops

    base = full_create(linear.shape, 10.0, dtype=dtype, device=device)
    return npu_ops.pow(base, linear)


def eye_create(n, m=None, dtype=None, device=None, out=None):
    if m is None:
        m = n
    if n < 0:
        raise ValueError(f"n must be greater or equal to 0, got {n}")
    if m < 0:
        raise ValueError(f"m must be greater or equal to 0, got {m}")
    dtype = _resolve_dtype(dtype)
    if not aclnn.eye_symbols_ok():
        raise RuntimeError("aclnnEye not available")

    shape = (int(n), int(m))
    stride = _contiguous_stride(shape)
    runtime = npu_runtime.get_runtime(_device_index(device))
    stream = npu_state.current_stream(_device_index(device))

    if out is not None:
        out_storage = out.storage()
        aclnn.eye(n, m, out_storage.data_ptr(), out.shape, out.stride, out.dtype, runtime, stream=stream.stream)
        return out

    size = int(n) * int(m)
    itemsize = np.dtype(npu_runtime._dtype_to_numpy(dtype)).itemsize
    ptr = npu_runtime._alloc_device(max(size, 1) * itemsize, runtime=runtime)
    aclnn.eye(n, m, ptr, shape, stride, dtype, runtime, stream=stream.stream)
    storage = npu_typed_storage_from_ptr(ptr, max(size, 1), dtype, device=device)
    return _wrap_tensor(storage, shape, stride, requires_grad=False)


def range_create(start, end, step=1, dtype=None, device=None):
    dtype = _resolve_dtype(dtype)
    if not aclnn.range_symbols_ok():
        raise RuntimeError("aclnnRange not available")

    # torch.range is inclusive, so shape is inferred from inclusive end.
    arr = np.arange(start, end + step, step, dtype=to_numpy_dtype(dtype))
    shape = tuple(arr.shape)
    stride = _contiguous_stride(shape)
    runtime = npu_runtime.get_runtime(_device_index(device))
    stream = npu_state.current_stream(_device_index(device))
    size = int(arr.size)
    itemsize = np.dtype(npu_runtime._dtype_to_numpy(dtype)).itemsize
    ptr = npu_runtime._alloc_device(max(size, 1) * itemsize, runtime=runtime)
    aclnn.range(start, end, step, ptr, shape, stride, dtype, runtime, stream=stream.stream)
    storage = npu_typed_storage_from_ptr(ptr, max(size, 1), dtype, device=device)
    return _wrap_tensor(storage, shape, stride, requires_grad=False)


def empty_strided_create(size, stride, dtype=None, device=None, requires_grad=False):
    shape = tuple(int(x) for x in size)
    stride = tuple(int(x) for x in stride)
    runtime = npu_runtime.get_runtime((device.index if hasattr(device, "index") else None) or 0)
    storage_size = 0
    if shape:
        storage_size = 1
        for dim, st in zip(shape, stride):
            if dim > 0:
                storage_size = max(storage_size, (dim - 1) * st + 1)
    out_size = storage_size * np.dtype(npu_runtime._dtype_to_numpy(dtype)).itemsize
    ptr = npu_runtime._alloc_device(max(out_size, 1), runtime=runtime)
    storage = npu_typed_storage_from_ptr(ptr, max(storage_size, 1), dtype, device=device)
    return _wrap_tensor(storage, shape, stride, requires_grad)


def scalar_tensor_create(data, dtype=None, device=None, requires_grad=False):
    arr = np.array([data], dtype=npu_runtime._dtype_to_numpy(dtype))
    runtime = npu_runtime.get_runtime((device.index if hasattr(device, "index") else None) or 0)
    ptr, _ = npu_runtime._copy_cpu_to_npu(arr, runtime=runtime)
    storage = npu_typed_storage_from_ptr(ptr, 1, dtype, device=device)
    return _wrap_tensor(storage, (), (), requires_grad)


def zeros_like_create(other, dtype=None, device=None, requires_grad=False, memory_format=None):
    return zeros_create(other.shape, dtype=dtype or other.dtype, device=device or other.device,
                       requires_grad=requires_grad)


def ones_like_create(other, dtype=None, device=None, requires_grad=False, memory_format=None):
    return ones_create(other.shape, dtype=dtype or other.dtype, device=device or other.device,
                      requires_grad=requires_grad)


def full_like_create(other, fill_value, dtype=None, device=None, requires_grad=False, memory_format=None):
    return full_create(other.shape, fill_value, dtype=dtype or other.dtype, device=device or other.device)


def empty_like_create(other, dtype=None, device=None, requires_grad=False, memory_format=None):
    return empty_create(other.shape, dtype=dtype or other.dtype, device=device or other.device,
                       requires_grad=requires_grad)


def randn_like_create(other, dtype=None, device=None, requires_grad=False, memory_format=None, generator=None):
    return randn_create(other.shape, dtype=dtype or other.dtype, device=device or other.device,
                       requires_grad=requires_grad, generator=generator)


def rand_like_create(other, dtype=None, device=None, requires_grad=False, memory_format=None, generator=None):
    return rand_create(other.shape, dtype=dtype or other.dtype, device=device or other.device,
                      requires_grad=requires_grad, generator=generator)


def randperm_create(n, dtype=None, device=None, requires_grad=False, generator=None, **kwargs):
    dtype = _resolve_dtype(dtype)
    if not aclnn.randperm_symbols_ok():
        raise RuntimeError("aclnnRandperm not available")
    if int(n) < 0:
        raise ValueError("n must be non-negative")
    shape = (int(n),)
    stride = _contiguous_stride(shape)
    runtime = npu_runtime.get_runtime(_device_index(device))
    stream = npu_state.current_stream(_device_index(device))
    size = int(n)
    itemsize = np.dtype(npu_runtime._dtype_to_numpy(dtype)).itemsize
    ptr = npu_runtime._alloc_device(max(size, 1) * itemsize, runtime=runtime)
    seed = int(npu_runtime._current_seed())
    offset = 0
    aclnn.randperm(seed, offset, int(n), ptr, shape, stride, dtype, runtime, stream=stream.stream)
    storage = npu_typed_storage_from_ptr(ptr, max(size, 1), dtype, device=device)
    return _wrap_tensor(storage, shape, stride, requires_grad=False)


def randint_create(low, high=None, size=None, dtype=None, device=None, requires_grad=False, generator=None, **kwargs):
    if high is None:
        low, high = 0, low
    if size is None:
        raise ValueError("size is required for randint")
    if isinstance(size, int):
        size = (size,)
    shape = tuple(size)
    dtype = _resolve_dtype(dtype)
    # Build from uniform [0,1), then scale+shift+floor
    t = rand_create(shape, dtype=float64_dtype, device=device, requires_grad=False, generator=generator)
    span = float(high - low)
    from . import ops as npu_ops
    t = npu_ops.mul(t, span)
    t = npu_ops.floor(t)
    t = npu_ops.add(t, float(low))
    if dtype != float64_dtype:
        t = npu_ops.to_dtype(t, dtype)
    return t


def new_empty_create(tensor, shape, dtype=None, device=None, requires_grad=False, memory_format=None):
    if shape is None:
        shape = ()
    return empty_create(shape, dtype=dtype or tensor.dtype, device=device or tensor.device,
                       requires_grad=requires_grad)


def new_full_create(tensor, shape, fill_value, dtype=None, device=None, requires_grad=False):
    if shape is None:
        shape = ()
    return full_create(shape, fill_value, dtype=dtype or tensor.dtype, device=device or tensor.device)


def new_zeros_create(tensor, shape, dtype=None, device=None, requires_grad=False, memory_format=None):
    if shape is None:
        shape = ()
    return zeros_create(shape, dtype=dtype or tensor.dtype, device=device or tensor.device,
                       requires_grad=requires_grad)


def new_ones_create(tensor, shape, dtype=None, device=None, requires_grad=False, memory_format=None):
    if shape is None:
        shape = ()
    return ones_create(shape, dtype=dtype or tensor.dtype, device=device or tensor.device,
                      requires_grad=requires_grad)


def new_tensor_create(tensor, data, dtype=None, device=None, requires_grad=False):
    return tensor_create(data, dtype=dtype or tensor.dtype, device=device or tensor.device,
                        requires_grad=requires_grad)


def tril_indices_create(row, col, offset=0, dtype=None, device=None):
    from ..._dtype import int64 as int64_dtype
    dtype = int64_dtype if dtype is None else dtype
    if not aclnn.tril_indices_symbols_ok():
        raise RuntimeError("aclnnTrilIndices not available")
    out_shape = tuple(_tril_indices_shape(row, col, offset))
    count = int(out_shape[1])
    shape = (2, count)
    stride = _contiguous_stride(shape)
    runtime = npu_runtime.get_runtime(_device_index(device))
    stream = npu_state.current_stream(_device_index(device))
    ptr = npu_runtime._alloc_device(max(2 * max(count, 1), 1) * np.dtype(np.int64).itemsize, runtime=runtime)
    aclnn.tril_indices(row, col, offset, ptr, shape, stride, dtype, runtime, stream=stream.stream)
    storage = npu_typed_storage_from_ptr(ptr, max(2 * max(count, 1), 1), dtype, device=device)
    return _wrap_tensor(storage, shape, stride, requires_grad=False)


def triu_indices_create(row, col, offset=0, dtype=None, device=None):
    from ..._dtype import int64 as int64_dtype
    dtype = int64_dtype if dtype is None else dtype
    if not aclnn.triu_indices_symbols_ok():
        raise RuntimeError("aclnnTriuIndices not available")
    out_shape = tuple(_triu_indices_shape(row, col, offset))
    count = int(out_shape[1])
    shape = (2, count)
    stride = _contiguous_stride(shape)
    runtime = npu_runtime.get_runtime(_device_index(device))
    stream = npu_state.current_stream(_device_index(device))
    ptr = npu_runtime._alloc_device(max(2 * max(count, 1), 1) * np.dtype(np.int64).itemsize, runtime=runtime)
    aclnn.triu_indices(row, col, offset, ptr, shape, stride, dtype, runtime, stream=stream.stream)
    storage = npu_typed_storage_from_ptr(ptr, max(2 * max(count, 1), 1), dtype, device=device)
    return _wrap_tensor(storage, shape, stride, requires_grad=False)


def _tril_indices_shape(row, col, offset):
    row = int(row)
    col = int(col)
    offset = int(offset)
    count = 0
    for r in range(max(row, 0)):
        max_c = min(col, r + offset + 1)
        if max_c > 0:
            count += max_c
    return (2, count)


def _triu_indices_shape(row, col, offset):
    row = int(row)
    col = int(col)
    offset = int(offset)
    count = 0
    for r in range(max(row, 0)):
        min_c = max(0, r + offset)
        if min_c < col:
            count += col - min_c
    return (2, count)


def eye_out_create(n, m=None, dtype=None, device=None, out=None):
    return eye_create(n, m=m, dtype=dtype, device=device, out=out)


def linspace_out_create(start, end, steps, dtype=None, device=None, out=None):
    if out is None:
        return linspace_create(start, end, steps, dtype=dtype, device=device)
    temp = linspace_create(start, end, steps, dtype=out.dtype, device=out.device)
    out_storage = out.storage()
    temp_storage = temp.storage()
    runtime = npu_runtime.get_runtime(_device_index(out.device))
    npu_runtime.memcpy_d2d(out_storage.data_ptr(), out.numel() * np.dtype(npu_runtime._dtype_to_numpy(out.dtype)).itemsize,
                           temp_storage.data_ptr(), runtime=runtime)
    return out


def arange_out_create(start, end, step=1, dtype=None, device=None, out=None):
    if out is None:
        return arange_create(start, end, step=step, dtype=dtype, device=device)
    temp = arange_create(start, end, step=step, dtype=out.dtype, device=out.device)
    out_storage = out.storage()
    temp_storage = temp.storage()
    runtime = npu_runtime.get_runtime(_device_index(out.device))
    npu_runtime.memcpy_d2d(out_storage.data_ptr(), out.numel() * np.dtype(npu_runtime._dtype_to_numpy(out.dtype)).itemsize,
                           temp_storage.data_ptr(), runtime=runtime)
    return out


def empty_memory_format_create(shape, dtype=None, device=None, memory_format=None):
    return empty_create(shape, dtype=dtype, device=device, requires_grad=False, memory_format=memory_format)


def full_out_create(shape, fill_value, dtype=None, device=None, out=None):
    if out is None:
        return full_create(shape, fill_value, dtype=dtype, device=device)
    temp = full_create(shape, fill_value, dtype=out.dtype, device=out.device)
    out_storage = out.storage()
    temp_storage = temp.storage()
    runtime = npu_runtime.get_runtime(_device_index(out.device))
    npu_runtime.memcpy_d2d(out_storage.data_ptr(), out.numel() * np.dtype(npu_runtime._dtype_to_numpy(out.dtype)).itemsize,
                           temp_storage.data_ptr(), runtime=runtime)
    return out


def _require_inplace_one_zero():
    if not aclnn.ones_zero_symbols_ok():
        raise RuntimeError("aclnnInplaceOne/Zero not available")


def tensor_create(data, dtype=None, device=None, requires_grad=False, memory_format=None):
    arr = np.array(data, dtype=npu_runtime._dtype_to_numpy(dtype))
    runtime = npu_runtime.get_runtime((device.index if hasattr(device, "index") else None) or 0)
    stream = npu_state.current_stream((device.index if hasattr(device, "index") else None) or 0)
    ptr, _ = npu_runtime._copy_cpu_to_npu(arr, runtime=runtime)
    storage = npu_typed_storage_from_ptr(ptr, arr.size, dtype, device=device)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return _wrap_tensor(storage, arr.shape, stride, requires_grad)


def zeros_create(shape, dtype=None, device=None, requires_grad=False, memory_format=None):
    runtime = npu_runtime.get_runtime((device.index if hasattr(device, "index") else None) or 0)
    stream = npu_state.current_stream((device.index if hasattr(device, "index") else None) or 0)
    _require_inplace_one_zero()
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    size = int(np.prod(shape))
    out_size = size * np.dtype(npu_runtime._dtype_to_numpy(dtype)).itemsize
    ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    stride = npu_runtime._contiguous_stride(shape)
    aclnn.inplace_zero(ptr, shape, stride, dtype, runtime, stream=stream.stream)
    storage = npu_typed_storage_from_ptr(ptr, size, dtype, device=device)
    return _wrap_tensor(storage, shape, stride, requires_grad)


def ones_create(shape, dtype=None, device=None, requires_grad=False, memory_format=None):
    runtime = npu_runtime.get_runtime((device.index if hasattr(device, "index") else None) or 0)
    stream = npu_state.current_stream((device.index if hasattr(device, "index") else None) or 0)
    _require_inplace_one_zero()
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    size = int(np.prod(shape))
    out_size = size * np.dtype(npu_runtime._dtype_to_numpy(dtype)).itemsize
    ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    stride = npu_runtime._contiguous_stride(shape)
    aclnn.inplace_one(ptr, shape, stride, dtype, runtime, stream=stream.stream)
    storage = npu_typed_storage_from_ptr(ptr, size, dtype, device=device)
    return _wrap_tensor(storage, shape, stride, requires_grad)


def empty_create(shape, dtype=None, device=None, requires_grad=False, memory_format=None):
    runtime = npu_runtime.get_runtime((device.index if hasattr(device, "index") else None) or 0)
    stream = npu_state.current_stream((device.index if hasattr(device, "index") else None) or 0)
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    size = int(np.prod(shape))
    out_size = size * np.dtype(npu_runtime._dtype_to_numpy(dtype)).itemsize
    ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    storage = npu_typed_storage_from_ptr(ptr, size, dtype, device=device)
    stride = npu_runtime._contiguous_stride(shape)
    return _wrap_tensor(storage, shape, stride, requires_grad)


def randn_create(shape, dtype=None, device=None, requires_grad=False, memory_format=None, generator=None):
    """Create a tensor filled with random numbers from N(0,1) on NPU."""
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    # Create empty NPU tensor, then fill with normal_ (uses ACLNN kernel)
    t = empty_create(shape, dtype=dtype, device=device, requires_grad=requires_grad,
                     memory_format=memory_format)
    from .ops import normal_
    normal_(t, mean=0.0, std=1.0, generator=generator)
    return t


def rand_create(shape, dtype=None, device=None, requires_grad=False, memory_format=None, generator=None):
    """Create a tensor filled with random numbers from U(0,1) on NPU."""
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    t = empty_create(shape, dtype=dtype, device=device, requires_grad=requires_grad,
                     memory_format=memory_format)
    from .ops import uniform_
    uniform_(t, low=0.0, high=1.0, generator=generator)
    return t


def _resolve_dtype(dtype):
    return float32_dtype if dtype is None else dtype


def _contiguous_stride(shape):
    stride = []
    acc = 1
    for d in reversed(shape):
        stride.append(acc)
        acc *= d
    return tuple(reversed(stride))


def _device_index(device):
    return (device.index if hasattr(device, "index") else None) or 0



def _linspace_fallback_npu(start, end, steps, dtype, device):
    from . import ops as npu_ops

    steps = int(steps)
    shape = (steps,)
    if steps == 0:
        return empty_create(shape, dtype=dtype, device=device, requires_grad=False)
    if steps == 1:
        return full_create(shape, start, dtype=dtype, device=device)

    # Build on NPU without calling aclnn.arange (keeps linspace single-op test semantics).
    one = ones_create(shape, dtype=dtype, device=device)
    idx = npu_ops.cumsum(one, dim=0)
    idx = npu_ops.sub(idx, 1)
    step = (float(end) - float(start)) / float(steps - 1)
    idx = npu_ops.mul(idx, step)
    return npu_ops.add(idx, float(start))


def arange_create(start, end, step=1, dtype=None, device=None):
    dtype = _resolve_dtype(dtype)
    if step == 0:
        raise ValueError("step must be nonzero")
    if not aclnn.arange_symbols_ok():
        raise RuntimeError("aclnnArange not available")

    arr = np.arange(start, end, step, dtype=to_numpy_dtype(dtype))
    shape = tuple(arr.shape)
    stride = _contiguous_stride(shape)
    runtime = npu_runtime.get_runtime(_device_index(device))
    stream = npu_state.current_stream(_device_index(device))
    size = int(arr.size)
    itemsize = np.dtype(npu_runtime._dtype_to_numpy(dtype)).itemsize
    ptr = npu_runtime._alloc_device(max(size, 1) * itemsize, runtime=runtime)
    aclnn.arange(start, end, step, ptr, shape, stride, dtype, runtime, stream=stream.stream)
    storage = npu_typed_storage_from_ptr(ptr, max(size, 1), dtype, device=device)
    return _wrap_tensor(storage, shape, stride, requires_grad=False)


def full_create(shape, fill_value, dtype=None, device=None):
    dtype = _resolve_dtype(dtype)
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    base = zeros_create(shape, dtype=dtype, device=device)
    from . import ops as npu_ops

    return npu_ops.add(base, fill_value)


def linspace_create(start, end, steps, dtype=None, device=None):
    if steps < 0:
        raise ValueError("number of steps must be non-negative")
    dtype = _resolve_dtype(dtype)

    if ops_soc.use_smallop_linspace():
        return _linspace_fallback_npu(start, end, steps, dtype=dtype, device=device)

    if not aclnn.linspace_symbols_ok():
        raise RuntimeError("aclnnLinspace not available")

    shape = (int(steps),)
    stride = _contiguous_stride(shape)
    runtime = npu_runtime.get_runtime(_device_index(device))
    stream = npu_state.current_stream(_device_index(device))
    size = int(steps)
    itemsize = np.dtype(npu_runtime._dtype_to_numpy(dtype)).itemsize
    ptr = npu_runtime._alloc_device(max(size, 1) * itemsize, runtime=runtime)
    aclnn.linspace(start, end, steps, ptr, shape, stride, dtype, runtime, stream=stream.stream)
    storage = npu_typed_storage_from_ptr(ptr, max(size, 1), dtype, device=device)
    return _wrap_tensor(storage, shape, stride, requires_grad=False)


def logspace_create(start, end, steps, dtype=None, device=None):
    dtype = _resolve_dtype(dtype)
    linear = linspace_create(start, end, steps, dtype=dtype, device=device)
    from . import ops as npu_ops

    base = full_create(linear.shape, 10.0, dtype=dtype, device=device)
    return npu_ops.pow(base, linear)


def eye_create(n, m=None, dtype=None, device=None, out=None):
    if m is None:
        m = n
    if n < 0:
        raise ValueError(f"n must be greater or equal to 0, got {n}")
    if m < 0:
        raise ValueError(f"m must be greater or equal to 0, got {m}")
    dtype = _resolve_dtype(dtype)
    if not aclnn.eye_symbols_ok():
        raise RuntimeError("aclnnEye not available")

    shape = (int(n), int(m))
    stride = _contiguous_stride(shape)
    runtime = npu_runtime.get_runtime(_device_index(device))
    stream = npu_state.current_stream(_device_index(device))

    if out is not None:
        out_storage = out.storage()
        aclnn.eye(n, m, out_storage.data_ptr(), out.shape, out.stride, out.dtype, runtime, stream=stream.stream)
        return out

    size = int(n) * int(m)
    itemsize = np.dtype(npu_runtime._dtype_to_numpy(dtype)).itemsize
    ptr = npu_runtime._alloc_device(max(size, 1) * itemsize, runtime=runtime)
    aclnn.eye(n, m, ptr, shape, stride, dtype, runtime, stream=stream.stream)
    storage = npu_typed_storage_from_ptr(ptr, max(size, 1), dtype, device=device)
    return _wrap_tensor(storage, shape, stride, requires_grad=False)


def range_create(start, end, step=1, dtype=None, device=None):
    dtype = _resolve_dtype(dtype)
    if not aclnn.range_symbols_ok():
        raise RuntimeError("aclnnRange not available")

    # torch.range is inclusive, so shape is inferred from inclusive end.
    arr = np.arange(start, end + step, step, dtype=to_numpy_dtype(dtype))
    shape = tuple(arr.shape)
    stride = _contiguous_stride(shape)
    runtime = npu_runtime.get_runtime(_device_index(device))
    stream = npu_state.current_stream(_device_index(device))

    # CANN limitation: aclnnRange can fail for float32/int32/int64 after aclnnArange in the same process.
    # Keep single-op path by running aclnnRange in float64 then casting back when needed.
    range_dtype = float64_dtype if dtype.name in ("float32", "int32", "int64") else dtype
    size = int(arr.size)
    itemsize = np.dtype(npu_runtime._dtype_to_numpy(range_dtype)).itemsize
    ptr = npu_runtime._alloc_device(max(size, 1) * itemsize, runtime=runtime)
    aclnn.range_(start, end, step, ptr, shape, stride, range_dtype, runtime, stream=stream.stream)

    if range_dtype != dtype:
        if not aclnn.cast_symbols_ok():
            raise RuntimeError("aclnnCast not available")
        out_itemsize = np.dtype(npu_runtime._dtype_to_numpy(dtype)).itemsize
        out_ptr = npu_runtime._alloc_device(max(size, 1) * out_itemsize, runtime=runtime)
        aclnn.cast(ptr, out_ptr, shape, stride, range_dtype, dtype, runtime, stream=stream.stream)
        runtime.defer_free(ptr)
        ptr = out_ptr

    storage = npu_typed_storage_from_ptr(ptr, max(size, 1), dtype, device=device)
    return _wrap_tensor(storage, shape, stride, requires_grad=False)


def randint_create(low, high=None, size=None, dtype=None, device=None, requires_grad=False, generator=None, **kwargs):
    """Create a tensor filled with random integers from [low, high) on NPU."""
    from ..._dtype import int64 as int64_dtype, float32 as f32
    if high is None:
        low, high = 0, low
    if size is None:
        raise ValueError("size is required for randint")
    size = tuple(size)
    out_dtype = dtype if dtype is not None else int64_dtype
    # Create float tensor, fill uniform [low, high), floor, then cast to int dtype
    t = empty_create(size, dtype=f32, device=device)
    from .ops import uniform_
    uniform_(t, float(low), float(high), generator=generator)
    # floor
    runtime = npu_runtime.get_runtime(_device_index(device))
    stream = npu_state.current_stream(_device_index(device))
    t_storage = t.storage()
    aclnn.floor(t_storage.data_ptr(), t_storage.data_ptr(), t.shape, t.stride, t.dtype, runtime, stream=stream.stream)
    # cast to target int dtype if needed
    if out_dtype != f32:
        if not aclnn.cast_symbols_ok():
            raise RuntimeError("aclnnCast not available")
        numel = int(np.prod(size))
        out_itemsize = np.dtype(npu_runtime._dtype_to_numpy(out_dtype)).itemsize
        out_ptr = npu_runtime._alloc_device(max(numel, 1) * out_itemsize, runtime=runtime)
        out_stride = _contiguous_stride(size)
        aclnn.cast(t_storage.data_ptr(), out_ptr, size, out_stride, f32, out_dtype, runtime, stream=stream.stream)
        out_storage = npu_typed_storage_from_ptr(out_ptr, max(numel, 1), out_dtype, device=device)
        return _wrap_tensor(out_storage, size, out_stride, requires_grad)
    return t


def randperm_create(n, dtype=None, device=None, requires_grad=False, generator=None, **kwargs):
    """Create a random permutation of integers from 0 to n-1 on NPU."""
    from .ops import randperm as randperm_op
    return randperm_op(n, dtype=dtype, device=device, generator=generator)
