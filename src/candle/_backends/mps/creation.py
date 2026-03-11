import ctypes

import numpy as np

from ..._dtype import to_numpy_dtype
from ..._storage import mps_typed_storage_from_numpy
from ..._tensor import Tensor


def _contiguous_stride(shape):
    stride = []
    acc = 1
    for d in reversed(shape):
        stride.append(acc)
        acc *= d
    return tuple(reversed(stride))


def _get_mps_generator(generator=None):
    """Return an MPS Philox generator (user-provided or default)."""
    if generator is not None and hasattr(generator, 'device') and generator.device.type == 'mps':
        return generator
    from ...mps import _get_default_generator
    return _get_default_generator()


def _philox_seed_parts(gen, numel):
    """Get (seed_lo, seed_hi, offset) and advance the generator."""
    increment = (numel + 3) // 4
    seed, offset = gen.philox_engine_inputs(increment)
    return seed & 0xffffffff, (seed >> 32) & 0xffffffff, offset


def tensor_create(data, dtype=None, device=None, requires_grad=False, memory_format=None):
    arr = np.array(data, dtype=to_numpy_dtype(dtype))
    storage = mps_typed_storage_from_numpy(arr.ravel(), dtype, device=device)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return Tensor(storage, arr.shape, stride, requires_grad=requires_grad)


def zeros_create(shape, dtype=None, device=None, requires_grad=False, memory_format=None):
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    arr = np.zeros(shape, dtype=to_numpy_dtype(dtype))
    storage = mps_typed_storage_from_numpy(arr.ravel(), dtype, device=device)
    stride = _contiguous_stride(shape)
    return Tensor(storage, shape, stride, requires_grad=requires_grad)


def ones_create(shape, dtype=None, device=None, requires_grad=False, memory_format=None):
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    arr = np.ones(shape, dtype=to_numpy_dtype(dtype))
    storage = mps_typed_storage_from_numpy(arr.ravel(), dtype, device=device)
    stride = _contiguous_stride(shape)
    return Tensor(storage, shape, stride, requires_grad=requires_grad)


def empty_create(shape, dtype=None, device=None, requires_grad=False, memory_format=None):
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    arr = np.empty(shape, dtype=to_numpy_dtype(dtype))
    storage = mps_typed_storage_from_numpy(arr.ravel(), dtype, device=device)
    stride = _contiguous_stride(shape)
    return Tensor(storage, shape, stride, requires_grad=requires_grad)


def full_create(shape, fill_value, dtype=None, device=None):
    shape = tuple(shape)
    arr = np.full(shape, fill_value, dtype=to_numpy_dtype(dtype))
    storage = mps_typed_storage_from_numpy(arr.ravel(), dtype, device=device)
    stride = _contiguous_stride(shape)
    return Tensor(storage, shape, stride)


def arange_create(start, end, step=1, dtype=None, device=None):
    arr = np.arange(start, end, step, dtype=to_numpy_dtype(dtype))
    storage = mps_typed_storage_from_numpy(arr.ravel(), dtype, device=device)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return Tensor(storage, arr.shape, stride)


def linspace_create(start, end, steps, dtype=None, device=None):
    arr = np.linspace(start, end, steps, dtype=to_numpy_dtype(dtype))
    storage = mps_typed_storage_from_numpy(arr.ravel(), dtype, device=device)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return Tensor(storage, arr.shape, stride)


def logspace_create(start, end, steps, dtype=None, device=None):
    arr = np.logspace(start, end, steps, dtype=to_numpy_dtype(dtype))
    storage = mps_typed_storage_from_numpy(arr.ravel(), dtype, device=device)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return Tensor(storage, arr.shape, stride)


def eye_create(n, m=None, dtype=None, device=None, out=None):
    if m is None:
        m = n
    arr = np.eye(n, m, dtype=to_numpy_dtype(dtype))
    if out is not None:
        out_arr = out._numpy_view()
        out_arr[:] = arr.astype(out_arr.dtype)
        return out
    storage = mps_typed_storage_from_numpy(arr.ravel(), dtype, device=device)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return Tensor(storage, arr.shape, stride)


def range_create(start, end, step=1, dtype=None, device=None):
    arr = np.arange(start, end + step, step, dtype=to_numpy_dtype(dtype))
    storage = mps_typed_storage_from_numpy(arr.ravel(), dtype, device=device)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return Tensor(storage, arr.shape, stride)


def randn_create(shape, dtype=None, device=None, requires_grad=False, memory_format=None, generator=None):
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    from ..._dtype import float32 as f32, float16 as f16
    out_dtype = dtype if dtype is not None else f32
    numel = 1
    for s in shape:
        numel *= s
    if out_dtype in (f32, f16) and numel > 0:
        gen = _get_mps_generator(generator)
        seed_lo, seed_hi, offset = _philox_seed_parts(gen, numel)
        sfx = "f32" if out_dtype == f32 else "f16"
        fmt = "f" if out_dtype == f32 else "e"
        from .ops import _alloc_output_buf, _from_metal_buffer, _get_dispatcher
        out_buf = _alloc_output_buf(numel, out_dtype)
        _get_dispatcher().dispatch_philox_fill(
            f"philox_normal_{sfx}", out_buf, seed_lo, seed_hi, offset,
            0.0, 1.0, numel, param_fmt=fmt)
        stride = _contiguous_stride(shape)
        t = _from_metal_buffer(out_buf, shape, stride, out_dtype, device)
        t.requires_grad = requires_grad
        return t
    from ..._random import _get_cpu_rng
    rng = generator._rng if (generator is not None and hasattr(generator, '_rng') and generator._rng is not None) else _get_cpu_rng()
    arr = rng.randn(*shape).astype(to_numpy_dtype(dtype))
    storage = mps_typed_storage_from_numpy(arr.ravel(), dtype, device=device)
    stride = _contiguous_stride(shape)
    return Tensor(storage, shape, stride, requires_grad=requires_grad)


def rand_create(shape, dtype=None, device=None, requires_grad=False, memory_format=None, generator=None):
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    from ..._dtype import float32 as f32, float16 as f16
    out_dtype = dtype if dtype is not None else f32
    numel = 1
    for s in shape:
        numel *= s
    if out_dtype in (f32, f16) and numel > 0:
        gen = _get_mps_generator(generator)
        seed_lo, seed_hi, offset = _philox_seed_parts(gen, numel)
        sfx = "f32" if out_dtype == f32 else "f16"
        fmt = "f" if out_dtype == f32 else "e"
        from .ops import _alloc_output_buf, _from_metal_buffer, _get_dispatcher
        out_buf = _alloc_output_buf(numel, out_dtype)
        _get_dispatcher().dispatch_philox_fill(
            f"philox_uniform_{sfx}", out_buf, seed_lo, seed_hi, offset,
            0.0, 1.0, numel, param_fmt=fmt)
        stride = _contiguous_stride(shape)
        t = _from_metal_buffer(out_buf, shape, stride, out_dtype, device)
        t.requires_grad = requires_grad
        return t
    from ..._random import _get_cpu_rng
    rng = generator._rng if (generator is not None and hasattr(generator, '_rng') and generator._rng is not None) else _get_cpu_rng()
    arr = rng.random_sample(shape).astype(to_numpy_dtype(dtype))
    storage = mps_typed_storage_from_numpy(arr.ravel(), dtype, device=device)
    stride = _contiguous_stride(shape)
    return Tensor(storage, shape, stride, requires_grad=requires_grad)


def randint_create(low, high=None, size=None, dtype=None, device=None, requires_grad=False, generator=None, **kwargs):
    from ..._dtype import int32 as i32, int64 as i64
    if high is None:
        low, high = 0, low
    if size is None:
        raise ValueError("size is required for randint")
    if isinstance(size, int):
        size = (size,)
    size = tuple(size)
    out_dtype = dtype if dtype is not None else i64
    numel = 1
    for s in size:
        numel *= s
    if out_dtype in (i32, i64) and numel > 0:
        gen = _get_mps_generator(generator)
        seed_lo, seed_hi, offset = _philox_seed_parts(gen, numel)
        sfx = "i32" if out_dtype == i32 else "i64"
        int_fmt = "i" if out_dtype == i32 else "q"
        from .ops import _alloc_output_buf, _from_metal_buffer, _get_dispatcher
        out_buf = _alloc_output_buf(numel, out_dtype)
        _get_dispatcher().dispatch_philox_randint(
            f"philox_randint_{sfx}", out_buf, int(low), int(high),
            seed_lo, seed_hi, offset, numel, int_fmt=int_fmt)
        stride = _contiguous_stride(size)
        t = _from_metal_buffer(out_buf, size, stride, out_dtype, device)
        t.requires_grad = requires_grad
        return t
    from ..._random import _get_cpu_rng
    rng = generator._rng if (generator is not None and hasattr(generator, '_rng') and generator._rng is not None) else _get_cpu_rng()
    arr = rng.randint(int(low), int(high), size=size).astype(np.int64)
    storage = mps_typed_storage_from_numpy(arr.ravel(), out_dtype, device=device)
    stride = _contiguous_stride(size)
    return Tensor(storage, size, stride, requires_grad=requires_grad)


def randperm_create(n, dtype=None, device=None, requires_grad=False, generator=None, **kwargs):
    from ..._dtype import int64 as i64, float32 as f32
    n = int(n)
    out_dtype = dtype if dtype is not None else i64
    if n > 0:
        # Generate n uniform floats on GPU, argsort on CPU, copy back
        gen = _get_mps_generator(generator)
        seed_lo, seed_hi, offset = _philox_seed_parts(gen, n)
        from .ops import _alloc_output_buf, _get_dispatcher
        from .runtime import buffer_contents
        float_buf = _alloc_output_buf(n, f32)
        _get_dispatcher().dispatch_philox_fill(
            "philox_uniform_f32", float_buf, seed_lo, seed_hi, offset,
            0.0, 1.0, n, param_fmt="f")
        ptr = buffer_contents(float_buf)
        float_arr = np.frombuffer(
            (ctypes.c_uint8 * (n * 4)).from_address(ptr),
            dtype=np.float32, count=n)
        indices = np.argsort(float_arr).astype(to_numpy_dtype(out_dtype))
        storage = mps_typed_storage_from_numpy(indices, out_dtype, device=device)
        stride = _contiguous_stride(indices.shape)
        return Tensor(storage, indices.shape, stride, requires_grad=requires_grad)
    arr = np.array([], dtype=to_numpy_dtype(out_dtype))
    storage = mps_typed_storage_from_numpy(arr, out_dtype, device=device)
    stride = _contiguous_stride(arr.shape)
    return Tensor(storage, arr.shape, stride, requires_grad=requires_grad)
