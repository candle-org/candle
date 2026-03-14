import math
import ctypes
import struct
import numpy as np

from ...._dtype import bool as bool_dtype
from ...._dtype import int32 as int32_dtype
from ...._dtype import int64 as int64_dtype
from ...._dtype import float16 as float16_dtype
from ...._dtype import float32 as float32_dtype
from ...._dtype import float64 as float64_dtype
from ...._dtype import to_numpy_dtype
from ...._storage import mps_typed_storage_from_numpy, _MPSUntypedStorage, TypedStorage
from ...._tensor import Tensor
from .. import accelerate as _accel

# ---------------------------------------------------------------------------
# GPU dispatch helpers
# ---------------------------------------------------------------------------
_GPU_DTYPES = frozenset({float32_dtype, float16_dtype, int32_dtype, int64_dtype, bool_dtype})


def _can_use_gpu(t):
    """Check if tensor can use Metal GPU kernels."""
    return (t.dtype in _GPU_DTYPES
            and t.numel() > 0
            and hasattr(t._storage, '_untyped')
            and hasattr(t._storage._untyped, '_metal_buffer')
            and t._storage._untyped._metal_buffer is not None)


def _metal_buf(t):
    """Get the raw Metal buffer from a tensor."""
    return t._storage._untyped._metal_buffer


def _kernel_suffix(dtype):
    """Return MSL kernel suffix for dtype."""
    _SUFFIX = {
        float32_dtype: "f32", float16_dtype: "f16",
        int32_dtype: "i32", int64_dtype: "i64",
        bool_dtype: "u8",
    }
    return _SUFFIX[dtype]


def _scalar_fmt(dtype):
    """Return struct format char for scalar encoding."""
    _FMT = {
        float32_dtype: "f", float16_dtype: "e",
        int32_dtype: "i", int64_dtype: "q",
        bool_dtype: "B",
    }
    return _FMT[dtype]


def _itemsize(dtype):
    """Return byte size per element."""
    _SIZES = {
        float32_dtype: 4, float16_dtype: 2,
        int32_dtype: 4, int64_dtype: 8,
        bool_dtype: 1,
    }
    return _SIZES.get(dtype, dtype.itemsize)


def _alloc_output_buf(numel, dtype):
    """Allocate a Metal buffer for output."""
    from ..runtime import get_runtime
    rt = get_runtime()
    return rt.create_buffer(numel * _itemsize(dtype))


def _metal_buf_to_bytes(metal_buf, nbytes):
    """Read raw bytes from a Metal buffer."""
    from ..runtime import buffer_contents
    ptr = buffer_contents(metal_buf)
    return bytes((ctypes.c_char * nbytes).from_address(ptr))


def _from_metal_buffer(metal_buf, shape, stride, dtype, device):
    """Wrap an existing Metal buffer into a Tensor without copying data."""
    from ..runtime import buffer_contents
    nbytes = 1
    for s in shape:
        nbytes *= s
    nbytes *= _itemsize(dtype)
    nbytes = max(nbytes, 1)
    untyped = _MPSUntypedStorage(metal_buf, nbytes, device=device)
    ptr = buffer_contents(metal_buf)
    numel = 1
    for s in shape:
        numel *= s
    data = np.frombuffer(
        (ctypes.c_uint8 * nbytes).from_address(ptr),
        dtype=to_numpy_dtype(dtype),
        count=max(numel, 1),
    )
    size = max(numel, 1)
    storage = TypedStorage(untyped, dtype, size, data=data)
    return Tensor(storage, shape, stride)


def _get_dispatcher():
    """Lazy import of the Metal kernel dispatcher singleton."""
    from ..metal_compute import get_dispatcher
    return get_dispatcher()


def _dispatch_unary_gpu(a, kernel_base):
    """Dispatch a unary GPU kernel, choosing contiguous or strided variant."""
    d = _get_dispatcher()
    sfx = _kernel_suffix(a.dtype)
    numel = a.numel()
    out_buf = _alloc_output_buf(numel, a.dtype)
    if a.is_contiguous():
        d.dispatch_unary(f"{kernel_base}_{sfx}", _metal_buf(a), out_buf, numel)
    else:
        d.dispatch_unary_strided(
            f"{kernel_base}_strided_{sfx}", _metal_buf(a), out_buf, numel,
            list(a.shape), list(a.stride), len(a.shape))
    from ...._tensor import _compute_strides
    out_shape = tuple(a.shape)
    out_stride = _compute_strides(out_shape)
    return _from_metal_buffer(out_buf, out_shape, out_stride, a.dtype, a.device)


def _dispatch_unary_predicate_gpu(a, kernel_base):
    """Dispatch a unary predicate GPU kernel (float → bool), contiguous or strided."""
    d = _get_dispatcher()
    sfx = _kernel_suffix(a.dtype)
    numel = a.numel()
    out_buf = _alloc_output_buf(numel, bool_dtype)
    if a.is_contiguous():
        d.dispatch_unary(f"{kernel_base}_{sfx}", _metal_buf(a), out_buf, numel)
    else:
        d.dispatch_unary_strided(
            f"{kernel_base}_strided_{sfx}", _metal_buf(a), out_buf, numel,
            list(a.shape), list(a.stride), len(a.shape))
    from ...._tensor import _compute_strides
    out_shape = tuple(a.shape)
    out_stride = _compute_strides(out_shape)
    return _from_metal_buffer(out_buf, out_shape, out_stride, bool_dtype, a.device)


def _scalar_value(val, dtype):
    """Convert a scalar to the appropriate Python type for the given dtype."""
    if dtype in (int32_dtype, int64_dtype, bool_dtype):
        return int(val)
    return float(val)


def _dispatch_binary_gpu(a, b, kernel_base):
    """Dispatch a binary GPU kernel, choosing contiguous or strided variant."""
    d = _get_dispatcher()
    sfx = _kernel_suffix(a.dtype)
    numel = a.numel()
    out_buf = _alloc_output_buf(numel, a.dtype)
    if isinstance(b, Tensor) and _can_use_gpu(b) and a.shape == b.shape:
        if a.is_contiguous() and b.is_contiguous():
            d.dispatch_binary(f"{kernel_base}_{sfx}", _metal_buf(a),
                              _metal_buf(b), out_buf, numel)
        else:
            d.dispatch_binary_strided(
                f"{kernel_base}_strided_{sfx}", _metal_buf(a), _metal_buf(b),
                out_buf, numel, list(a.shape), list(a.stride),
                list(b.stride), len(a.shape))
    else:
        raw = float(b) if not isinstance(b, Tensor) else float(_to_numpy(b).ravel()[0])
        scalar = _scalar_value(raw, a.dtype)
        if a.is_contiguous():
            d.dispatch_binary_scalar(f"{kernel_base}_scalar_{sfx}",
                                     _metal_buf(a), scalar, out_buf, numel,
                                     scalar_fmt=_scalar_fmt(a.dtype))
        else:
            d.dispatch_binary_scalar_strided(
                f"{kernel_base}_scalar_strided_{sfx}", _metal_buf(a), scalar,
                out_buf, numel, list(a.shape), list(a.stride),
                len(a.shape), scalar_fmt=_scalar_fmt(a.dtype))
    from ...._tensor import _compute_strides
    out_shape = tuple(a.shape)
    out_stride = _compute_strides(out_shape)
    return _from_metal_buffer(out_buf, out_shape, out_stride, a.dtype, a.device)


def _to_numpy(t):
    return t._numpy_view()


def _compute_reduce_dims(shape, dim):
    """Compute (outer_size, reduce_size, inner_size) for axis reduction."""
    ndim = len(shape)
    if isinstance(dim, int):
        dim = (dim,)
    dims = tuple(d % ndim for d in dim)
    outer = 1
    reduce = 1
    inner = 1
    # Find contiguous range or single dim
    sorted_dims = sorted(dims)
    for i in range(ndim):
        if i < sorted_dims[0]:
            outer *= shape[i]
        elif i in sorted_dims:
            reduce *= shape[i]
        else:
            inner *= shape[i]
    return outer, reduce, inner


def _reduce_shape(shape, dim, keepdim):
    """Compute output shape after reduction."""
    ndim = len(shape)
    if isinstance(dim, int):
        dim = (dim,)
    dims = set(d % ndim for d in dim)
    if keepdim:
        return tuple(1 if i in dims else s for i, s in enumerate(shape))
    return tuple(s for i, s in enumerate(shape) if i not in dims)


def _gpu_reduce_single_dim(a, dim, op_name, keepdim):
    """Reduce a single dimension on GPU using axis-reduce kernels."""
    d = _get_dispatcher()
    sfx = _kernel_suffix(a.dtype)
    ndim = len(a.shape)
    dim = dim % ndim

    outer = 1
    for i in range(dim):
        outer *= a.shape[i]
    reduce_size = a.shape[dim]
    inner = 1
    for i in range(dim + 1, ndim):
        inner *= a.shape[i]

    out_numel = outer * inner
    # argmax/argmin output uint, stored as int64
    if op_name in ("argmax", "argmin"):
        out_buf = _alloc_output_buf(out_numel, int32_dtype)
        out_dtype = int64_dtype
    elif op_name in ("any", "all"):
        out_buf = _alloc_output_buf(out_numel, bool_dtype)
        out_dtype = bool_dtype
    else:
        out_buf = _alloc_output_buf(out_numel, a.dtype)
        out_dtype = a.dtype

    kernel = f"reduce_{op_name}_dim_{sfx}"
    d.dispatch_reduce_dim(kernel, _metal_buf(a), out_buf,
                          outer, reduce_size, inner, out_numel)

    out_shape = _reduce_shape(a.shape, dim, keepdim)
    from ...._tensor import _compute_strides
    out_stride = _compute_strides(out_shape)

    if op_name in ("argmax", "argmin"):
        # Read uint32 results, convert to int64 numpy
        from ..runtime import buffer_contents
        ptr = buffer_contents(out_buf)
        raw = (ctypes.c_char * (out_numel * 4)).from_address(ptr)
        arr = np.frombuffer(bytes(raw), dtype=np.uint32, count=out_numel)
        arr = arr.astype(np.int64).reshape(out_shape)
        return _from_numpy(np.ascontiguousarray(arr), int64_dtype, a.device)

    return _from_metal_buffer(out_buf, out_shape, out_stride, out_dtype, a.device)


def _normalize_tensor_sequence_args(tensors):
    if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
        return tuple(tensors[0])
    return tuple(tensors)


def _from_numpy(arr, dtype, device):
    storage = mps_typed_storage_from_numpy(arr, dtype, device=device)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return Tensor(storage, arr.shape, stride)


def _can_use_blas(arr):
    """Check if array is contiguous float32 or float64 for BLAS."""
    return (arr.flags['C_CONTIGUOUS'] and
            arr.dtype in (np.float32, np.float64) and
            _accel.available())


def _blas_gemm(a_np, b_np, dtype):
    """Matrix-matrix multiply via Accelerate BLAS."""
    M, K = a_np.shape
    K2, N = b_np.shape
    out = np.empty((M, N), dtype=a_np.dtype)
    if a_np.dtype == np.float32:
        _accel.cblas_sgemm(111, 111, M, N, K, 1.0,
                           a_np.ctypes.data, K, b_np.ctypes.data, N,
                           0.0, out.ctypes.data, N)
    else:
        _accel.cblas_dgemm(111, 111, M, N, K, 1.0,
                           a_np.ctypes.data, K, b_np.ctypes.data, N,
                           0.0, out.ctypes.data, N)
    return out

