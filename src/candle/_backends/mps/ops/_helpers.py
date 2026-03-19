import ctypes
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

from candle._cython._mps_helpers import (  # pylint: disable=import-error,no-name-in-module
    can_use_gpu as _cy_can_use_gpu,
    dispatch_unary_gpu as _cy_dispatch_unary_gpu,
    dispatch_unary_predicate_gpu as _cy_dispatch_unary_predicate_gpu,
    dispatch_binary_gpu as _cy_dispatch_binary_gpu,
    from_metal_buffer as _cy_from_metal_buffer,
    alloc_output_buf as _cy_alloc_output_buf,
    get_metal_buf as _cy_get_metal_buf,
    kernel_suffix as _cy_kernel_suffix,
    scalar_fmt as _cy_scalar_fmt,
    itemsize as _cy_itemsize,
    compute_reduce_dims as _cy_compute_reduce_dims,
    reduce_shape as _cy_reduce_shape,
)

# ---------------------------------------------------------------------------
# GPU dispatch helpers
# ---------------------------------------------------------------------------
_GPU_DTYPES = frozenset({float32_dtype, float16_dtype, int32_dtype, int64_dtype, bool_dtype})


def _can_use_gpu(t):
    """Check if tensor can use Metal GPU kernels."""
    return _cy_can_use_gpu(t)


def _empty_like(t):
    """Return an empty contiguous tensor with the same shape/dtype/device."""
    from ...._tensor import _compute_strides
    shape = tuple(t.shape)
    stride = tuple(_compute_strides(shape))
    buf = _cy_alloc_output_buf(max(t.numel(), 1), t.dtype)
    return _cy_from_metal_buffer(buf, shape, stride, t.dtype, t.device)


def _unsupported_dtype(op_name, t):
    """Raise TypeError for unsupported MPS dtype."""
    raise TypeError(
        f"MPS {op_name}: unsupported dtype {t.dtype}. "
        f"Supported: float32, float16, int32, int64, bool"
    )


def _metal_buf(t):
    """Get the raw Metal buffer from a tensor."""
    return _cy_get_metal_buf(t)


def _kernel_suffix(dtype):
    """Return MSL kernel suffix for dtype."""
    return _cy_kernel_suffix(dtype)


def _scalar_fmt(dtype):
    """Return struct format char for scalar encoding."""
    return _cy_scalar_fmt(dtype)


def _itemsize(dtype):
    """Return byte size per element."""
    return _cy_itemsize(dtype)


def _alloc_output_buf(numel, dtype):
    """Allocate a Metal buffer for output."""
    return _cy_alloc_output_buf(numel, dtype)


def _metal_buf_to_bytes(metal_buf, nbytes):
    """Read raw bytes from a Metal buffer."""
    from ..runtime import buffer_contents
    ptr = buffer_contents(metal_buf)
    return bytes((ctypes.c_char * nbytes).from_address(ptr))


def _read_scalar(t):
    """Read a single scalar from a GPU tensor without numpy."""
    import struct
    from ..runtime import buffer_contents
    nbytes = _cy_itemsize(t.dtype)
    ptr = buffer_contents(_cy_get_metal_buf(t))
    raw = bytes((ctypes.c_char * nbytes).from_address(ptr))
    return struct.unpack(_cy_scalar_fmt(t.dtype), raw)[0]


def _from_metal_buffer(metal_buf, shape, stride, dtype, device):
    """Wrap an existing Metal buffer into a Tensor without copying data."""
    return _cy_from_metal_buffer(metal_buf, tuple(shape), tuple(stride), dtype, device)


def _get_dispatcher():
    """Lazy import of the Metal kernel dispatcher singleton."""
    from ..metal_compute import get_dispatcher
    return get_dispatcher()


def _dispatch_unary_gpu(a, kernel_base):
    """Dispatch a unary GPU kernel, choosing contiguous or strided variant."""
    return _cy_dispatch_unary_gpu(a, kernel_base)


def _dispatch_unary_predicate_gpu(a, kernel_base):
    """Dispatch a unary predicate GPU kernel (float -> bool), contiguous or strided."""
    return _cy_dispatch_unary_predicate_gpu(a, kernel_base)


def _scalar_value(val, dtype):
    """Convert a scalar to the appropriate Python type for the given dtype."""
    if dtype in (int32_dtype, int64_dtype, bool_dtype):
        return int(val)
    return float(val)


def _dispatch_binary_gpu(a, b, kernel_base):
    """Dispatch a binary GPU kernel, choosing contiguous or strided variant."""
    return _cy_dispatch_binary_gpu(a, b, kernel_base)


def _to_numpy(t):
    return t._numpy_view()


def _compute_reduce_dims(shape, dim):
    """Compute (outer_size, reduce_size, inner_size) for axis reduction."""
    return _cy_compute_reduce_dims(tuple(shape), dim)


def _reduce_shape(shape, dim, keepdim):
    """Compute output shape after reduction."""
    return _cy_reduce_shape(tuple(shape), dim, keepdim)


def _gpu_reduce_single_dim(a, dim, op_name, keepdim):
    """Reduce a single dimension on GPU using axis-reduce kernels."""
    d = _get_dispatcher()
    sfx = _cy_kernel_suffix(a.dtype)
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
    if op_name in ("argmax", "argmin"):
        out_buf = _cy_alloc_output_buf(out_numel, int32_dtype)
        out_dtype = int64_dtype
    elif op_name in ("any", "all"):
        out_buf = _cy_alloc_output_buf(out_numel, bool_dtype)
        out_dtype = bool_dtype
    else:
        out_buf = _cy_alloc_output_buf(out_numel, a.dtype)
        out_dtype = a.dtype

    kernel = f"reduce_{op_name}_dim_{sfx}"
    d.dispatch_reduce_dim(kernel, _cy_get_metal_buf(a), out_buf,
                          outer, reduce_size, inner, out_numel)

    out_shape = _cy_reduce_shape(tuple(a.shape), dim, keepdim)
    from ...._tensor import _compute_strides
    out_stride = tuple(_compute_strides(out_shape))

    if op_name in ("argmax", "argmin"):
        from ..runtime import buffer_contents
        ptr = buffer_contents(out_buf)
        raw = (ctypes.c_char * (out_numel * 4)).from_address(ptr)
        arr = np.frombuffer(bytes(raw), dtype=np.uint32, count=out_numel)
        arr = arr.astype(np.int64).reshape(out_shape)
        return _from_numpy(np.ascontiguousarray(arr), int64_dtype, a.device)

    return _cy_from_metal_buffer(out_buf, out_shape, out_stride, out_dtype, a.device)


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
