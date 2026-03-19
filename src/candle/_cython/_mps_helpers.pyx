# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython hot-path helpers for the MPS (Metal) backend.

Accelerates the metadata-computation layer that sits between every MPS op
and the Metal dispatcher:

  can_use_gpu           — dtype + buffer guard (called on every op)
  kernel_suffix         — dtype -> MSL suffix string
  scalar_fmt            — dtype -> struct format char
  itemsize              — dtype -> bytes per element
  dispatch_unary_gpu    — full unary dispatch (alloc + kernel name + wrap)
  dispatch_unary_predicate_gpu — unary dispatch with bool output
  dispatch_binary_gpu   — full binary dispatch (contiguous + strided + scalar)
  from_metal_buffer     — wrap a raw Metal buffer into a Tensor
  compute_reduce_dims   — (outer, reduce, inner) for axis reduction
  reduce_shape          — output shape after reduction

Python fallback: ``candle._backends.mps.ops._helpers`` is unchanged.
"""

from libc.stdint cimport int64_t

import ctypes
import struct
import numpy as np

# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------
cdef object _float32_dtype = None
cdef object _float16_dtype = None
cdef object _int32_dtype   = None
cdef object _int64_dtype   = None
cdef object _bool_dtype    = None
cdef object _GPU_DTYPES    = None

cdef dict _SUFFIX = {}
cdef dict _FMT    = {}
cdef dict _SIZES  = {}
cdef dict _NP_DTYPE = {}

cdef object _compute_strides_fn              = None
cdef object _mps_typed_storage_from_numpy_fn = None
cdef object _MPSUntypedStorage_cls           = None
cdef object _TypedStorage_cls                = None
cdef object _Tensor_cls                      = None
cdef object _buffer_contents_fn              = None
cdef object _get_runtime_fn                  = None
cdef object _get_dispatcher_fn               = None


cdef inline void _ensure_dtypes():
    global _float32_dtype, _float16_dtype, _int32_dtype, _int64_dtype
    global _bool_dtype, _GPU_DTYPES, _SUFFIX, _FMT, _SIZES, _NP_DTYPE
    if _GPU_DTYPES is not None:
        return
    from candle._dtype import float32 as f32, float16 as f16
    from candle._dtype import int32 as i32, int64 as i64
    from candle._dtype import bool as bdt
    _float32_dtype = f32
    _float16_dtype = f16
    _int32_dtype   = i32
    _int64_dtype   = i64
    _bool_dtype    = bdt
    _GPU_DTYPES = frozenset({f32, f16, i32, i64, bdt})
    _SUFFIX  = {f32: "f32", f16: "f16", i32: "i32", i64: "i64", bdt: "u8"}
    _FMT     = {f32: "f",   f16: "e",   i32: "i",   i64: "q",   bdt: "B"}
    _SIZES   = {f32: 4,     f16: 2,     i32: 4,     i64: 8,     bdt: 1}
    _NP_DTYPE = {
        f32: np.float32, f16: np.float16,
        i32: np.int32,   i64: np.int64,
        bdt: np.uint8,
    }


cdef inline void _ensure_runtime():
    global _compute_strides_fn, _mps_typed_storage_from_numpy_fn
    global _MPSUntypedStorage_cls, _TypedStorage_cls, _Tensor_cls
    global _buffer_contents_fn, _get_runtime_fn, _get_dispatcher_fn
    if _compute_strides_fn is not None:
        return
    from candle._tensor import _compute_strides, Tensor
    _compute_strides_fn = _compute_strides
    _Tensor_cls = Tensor
    from candle._storage import (
        mps_typed_storage_from_numpy, _MPSUntypedStorage, TypedStorage,
    )
    _mps_typed_storage_from_numpy_fn = mps_typed_storage_from_numpy
    _MPSUntypedStorage_cls = _MPSUntypedStorage
    _TypedStorage_cls = TypedStorage
    from candle._backends.mps.runtime import get_runtime, buffer_contents
    _get_runtime_fn = get_runtime
    _buffer_contents_fn = buffer_contents
    from candle._backends.mps.metal_compute import get_dispatcher
    _get_dispatcher_fn = get_dispatcher


# ---------------------------------------------------------------------------
# C-level guard (inlined, not exposed to Python)
# ---------------------------------------------------------------------------

cdef inline bint _can_use_gpu_c(object t):
    _ensure_dtypes()
    if t.dtype not in _GPU_DTYPES:
        return False
    if t.numel() <= 0:
        return False
    cdef object storage = t._storage
    if not hasattr(storage, '_untyped'):
        return False
    cdef object untyped = storage._untyped
    if not hasattr(untyped, '_metal_buffer'):
        return False
    return untyped._metal_buffer is not None


# ---------------------------------------------------------------------------
# Public helpers (cpdef)
# ---------------------------------------------------------------------------

cpdef bint can_use_gpu(object t):
    """Return True if tensor *t* can use Metal GPU kernels."""
    return _can_use_gpu_c(t)


cpdef str kernel_suffix(object dtype):
    """Return the MSL kernel suffix string for *dtype*."""
    _ensure_dtypes()
    return _SUFFIX[dtype]


cpdef str scalar_fmt(object dtype):
    """Return the struct format character for *dtype*."""
    _ensure_dtypes()
    return _FMT[dtype]


cpdef int itemsize(object dtype):
    """Return byte size per element for *dtype*."""
    _ensure_dtypes()
    try:
        return _SIZES[dtype]
    except KeyError:
        return dtype.itemsize


cpdef object get_metal_buf(object t):
    """Return the raw Metal buffer from tensor *t*."""
    return t._storage._untyped._metal_buffer


cpdef object alloc_output_buf(int numel, object dtype):
    """Allocate a new Metal buffer for *numel* elements of *dtype*."""
    _ensure_dtypes()
    _ensure_runtime()
    cdef object rt = _get_runtime_fn()
    return rt.create_buffer(numel * itemsize(dtype))


cpdef object from_metal_buffer(object metal_buf_obj, object shape, object stride,
                               object dtype, object device):
    """Wrap an existing Metal buffer into a Tensor without copying."""
    _ensure_dtypes()
    _ensure_runtime()
    cdef tuple shape_t  = tuple(shape)
    cdef tuple stride_t = tuple(stride)
    cdef int nbytes = 1
    cdef int numel  = 1
    for s in shape_t:
        nbytes *= s
        numel  *= s
    nbytes *= itemsize(dtype)
    if nbytes < 1:
        nbytes = 1
    if numel < 1:
        numel = 1

    cdef object untyped = _MPSUntypedStorage_cls(metal_buf_obj, nbytes, device=device)
    cdef long ptr = _buffer_contents_fn(metal_buf_obj)
    cdef object np_dtype = _NP_DTYPE.get(dtype, np.float32)
    cdef object data = np.frombuffer(
        (ctypes.c_uint8 * nbytes).from_address(ptr),
        dtype=np_dtype,
        count=numel,
    )
    cdef object typed_storage = _TypedStorage_cls(untyped, dtype, numel, data=data)
    return _Tensor_cls(typed_storage, shape_t, stride_t)


cdef inline object _read_scalar_c(object t):
    """Read a single scalar value from a Metal buffer."""
    _ensure_dtypes()
    _ensure_runtime()
    cdef int nb   = itemsize(t.dtype)
    cdef long ptr = _buffer_contents_fn(t._storage._untyped._metal_buffer)
    cdef bytes raw = bytes((ctypes.c_char * nb).from_address(ptr))
    return struct.unpack(_FMT[t.dtype], raw)[0]


cdef inline object _scalar_value_c(object val, object dtype):
    """Cast *val* to the right Python scalar type for *dtype*."""
    _ensure_dtypes()
    if dtype is _int32_dtype or dtype is _int64_dtype or dtype is _bool_dtype:
        return int(val)
    return float(val)


# ---------------------------------------------------------------------------
# Unary dispatch
# ---------------------------------------------------------------------------

cpdef object dispatch_unary_gpu(object a, str kernel_base):
    """Dispatch a unary GPU kernel. Caller must verify can_use_gpu(a)."""
    _ensure_dtypes()
    _ensure_runtime()
    cdef object d       = _get_dispatcher_fn()
    cdef str sfx        = _SUFFIX[a.dtype]
    cdef int numel      = a.numel()
    cdef object out_buf = alloc_output_buf(numel, a.dtype)
    cdef object mb      = a._storage._untyped._metal_buffer

    if a.is_contiguous():
        d.dispatch_unary(f"{kernel_base}_{sfx}", mb, out_buf, numel)
    else:
        d.dispatch_unary_strided(
            f"{kernel_base}_strided_{sfx}", mb, out_buf, numel,
            list(a.shape), list(a.stride), len(a.shape),
        )

    cdef tuple out_shape  = tuple(a.shape)
    cdef tuple out_stride = tuple(_compute_strides_fn(out_shape))
    return from_metal_buffer(out_buf, out_shape, out_stride, a.dtype, a.device)


cpdef object dispatch_unary_predicate_gpu(object a, str kernel_base):
    """Dispatch a unary predicate kernel (bool output). Caller must verify can_use_gpu(a)."""
    _ensure_dtypes()
    _ensure_runtime()
    cdef object d       = _get_dispatcher_fn()
    cdef str sfx        = _SUFFIX[a.dtype]
    cdef int numel      = a.numel()
    cdef object out_buf = alloc_output_buf(numel, _bool_dtype)
    cdef object mb      = a._storage._untyped._metal_buffer

    if a.is_contiguous():
        d.dispatch_unary(f"{kernel_base}_{sfx}", mb, out_buf, numel)
    else:
        d.dispatch_unary_strided(
            f"{kernel_base}_strided_{sfx}", mb, out_buf, numel,
            list(a.shape), list(a.stride), len(a.shape),
        )

    cdef tuple out_shape  = tuple(a.shape)
    cdef tuple out_stride = tuple(_compute_strides_fn(out_shape))
    return from_metal_buffer(out_buf, out_shape, out_stride, _bool_dtype, a.device)


# ---------------------------------------------------------------------------
# Binary dispatch
# ---------------------------------------------------------------------------

cpdef object dispatch_binary_gpu(object a, object b, str kernel_base):
    """Dispatch a binary GPU kernel (tensor*tensor or tensor*scalar).

    Handles contiguous, strided, and scalar-rhs cases.
    Caller must verify can_use_gpu(a).
    """
    _ensure_dtypes()
    _ensure_runtime()
    cdef object d       = _get_dispatcher_fn()
    cdef str sfx        = _SUFFIX[a.dtype]
    cdef int numel      = a.numel()
    cdef object out_buf = alloc_output_buf(numel, a.dtype)
    cdef object mb_a    = a._storage._untyped._metal_buffer
    cdef str fmt        = _FMT[a.dtype]
    cdef object mb_b
    cdef object raw_val
    cdef object scalar

    if (isinstance(b, _Tensor_cls)
            and _can_use_gpu_c(b)
            and a.shape == b.shape):
        mb_b = b._storage._untyped._metal_buffer
        if a.is_contiguous() and b.is_contiguous():
            d.dispatch_binary(f"{kernel_base}_{sfx}", mb_a, mb_b, out_buf, numel)
        else:
            d.dispatch_binary_strided(
                f"{kernel_base}_strided_{sfx}", mb_a, mb_b, out_buf, numel,
                list(a.shape), list(a.stride), list(b.stride), len(a.shape),
            )
    else:
        if isinstance(b, _Tensor_cls):
            raw_val = _read_scalar_c(b)
        else:
            raw_val = b
        scalar = _scalar_value_c(raw_val, a.dtype)
        if a.is_contiguous():
            d.dispatch_binary_scalar(
                f"{kernel_base}_scalar_{sfx}", mb_a, scalar, out_buf, numel,
                scalar_fmt=fmt,
            )
        else:
            d.dispatch_binary_scalar_strided(
                f"{kernel_base}_scalar_strided_{sfx}", mb_a, scalar,
                out_buf, numel,
                list(a.shape), list(a.stride), len(a.shape),
                scalar_fmt=fmt,
            )

    cdef tuple out_shape  = tuple(a.shape)
    cdef tuple out_stride = tuple(_compute_strides_fn(out_shape))
    return from_metal_buffer(out_buf, out_shape, out_stride, a.dtype, a.device)


# ---------------------------------------------------------------------------
# Reduction shape helpers
# ---------------------------------------------------------------------------

cpdef tuple compute_reduce_dims(tuple shape, object dim):
    """Return (outer_size, reduce_size, inner_size) for single-axis reduction."""
    cdef int ndim = len(shape)
    cdef list dims_list
    cdef int d
    if isinstance(dim, int):
        dims_list = [<int>dim % ndim]
    else:
        dims_list = [<int>d % ndim for d in dim]
    cdef list sorted_dims = sorted(dims_list)
    cdef set dims_set = set(dims_list)

    cdef int64_t outer  = 1
    cdef int64_t reduce = 1
    cdef int64_t inner  = 1
    cdef int i
    for i in range(ndim):
        if i < sorted_dims[0]:
            outer  *= shape[i]
        elif i in dims_set:
            reduce *= shape[i]
        else:
            inner  *= shape[i]
    return (outer, reduce, inner)


cpdef tuple reduce_shape(tuple shape, object dim, bint keepdim):
    """Return output shape after reducing *dim* from *shape*."""
    cdef int ndim = len(shape)
    cdef list dims_list
    cdef int d
    if isinstance(dim, int):
        dims_list = [<int>dim % ndim]
    else:
        dims_list = [<int>d % ndim for d in dim]
    cdef set dims_set = set(dims_list)
    cdef list out = []
    cdef int i
    cdef int s
    for i in range(ndim):
        s = shape[i]
        if keepdim:
            out.append(1 if i in dims_set else s)
        elif i not in dims_set:
            out.append(s)
    return tuple(out)
