# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython fast-path for _binary_op helper.

Replaces Python-level shape computation (broadcast, stride, numel) with
C-level loops, reducing per-op overhead by ~0.05-0.10ms on the hot path.

The heavy operations (device malloc, aclnn kernel, output wrapping) remain
in Python — this module only accelerates the metadata computation.
"""

from libc.stdint cimport int64_t, int32_t, uint64_t
from libc.stdint cimport uintptr_t
from candle._cython._tensor_impl cimport TensorImpl
from candle._cython._storage_impl cimport StorageImpl
import importlib

DEF MAX_NDIM = 16

# ---------------------------------------------------------------------------
# C-level shape utilities (nogil)
# ---------------------------------------------------------------------------

cdef int c_broadcast_shape(
    const int64_t* a, int a_ndim,
    const int64_t* b, int b_ndim,
    int64_t* out) except -1 nogil:
    """Compute broadcast shape into *out*.  Returns out_ndim.

    Raises ValueError (via except -1) on shape mismatch.
    """
    cdef int out_ndim = a_ndim if a_ndim > b_ndim else b_ndim
    cdef int i
    cdef int64_t a_dim, b_dim
    # Fill from the right (index out_ndim-1 down to 0)
    for i in range(out_ndim):
        a_dim = a[a_ndim - 1 - i] if i < a_ndim else 1
        b_dim = b[b_ndim - 1 - i] if i < b_ndim else 1
        if a_dim == b_dim:
            out[out_ndim - 1 - i] = a_dim
        elif a_dim == 1:
            out[out_ndim - 1 - i] = b_dim
        elif b_dim == 1:
            out[out_ndim - 1 - i] = a_dim
        else:
            with gil:
                raise ValueError("broadcast shape mismatch")
            return -1  # unreachable, keeps compiler happy
    return out_ndim


cdef void c_contiguous_stride(
    const int64_t* shape, int ndim, int64_t* out) noexcept nogil:
    """Compute contiguous strides in-place."""
    cdef int64_t acc = 1
    cdef int i, j
    # Iterate forward using (ndim - 1 - j) to avoid negative index
    for j in range(ndim):
        i = ndim - 1 - j
        out[i] = acc
        acc = acc * shape[i]


cdef int64_t c_numel(const int64_t* shape, int ndim) nogil:
    """Product of shape dims."""
    cdef int64_t n = 1
    cdef int i
    for i in range(ndim):
        n = n * shape[i]
    return n


# ---------------------------------------------------------------------------
# dtype itemsize (C switch, no dict lookup)
# ---------------------------------------------------------------------------

cdef int c_dtype_itemsize(object dtype):
    """Return byte size from a candle dtype object."""
    cdef object size = getattr(dtype, "itemsize", None)
    if size is not None:
        return <int>size
    cdef str name = getattr(dtype, "name", None)
    if name is None:
        s = str(dtype)
        parts = s.split(".")
        name = parts[len(parts) - 1]
    if name == "float32" or name == "int32":
        return 4
    if name == "float64" or name == "int64":
        return 8
    if name == "float16" or name == "bfloat16" or name == "int16":
        return 2
    if name == "int8" or name == "uint8" or name == "bool":
        return 1
    return 4  # fallback


cdef inline int _dtype_to_acl_code(object dtype):
    """Map a candle dtype object to its ACL dtype code integer.

    Returns 0 (float32) as fallback for unknown dtypes.
    """
    cdef str name = getattr(dtype, 'name', None)
    if name is None:
        name = str(dtype)
    if name == 'float32':
        return 0
    elif name == 'float16':
        return 1
    elif name == 'bfloat16':
        return 27
    elif name == 'int32':
        return 3
    elif name == 'int64':
        return 9
    elif name == 'float64':
        return 11
    elif name == 'int8':
        return 2
    elif name == 'uint8':
        return 4
    elif name == 'int16':
        return 6
    elif name == 'bool':
        return 12
    else:
        return 0  # fallback to float32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

cdef inline void _fill_shape(object py_tuple, int64_t* buf, int ndim):
    """Copy Python tuple of ints into a C array."""
    cdef int i
    for i in range(ndim):
        buf[i] = <int64_t>py_tuple[i]


cdef inline tuple _to_tuple(const int64_t* arr, int n):
    """Convert C int64 array to Python tuple."""
    return tuple([arr[i] for i in range(n)])


cdef inline int _validate_npu_binary(object a, object b, str name,
                                      int* a_dev_idx_out) except -1:
    """Validate both tensors are NPU with matching dtype. Returns 0 on success.

    Uses direct C field access when tensor is TensorImpl, falls back to
    Python attribute access otherwise.
    """
    cdef int a_dev_type, b_dev_type
    cdef int a_dtype_code, b_dtype_code

    if isinstance(a, TensorImpl):
        a_dev_type = (<TensorImpl>a)._device_type
        a_dev_idx_out[0] = (<TensorImpl>a)._device_index
        a_dtype_code = (<TensorImpl>a)._dtype_code
    else:
        a_dev = a.device
        a_dev_type = 1 if getattr(a_dev, "type", "") == "npu" else -1
        a_dev_idx_out[0] = getattr(a_dev, "index", 0) or 0
        a_dtype_code = -1  # will use Python path

    if isinstance(b, TensorImpl):
        b_dev_type = (<TensorImpl>b)._device_type
        b_dtype_code = (<TensorImpl>b)._dtype_code
    else:
        b_dev = b.device
        b_dev_type = 1 if getattr(b_dev, "type", "") == "npu" else -1
        b_dtype_code = -1

    if a_dev_type != 1 or b_dev_type != 1:
        raise ValueError(f"NPU {name} expects NPU tensors")

    # dtype check: use dtype_code if both are TensorImpl, otherwise fall back
    # to Python dtype objects. The asymmetric path (only one TensorImpl) still
    # uses Python comparison intentionally to preserve compatibility.
    if a_dtype_code >= 0 and b_dtype_code >= 0:
        if a_dtype_code != b_dtype_code:
            raise ValueError(f"NPU {name} requires matching dtypes")
    else:
        if a.dtype != b.dtype:
            raise ValueError(f"NPU {name} requires matching dtypes")

    return 0

cdef inline object _device_obj_fast(object t):
    """Return cached device object directly from TensorImpl when available."""
    if isinstance(t, TensorImpl):
        return (<TensorImpl>t)._device_obj
    return t.device


# ---------------------------------------------------------------------------
# fast_binary_op — drop-in replacement for _binary_op in _helpers.py
# ---------------------------------------------------------------------------

# Cached module references (loaded once)
cdef object _npu_runtime = None
cdef object _npu_state = None
cdef object _cy_make_npu_tensor = None

cdef object _get_runtime_fast = None
cdef object _get_stream_fast = None
cdef object _aclrt_sync_stream_fn = None   # _aclrt_ffi.synchronize_stream
cdef object _flush_executors_fn = None     # aclnn.flush_deferred_executors
cdef object _get_allocator_fn_ref = None   # allocator.get_allocator (for sync path)

# Per-device allocator cache: avoids get_allocator() dict lookup on hot path.
# Index 0 covers the overwhelmingly common single-device case.
cdef object _fast_allocator_dev0 = None    # FastNpuAllocator for device 0

cdef inline void _ensure_allocator_dev0():
    """Populate _fast_allocator_dev0 on first call (device 0 only)."""
    global _fast_allocator_dev0
    if _fast_allocator_dev0 is not None:
        return
    _ensure_npu_imports()
    _fast_allocator_dev0 = _get_allocator_fn_ref(0)

cdef inline void _ensure_npu_imports():
    global _npu_runtime, _npu_state, _cy_make_npu_tensor
    global _get_runtime_fast, _get_stream_fast
    global _aclrt_sync_stream_fn, _flush_executors_fn, _get_allocator_fn_ref
    if _npu_runtime is not None:
        return
    from candle._backends.npu import runtime as rt
    from candle._backends.npu import state as st
    from candle._backends.npu import allocator as _alloc_mod
    from candle._cython._storage import cy_make_npu_tensor as _cymt  # pylint: disable=import-error,no-name-in-module
    from candle._cython._aclrt_ffi import synchronize_stream as _ssf  # pylint: disable=import-error,no-name-in-module
    from candle._backends.npu.aclnn import flush_deferred_executors as _fef
    _npu_runtime = rt
    _npu_state = st
    _cy_make_npu_tensor = _cymt
    _get_runtime_fast = rt.get_runtime_fast
    _get_stream_fast = st.current_stream_fast
    _aclrt_sync_stream_fn = _ssf
    _flush_executors_fn = _fef
    _get_allocator_fn_ref = _alloc_mod.get_allocator


def fast_binary_op(a, b, fn, str name):
    """Drop-in replacement for _binary_op in _helpers.py.

    Does shape/stride/numel computation in C, then calls Python for:
    - runtime/stream lookup (dict + TLS)
    - allocator (complex caching + GC)
    - aclnn kernel (already Cython-ized)
    - output tensor wrapping (weakref + Python objects)
    """
    _ensure_npu_imports()

    # 1. Validate device/dtype — C field access when TensorImpl
    cdef int dev_idx
    _validate_npu_binary(a, b, name, &dev_idx)
    a_dtype = a.dtype

    # 2. Get runtime + stream (fast path: skip activate() and TLS lock)
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    # 3. Extract shapes into C arrays
    py_a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    py_b_shape = (<TensorImpl>b)._shape_tuple if isinstance(b, TensorImpl) else b.shape
    cdef int a_ndim = len(py_a_shape)
    cdef int b_ndim = len(py_b_shape)

    if a_ndim > MAX_NDIM or b_ndim > MAX_NDIM:
        raise ValueError(f"ndim exceeds MAX_NDIM ({MAX_NDIM})")

    cdef int64_t[MAX_NDIM] a_shape_buf, b_shape_buf
    cdef int64_t[MAX_NDIM] out_shape_buf, out_stride_buf

    _fill_shape(py_a_shape, a_shape_buf, a_ndim)
    _fill_shape(py_b_shape, b_shape_buf, b_ndim)

    # 4. C-level shape computation (nogil)
    cdef int out_ndim
    cdef int64_t n
    with nogil:
        out_ndim = c_broadcast_shape(
            a_shape_buf, a_ndim, b_shape_buf, b_ndim, out_shape_buf)
        c_contiguous_stride(out_shape_buf, out_ndim, out_stride_buf)
        n = c_numel(out_shape_buf, out_ndim)

    # 5. Convert to Python tuples (one allocation each)
    out_shape = _to_tuple(out_shape_buf, out_ndim)
    out_stride = _to_tuple(out_stride_buf, out_ndim)

    # 6. Allocate output — bypass _alloc_device (avoids 2 lazy imports +
    #    current_stream() Python call per op). stream.stream is the raw ACL
    #    stream pointer (int) that FastNpuAllocator.malloc expects.
    out_dtype = a_dtype
    if name == "eq" or name == "ne" or name == "lt" or name == "le" or name == "gt" or name == "ge":
        from candle._dtype import bool as _bool_dtype
        out_dtype = _bool_dtype
    cdef int64_t alloc_size = n * c_dtype_itemsize(out_dtype)
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size, stream=stream.stream)
    else:
        # Multi-device fallback: still avoids the two lazy imports inside
        # _alloc_device by going directly to get_allocator().
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size, stream=stream.stream)

    # 7. Get data pointers without Python storage() calls on the hot path
    cdef uintptr_t a_ptr
    cdef uintptr_t b_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    if isinstance(b, TensorImpl):
        b_ptr = <uintptr_t>(<TensorImpl>b)._storage._untyped._device_ptr
    else:
        b_ptr = <uintptr_t>b.storage().data_ptr()

    # 8. Call aclnn
    if name in ("atan2", "logaddexp", "logaddexp2", "remainder", "fmod", "pow", "floor_divide", "eq", "ne", "lt", "le", "gt", "ge", "logical_and", "logical_or", "logical_xor", "bitwise_and", "bitwise_or", "bitwise_xor", "max", "maximum", "min", "minimum"):
        from candle._cython import _aclnn_ffi as _ffi  # pylint: disable=import-error,no-name-in-module
        from candle._backends.npu.aclnn import ensure_acl as _ensure_acl

        acl = _ensure_acl()
        dtype_code = _dtype_to_acl_code(a_dtype)
        if name == "atan2":
            pretty = "aclnnAtan2"
            getws_ptr, exec_ptr = _ffi.resolve_op("Atan2")
            ws_size, executor = _ffi.binary_op_no_alpha(
                getws_ptr, exec_ptr,
                py_a_shape, a.stride,
                py_b_shape, b.stride,
                out_shape, out_stride,
                dtype_code, 2,
                int(a_ptr), int(b_ptr), int(out_ptr),
                int(stream.stream))
        elif name == "logaddexp":
            pretty = "aclnnLogAddExp"
            getws_ptr, exec_ptr = _ffi.resolve_op("LogAddExp")
            ws_size, executor = _ffi.binary_op_no_alpha(
                getws_ptr, exec_ptr,
                py_a_shape, a.stride,
                py_b_shape, b.stride,
                out_shape, out_stride,
                dtype_code, 2,
                int(a_ptr), int(b_ptr), int(out_ptr),
                int(stream.stream))
        elif name == "logaddexp2":
            pretty = "aclnnLogAddExp2"
            getws_ptr, exec_ptr = _ffi.resolve_op("LogAddExp2")
            ws_size, executor = _ffi.binary_op_no_alpha(
                getws_ptr, exec_ptr,
                py_a_shape, a.stride,
                py_b_shape, b.stride,
                out_shape, out_stride,
                dtype_code, 2,
                int(a_ptr), int(b_ptr), int(out_ptr),
                int(stream.stream))
        elif name == "remainder":
            pretty = "aclnnRemainderTensorTensor"
            getws_ptr, exec_ptr = _ffi.resolve_op("RemainderTensorTensor")
            ws_size, executor = _ffi.binary_two_inputs_op(
                getws_ptr, exec_ptr,
                py_a_shape, a.stride,
                py_b_shape, b.stride,
                out_shape, out_stride,
                dtype_code, dtype_code, dtype_code,
                2,
                int(a_ptr), int(b_ptr), int(out_ptr),
                int(stream.stream))
        elif name == "fmod":
            pretty = "aclnnFmodTensor"
            getws_ptr, exec_ptr = _ffi.resolve_op("FmodTensor")
            ws_size, executor = _ffi.binary_two_inputs_op(
                getws_ptr, exec_ptr,
                py_a_shape, a.stride,
                py_b_shape, b.stride,
                out_shape, out_stride,
                dtype_code, dtype_code, dtype_code,
                2,
                int(a_ptr), int(b_ptr), int(out_ptr),
                int(stream.stream))
        elif name == "pow":
            pretty = "aclnnPowTensorTensor"
            getws_ptr, exec_ptr = _ffi.resolve_op("PowTensorTensor")
            ws_size, executor = _ffi.binary_two_inputs_op(
                getws_ptr, exec_ptr,
                py_a_shape, a.stride,
                py_b_shape, b.stride,
                out_shape, out_stride,
                dtype_code, dtype_code, dtype_code,
                2,
                int(a_ptr), int(b_ptr), int(out_ptr),
                int(stream.stream))
        elif name == "floor_divide":
            pretty = "aclnnFloorDivide"
            getws_ptr, exec_ptr = _ffi.resolve_op("FloorDivide")
            ws_size, executor = _ffi.binary_two_inputs_op(
                getws_ptr, exec_ptr,
                py_a_shape, a.stride,
                py_b_shape, b.stride,
                out_shape, out_stride,
                dtype_code, dtype_code, dtype_code,
                2,
                int(a_ptr), int(b_ptr), int(out_ptr),
                int(stream.stream))
        elif name == "eq":
            pretty = "aclnnEqTensor"
            getws_ptr, exec_ptr = _ffi.resolve_op("EqTensor")
            ws_size, executor = _ffi.binary_two_inputs_op(
                getws_ptr, exec_ptr,
                py_a_shape, a.stride,
                py_b_shape, b.stride,
                out_shape, out_stride,
                dtype_code, dtype_code, 12,
                2,
                int(a_ptr), int(b_ptr), int(out_ptr),
                int(stream.stream))
        elif name == "ne":
            pretty = "aclnnNeTensor"
            getws_ptr, exec_ptr = _ffi.resolve_op("NeTensor")
            ws_size, executor = _ffi.binary_two_inputs_op(
                getws_ptr, exec_ptr,
                py_a_shape, a.stride,
                py_b_shape, b.stride,
                out_shape, out_stride,
                dtype_code, dtype_code, 12,
                2,
                int(a_ptr), int(b_ptr), int(out_ptr),
                int(stream.stream))
        elif name == "lt":
            pretty = "aclnnLtTensor"
            getws_ptr, exec_ptr = _ffi.resolve_op("LtTensor")
            ws_size, executor = _ffi.binary_two_inputs_op(
                getws_ptr, exec_ptr,
                py_a_shape, a.stride,
                py_b_shape, b.stride,
                out_shape, out_stride,
                dtype_code, dtype_code, 12,
                2,
                int(a_ptr), int(b_ptr), int(out_ptr),
                int(stream.stream))
        elif name == "le":
            pretty = "aclnnLeTensor"
            getws_ptr, exec_ptr = _ffi.resolve_op("LeTensor")
            ws_size, executor = _ffi.binary_two_inputs_op(
                getws_ptr, exec_ptr,
                py_a_shape, a.stride,
                py_b_shape, b.stride,
                out_shape, out_stride,
                dtype_code, dtype_code, 12,
                2,
                int(a_ptr), int(b_ptr), int(out_ptr),
                int(stream.stream))
        elif name == "gt":
            pretty = "aclnnGtTensor"
            getws_ptr, exec_ptr = _ffi.resolve_op("GtTensor")
            ws_size, executor = _ffi.binary_two_inputs_op(
                getws_ptr, exec_ptr,
                py_a_shape, a.stride,
                py_b_shape, b.stride,
                out_shape, out_stride,
                dtype_code, dtype_code, 12,
                2,
                int(a_ptr), int(b_ptr), int(out_ptr),
                int(stream.stream))
        elif name == "ge":
            pretty = "aclnnGeTensor"
            getws_ptr, exec_ptr = _ffi.resolve_op("GeTensor")
            ws_size, executor = _ffi.binary_two_inputs_op(
                getws_ptr, exec_ptr,
                py_a_shape, a.stride,
                py_b_shape, b.stride,
                out_shape, out_stride,
                dtype_code, dtype_code, 12,
                2,
                int(a_ptr), int(b_ptr), int(out_ptr),
                int(stream.stream))
        elif name == "logical_and":
            pretty = "aclnnLogicalAnd"
            getws_ptr, exec_ptr = _ffi.resolve_op("LogicalAnd")
            ws_size, executor = _ffi.binary_two_inputs_op(
                getws_ptr, exec_ptr,
                py_a_shape, a.stride,
                py_b_shape, b.stride,
                out_shape, out_stride,
                dtype_code, dtype_code, dtype_code,
                2,
                int(a_ptr), int(b_ptr), int(out_ptr),
                int(stream.stream))
        elif name == "logical_or":
            pretty = "aclnnLogicalOr"
            getws_ptr, exec_ptr = _ffi.resolve_op("LogicalOr")
            ws_size, executor = _ffi.binary_two_inputs_op(
                getws_ptr, exec_ptr,
                py_a_shape, a.stride,
                py_b_shape, b.stride,
                out_shape, out_stride,
                dtype_code, dtype_code, dtype_code,
                2,
                int(a_ptr), int(b_ptr), int(out_ptr),
                int(stream.stream))
        elif name == "logical_xor":
            pretty = "aclnnLogicalXor"
            getws_ptr, exec_ptr = _ffi.resolve_op("LogicalXor")
            ws_size, executor = _ffi.binary_two_inputs_op(
                getws_ptr, exec_ptr,
                py_a_shape, a.stride,
                py_b_shape, b.stride,
                out_shape, out_stride,
                dtype_code, dtype_code, 12,
                2,
                int(a_ptr), int(b_ptr), int(out_ptr),
                int(stream.stream))
        elif name == "bitwise_and":
            pretty = "aclnnBitwiseAndTensor"
            getws_ptr, exec_ptr = _ffi.resolve_op("BitwiseAndTensor")
            ws_size, executor = _ffi.binary_two_inputs_op(
                getws_ptr, exec_ptr,
                py_a_shape, a.stride,
                py_b_shape, b.stride,
                out_shape, out_stride,
                dtype_code, dtype_code, dtype_code,
                2,
                int(a_ptr), int(b_ptr), int(out_ptr),
                int(stream.stream))
        elif name == "bitwise_or":
            pretty = "aclnnBitwiseOrTensor"
            getws_ptr, exec_ptr = _ffi.resolve_op("BitwiseOrTensor")
            ws_size, executor = _ffi.binary_two_inputs_op(
                getws_ptr, exec_ptr,
                py_a_shape, a.stride,
                py_b_shape, b.stride,
                out_shape, out_stride,
                dtype_code, dtype_code, dtype_code,
                2,
                int(a_ptr), int(b_ptr), int(out_ptr),
                int(stream.stream))
        elif name == "bitwise_xor":
            pretty = "aclnnBitwiseXorTensor"
            getws_ptr, exec_ptr = _ffi.resolve_op("BitwiseXorTensor")
            ws_size, executor = _ffi.binary_two_inputs_op(
                getws_ptr, exec_ptr,
                py_a_shape, a.stride,
                py_b_shape, b.stride,
                out_shape, out_stride,
                dtype_code, dtype_code, dtype_code,
                2,
                int(a_ptr), int(b_ptr), int(out_ptr),
                int(stream.stream))
        elif name == "max" or name == "maximum":
            pretty = "aclnnMaximum"
            getws_ptr, exec_ptr = _ffi.resolve_op("Maximum")
            ws_size, executor = _ffi.binary_two_inputs_op(
                getws_ptr, exec_ptr,
                py_a_shape, a.stride,
                py_b_shape, b.stride,
                out_shape, out_stride,
                dtype_code, dtype_code, dtype_code,
                2,
                int(a_ptr), int(b_ptr), int(out_ptr),
                int(stream.stream))
        else:
            pretty = "aclnnMinimum"
            getws_ptr, exec_ptr = _ffi.resolve_op("Minimum")
            ws_size, executor = _ffi.binary_two_inputs_op(
                getws_ptr, exec_ptr,
                py_a_shape, a.stride,
                py_b_shape, b.stride,
                out_shape, out_stride,
                dtype_code, dtype_code, dtype_code,
                2,
                int(a_ptr), int(b_ptr), int(out_ptr),
                int(stream.stream))
        if ws_size:
            workspace_ptr, ret = acl.rt.malloc(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            try:
                ret = _ffi.execute(exec_ptr, int(workspace_ptr), ws_size, executor, int(stream.stream))
                if ret != 0:
                    raise RuntimeError(f"{pretty} failed: {ret}")
            finally:
                runtime.defer_raw_free(workspace_ptr)
    else:
        fn(
            a_ptr,
            b_ptr,
            out_ptr,
            py_a_shape,
            a.stride,
            py_b_shape,
            b.stride,
            out_shape,
            out_stride,
            a_dtype,
            runtime,
            stream=stream.stream,
        )

    # 9. Wrap output — construct Tensor entirely in Cython (skips Python __init__)
    a_dev = _device_obj_fast(a)
    return _cy_make_npu_tensor(out_ptr, n, out_dtype, a_dev, out_shape, out_stride)


# ---------------------------------------------------------------------------
# fast_add — hardwired add(a, b, alpha=1) that skips aclnn.py wrapper
# ---------------------------------------------------------------------------

cdef object _ffi_ref = None              # _aclnn_ffi module
cdef object _add_getws_ptr = None        # cached Add getws pointer
cdef object _add_exec_ptr = None         # cached Add exec pointer
cdef object _mul_getws_ptr = None        # cached Mul getws pointer
cdef object _mul_exec_ptr = None         # cached Mul exec pointer
cdef object _sub_getws_ptr = None        # cached Sub getws pointer
cdef object _sub_exec_ptr = None         # cached Sub exec pointer
cdef object _div_getws_ptr = None        # cached Div getws pointer
cdef object _div_exec_ptr = None         # cached Div exec pointer
cdef object _defer_executor_fn = None    # aclnn._defer_executor
cdef object _acl_rt_malloc_fn = None     # acl.rt.malloc
cdef object _acl_rt_free_fn = None       # acl.rt.free (for workspace)
cdef dict _alpha_one_handles = {}        # dtype_code -> alpha=1 scalar handle (int)
cdef dict _alpha_one_bytes_cache = {}    # dtype_code -> (bytes, alpha_dtype_code) for PTA hash
cdef object _pta_cache_begin_fn = None   # _aclnn_ffi.pta_begin_add_cache_lookup
cdef object _pta_cache_end_fn = None     # _aclnn_ffi.pta_end_cache_lookup
# Disabled by default: the PTA cached-executor path can reuse stale bound tensor
# addresses across NPU add calls, which corrupts later view-based adds
# (e.g. a prior x+x can make a later view+ones behave like view+view).
# Keep the integration wired but guarded so it can be re-enabled once the
# cached executor path safely rebinds descriptors/addresses.
cdef bint _use_add_pta_cache = False


cdef inline void _ensure_ffi_binary() except *:
    global _ffi_ref, _add_getws_ptr, _add_exec_ptr
    global _mul_getws_ptr, _mul_exec_ptr
    global _sub_getws_ptr, _sub_exec_ptr
    global _div_getws_ptr, _div_exec_ptr
    global _defer_executor_fn, _acl_rt_malloc_fn, _acl_rt_free_fn
    global _pta_cache_begin_fn, _pta_cache_end_fn
    if _ffi_ref is not None:
        return
    from candle._cython import _aclnn_ffi as _f  # pylint: disable=import-error,no-name-in-module
    from candle._backends.npu.aclnn import _defer_executor as _def_ex, ensure_acl as _eacl
    _ffi_ref = _f
    _add_getws_ptr, _add_exec_ptr = _f.resolve_op("Add")
    _mul_getws_ptr, _mul_exec_ptr = _f.resolve_op("Mul")
    _sub_getws_ptr, _sub_exec_ptr = _f.resolve_op("Sub")
    _div_getws_ptr, _div_exec_ptr = _f.resolve_op("Div")
    _defer_executor_fn = _def_ex
    _acl = _eacl()
    _acl_rt_malloc_fn = _acl.rt.malloc
    _acl_rt_free_fn = _acl.rt.free
    if _f.is_pta_cache_available():
        _pta_cache_begin_fn = _f.pta_begin_add_cache_lookup
        _pta_cache_end_fn = _f.pta_end_cache_lookup


cdef uintptr_t _get_alpha_one(int dtype_code) except? 0:
    """Return a cached alpha=1 scalar handle for the given dtype_code."""
    global _alpha_one_handles
    cdef object existing = _alpha_one_handles.get(dtype_code)
    if existing is not None:
        return <uintptr_t>existing
    import struct
    if dtype_code == 0:    # float32
        scalar_bytes = struct.pack('<f', 1.0)
    elif dtype_code == 1:  # float16 — bits = 0x3C00, little-endian
        scalar_bytes = b'\x00\x3c'
    elif dtype_code == 27: # bfloat16 — bits = 0x3F80, little-endian
        scalar_bytes = b'\x80\x3f'
    elif dtype_code == 3:  # int32
        scalar_bytes = struct.pack('<i', 1)
    elif dtype_code == 9:  # int64
        scalar_bytes = struct.pack('<q', 1)
    elif dtype_code == 11: # float64
        scalar_bytes = struct.pack('<d', 1.0)
    elif dtype_code == 2:  # int8
        scalar_bytes = b'\x01'
    elif dtype_code == 4:  # uint8
        scalar_bytes = b'\x01'
    elif dtype_code == 6:  # int16
        scalar_bytes = b'\x01\x00'
    elif dtype_code == 12: # bool
        scalar_bytes = b'\x01'
    else:
        scalar_bytes = struct.pack('<f', 1.0)  # fallback
    cdef uintptr_t handle = <uintptr_t>_ffi_ref.create_scalar(scalar_bytes, dtype_code)
    _alpha_one_handles[dtype_code] = handle
    return handle


cdef object _get_alpha_one_bytes(int dtype_code):
    """Return cached (scalar_bytes, scalar_dtype_code) for alpha=1."""
    global _alpha_one_bytes_cache
    cdef object existing = _alpha_one_bytes_cache.get(dtype_code)
    if existing is not None:
        return existing
    import struct
    if dtype_code == 0:    # float32
        scalar_bytes = struct.pack('<f', 1.0)
    elif dtype_code == 1:  # float16
        scalar_bytes = b'\x00\x3c'
    elif dtype_code == 27: # bfloat16
        scalar_bytes = b'\x80\x3f'
    elif dtype_code == 3:  # int32
        scalar_bytes = struct.pack('<i', 1)
    elif dtype_code == 9:  # int64
        scalar_bytes = struct.pack('<q', 1)
    elif dtype_code == 11: # float64
        scalar_bytes = struct.pack('<d', 1.0)
    elif dtype_code == 2:  # int8
        scalar_bytes = b'\x01'
    elif dtype_code == 4:  # uint8
        scalar_bytes = b'\x01'
    elif dtype_code == 6:  # int16
        scalar_bytes = b'\x01\x00'
    elif dtype_code == 12: # bool
        scalar_bytes = b'\x01'
    else:
        scalar_bytes = struct.pack('<f', 1.0)
        dtype_code = 0
    existing = (scalar_bytes, dtype_code)
    _alpha_one_bytes_cache[dtype_code] = existing
    return existing


def fast_add(a, b):
    """Optimized add(a, b, alpha=1) that calls _ffi.binary_op_with_alpha directly.

    Skips aclnn.add wrapper overhead:
    - No get_bindings() dict lookup
    - No _require_native_npu_ffi check
    - No _scalar_bytes creation each call (cached per dtype)
    - No resolve_op each call (cached on first use)
    - No ctypes.c_void_p wrapping of executor
    - No a.storage().data_ptr() Python method calls (direct C attribute access)
    """
    _ensure_npu_imports()
    _ensure_ffi_binary()

    # 1. Validate device/dtype — C field access when TensorImpl
    cdef int dev_idx
    _validate_npu_binary(a, b, "add", &dev_idx)
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype

    # 2. Get runtime + stream
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    # 3. Extract shapes into C arrays
    py_a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    py_b_shape = (<TensorImpl>b)._shape_tuple if isinstance(b, TensorImpl) else b.shape
    cdef int a_ndim = len(py_a_shape)
    cdef int b_ndim = len(py_b_shape)

    if a_ndim > MAX_NDIM or b_ndim > MAX_NDIM:
        raise ValueError(f"ndim exceeds MAX_NDIM ({MAX_NDIM})")

    cdef int64_t[MAX_NDIM] a_shape_buf, b_shape_buf
    cdef int64_t[MAX_NDIM] out_shape_buf, out_stride_buf

    _fill_shape(py_a_shape, a_shape_buf, a_ndim)
    _fill_shape(py_b_shape, b_shape_buf, b_ndim)

    # 4. C-level shape computation
    cdef int out_ndim
    cdef int64_t n
    with nogil:
        out_ndim = c_broadcast_shape(
            a_shape_buf, a_ndim, b_shape_buf, b_ndim, out_shape_buf)
        c_contiguous_stride(out_shape_buf, out_ndim, out_stride_buf)
        n = c_numel(out_shape_buf, out_ndim)

    # 5. Convert to Python tuples
    out_shape = _to_tuple(out_shape_buf, out_ndim)
    out_stride = _to_tuple(out_stride_buf, out_ndim)

    # 6. Allocate output via cached allocator (bypasses _alloc_device Python overhead)
    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_fa = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fa, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fa, stream=stream.stream)

    # 7. Get dtype code and cached alpha=1 handle
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)

    # 8. Get data pointers — direct C attribute access (no Python method calls)
    cdef uintptr_t a_ptr, b_ptr, o_ptr
    a_ptr = a._storage._untyped._device_ptr
    b_ptr = b._storage._untyped._device_ptr
    o_ptr = out_ptr

    cdef uintptr_t stream_raw = int(stream.stream)
    cdef bint pta_active = False
    cdef uintptr_t alpha_handle

    # 9. Try PTA executor cache (torch_npu-aligned hit_cache_v2 path)
    if _use_add_pta_cache and _pta_cache_begin_fn is not None:
        alpha_bytes_pair = _get_alpha_one_bytes(dtype_code)
        state = _pta_cache_begin_fn(
            py_a_shape, a.stride,
            py_b_shape, b.stride,
            out_shape, out_stride,
            dtype_code,
            a_ptr, b_ptr, o_ptr,
            alpha_bytes_pair[0], alpha_bytes_pair[1],
            stream_raw)
        if state is not None:
            pta_active = bool(state[0])
            ws_size = state[1]
            executor = state[2]
            if executor:
                try:
                    if ws_size:
                        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
                        if ret != 0:
                            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
                        try:
                            ret = _ffi_ref.execute(
                                _add_exec_ptr, int(workspace_ptr), ws_size,
                                executor, stream_raw)
                            if ret != 0:
                                raise RuntimeError(f"aclnnAdd execute failed: {ret}")
                        finally:
                            runtime.defer_raw_free(workspace_ptr)
                    else:
                        ret = _ffi_ref.execute(_add_exec_ptr, 0, 0, executor, stream_raw)
                        if ret != 0:
                            raise RuntimeError(f"aclnnAdd execute failed: {ret}")
                    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)
                finally:
                    if pta_active:
                        _pta_cache_end_fn()
                        pta_active = False

    try:
        # 10. Cache miss: full GetWorkspaceSize + Execute path
        alpha_handle = _get_alpha_one(dtype_code)
        ws_size, executor = _ffi_ref.binary_op_with_alpha(
            _add_getws_ptr, _add_exec_ptr,
            py_a_shape, a.stride,
            py_b_shape, b.stride,
            out_shape, out_stride,
            dtype_code, 2,  # ACL_FORMAT_ND = 2
            a_ptr, b_ptr, o_ptr,
            alpha_handle,
            stream_raw)

        # 11. Handle workspace (rare: ws_size > 0 means execute not yet called)
        if ws_size:
            workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            try:
                ret = _ffi_ref.execute(
                    _add_exec_ptr, int(workspace_ptr), ws_size,
                    executor, stream_raw)
                if ret != 0:
                    raise RuntimeError(f"aclnnAdd execute failed: {ret}")
            finally:
                runtime.defer_raw_free(workspace_ptr)

        # 12. Defer executor cleanup — pass raw int handle directly
        #     (skips ctypes.c_void_p wrapping; _defer_executor extracts int via
        #     _executor_handle which handles both c_void_p and plain int)
        _defer_executor_fn(executor)
    finally:
        if pta_active:
            _pta_cache_end_fn()

    # 13. Wrap output — construct Tensor entirely in Cython (same as fast_binary_op)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


# ---------------------------------------------------------------------------
# fast_mul — hardwired mul(a, b) that skips aclnn.py wrapper
# ---------------------------------------------------------------------------

def fast_mul(a, b):
    """Optimized mul(a, b) that calls _ffi.binary_op_no_alpha directly.

    Skips aclnn.mul wrapper overhead:
    - No get_bindings() dict lookup
    - No _require_native_npu_ffi check
    - No resolve_op each call (cached in _ensure_ffi_binary)
    - No ctypes.c_void_p wrapping of executor
    - Direct C attribute access for device pointers
    """
    _ensure_npu_imports()
    _ensure_ffi_binary()

    # 1. Validate device/dtype — C field access when TensorImpl
    cdef int dev_idx
    _validate_npu_binary(a, b, "mul", &dev_idx)
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype

    # 2. Get runtime + stream
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    # 3. Extract shapes into C arrays
    py_a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    py_b_shape = (<TensorImpl>b)._shape_tuple if isinstance(b, TensorImpl) else b.shape
    cdef int a_ndim = len(py_a_shape)
    cdef int b_ndim = len(py_b_shape)

    if a_ndim > MAX_NDIM or b_ndim > MAX_NDIM:
        raise ValueError(f"ndim exceeds MAX_NDIM ({MAX_NDIM})")

    cdef int64_t[MAX_NDIM] a_shape_buf, b_shape_buf
    cdef int64_t[MAX_NDIM] out_shape_buf, out_stride_buf

    _fill_shape(py_a_shape, a_shape_buf, a_ndim)
    _fill_shape(py_b_shape, b_shape_buf, b_ndim)

    # 4. C-level shape computation
    cdef int out_ndim
    cdef int64_t n
    with nogil:
        out_ndim = c_broadcast_shape(
            a_shape_buf, a_ndim, b_shape_buf, b_ndim, out_shape_buf)
        c_contiguous_stride(out_shape_buf, out_ndim, out_stride_buf)
        n = c_numel(out_shape_buf, out_ndim)

    # 5. Convert to Python tuples
    out_shape = _to_tuple(out_shape_buf, out_ndim)
    out_stride = _to_tuple(out_stride_buf, out_ndim)

    # 6. Allocate output via cached allocator
    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_fm = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fm, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fm, stream=stream.stream)

    # 7. Get dtype code
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)

    # 8. Get data pointers — direct C attribute access (no Python method calls)
    cdef uintptr_t a_ptr, b_ptr, o_ptr
    a_ptr = a._storage._untyped._device_ptr
    b_ptr = b._storage._untyped._device_ptr
    o_ptr = out_ptr

    cdef uintptr_t stream_raw = int(stream.stream)

    # 9. Full GetWorkspaceSize + Execute path (no PTA cache for mul)
    ws_size, executor = _ffi_ref.binary_op_no_alpha(
        _mul_getws_ptr, _mul_exec_ptr,
        py_a_shape, a.stride,
        py_b_shape, b.stride,
        out_shape, out_stride,
        dtype_code, 2,  # ACL_FORMAT_ND = 2
        a_ptr, b_ptr, o_ptr,
        stream_raw)

    # 10. Handle workspace (rare: ws_size > 0 means execute not yet called)
    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _mul_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnMul execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    # 11. Defer executor cleanup
    _defer_executor_fn(executor)

    # 12. Wrap output
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


# ---------------------------------------------------------------------------
# fast_sub — hardwired sub(a, b, alpha=1) that skips aclnn.py wrapper
# ---------------------------------------------------------------------------

def fast_sub(a, b):
    """Optimized sub(a, b, alpha=1) that calls _ffi.binary_op_with_alpha directly."""
    _ensure_npu_imports()
    _ensure_ffi_binary()

    # 1. Validate device/dtype — C field access when TensorImpl
    cdef int dev_idx
    _validate_npu_binary(a, b, "sub", &dev_idx)
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype

    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    py_a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    py_b_shape = (<TensorImpl>b)._shape_tuple if isinstance(b, TensorImpl) else b.shape
    cdef int a_ndim = len(py_a_shape)
    cdef int b_ndim = len(py_b_shape)

    if a_ndim > MAX_NDIM or b_ndim > MAX_NDIM:
        raise ValueError(f"ndim exceeds MAX_NDIM ({MAX_NDIM})")

    cdef int64_t[MAX_NDIM] a_shape_buf, b_shape_buf
    cdef int64_t[MAX_NDIM] out_shape_buf, out_stride_buf

    _fill_shape(py_a_shape, a_shape_buf, a_ndim)
    _fill_shape(py_b_shape, b_shape_buf, b_ndim)

    cdef int out_ndim
    cdef int64_t n
    with nogil:
        out_ndim = c_broadcast_shape(
            a_shape_buf, a_ndim, b_shape_buf, b_ndim, out_shape_buf)
        c_contiguous_stride(out_shape_buf, out_ndim, out_stride_buf)
        n = c_numel(out_shape_buf, out_ndim)

    out_shape = _to_tuple(out_shape_buf, out_ndim)
    out_stride = _to_tuple(out_stride_buf, out_ndim)

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_fs = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fs, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fs, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t alpha_handle = _get_alpha_one(dtype_code)
    cdef uintptr_t a_ptr = a._storage._untyped._device_ptr
    cdef uintptr_t b_ptr = b._storage._untyped._device_ptr
    cdef uintptr_t o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.binary_op_with_alpha(
        _sub_getws_ptr, _sub_exec_ptr,
        py_a_shape, a.stride,
        py_b_shape, b.stride,
        out_shape, out_stride,
        dtype_code, 2,
        a_ptr, b_ptr, o_ptr,
        alpha_handle,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _sub_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnSub execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)

    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


# ---------------------------------------------------------------------------
# fast_div — hardwired div(a, b) that skips aclnn.py wrapper
# ---------------------------------------------------------------------------

def fast_div(a, b):
    """Optimized div(a, b) that calls _ffi.binary_op_no_alpha directly."""
    _ensure_npu_imports()
    _ensure_ffi_binary()

    # 1. Validate device/dtype — C field access when TensorImpl
    cdef int dev_idx
    _validate_npu_binary(a, b, "div", &dev_idx)
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype

    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    py_a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    py_b_shape = (<TensorImpl>b)._shape_tuple if isinstance(b, TensorImpl) else b.shape
    cdef int a_ndim = len(py_a_shape)
    cdef int b_ndim = len(py_b_shape)

    if a_ndim > MAX_NDIM or b_ndim > MAX_NDIM:
        raise ValueError(f"ndim exceeds MAX_NDIM ({MAX_NDIM})")

    cdef int64_t[MAX_NDIM] a_shape_buf, b_shape_buf
    cdef int64_t[MAX_NDIM] out_shape_buf, out_stride_buf

    _fill_shape(py_a_shape, a_shape_buf, a_ndim)
    _fill_shape(py_b_shape, b_shape_buf, b_ndim)

    cdef int out_ndim
    cdef int64_t n
    with nogil:
        out_ndim = c_broadcast_shape(
            a_shape_buf, a_ndim, b_shape_buf, b_ndim, out_shape_buf)
        c_contiguous_stride(out_shape_buf, out_ndim, out_stride_buf)
        n = c_numel(out_shape_buf, out_ndim)

    out_shape = _to_tuple(out_shape_buf, out_ndim)
    out_stride = _to_tuple(out_stride_buf, out_ndim)

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_fd = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fd, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fd, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr = a._storage._untyped._device_ptr
    cdef uintptr_t b_ptr = b._storage._untyped._device_ptr
    cdef uintptr_t o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.binary_op_no_alpha(
        _div_getws_ptr, _div_exec_ptr,
        py_a_shape, a.stride,
        py_b_shape, b.stride,
        out_shape, out_stride,
        dtype_code, 2,
        a_ptr, b_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _div_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnDiv execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)

    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


# ---------------------------------------------------------------------------
# fast_lerp_tensor — hardwired lerp(a, b, weight_tensor) that skips aclnn.py wrapper
# ---------------------------------------------------------------------------

cdef object _where_getws_ptr = None      # cached SWhere getws pointer
cdef object _where_exec_ptr = None       # cached SWhere exec pointer
cdef object _digamma_getws_ptr = None    # cached Digamma getws pointer
cdef object _digamma_exec_ptr = None     # cached Digamma exec pointer
cdef object _lgamma_getws_ptr = None     # cached Lgamma getws pointer
cdef object _lgamma_exec_ptr = None      # cached Lgamma exec pointer
cdef object _sinc_getws_ptr = None       # cached Sinc getws pointer
cdef object _sinc_exec_ptr = None        # cached Sinc exec pointer
cdef object _abs_getws_ptr = None        # cached Abs getws pointer
cdef object _abs_exec_ptr = None         # cached Abs exec pointer
cdef object _neg_getws_ptr = None        # cached Neg getws pointer
cdef object _neg_exec_ptr = None         # cached Neg exec pointer
cdef object _sign_getws_ptr = None       # cached Sign getws pointer
cdef object _sign_exec_ptr = None        # cached Sign exec pointer
cdef object _signbit_getws_ptr = None    # cached Signbit getws pointer
cdef object _signbit_exec_ptr = None     # cached Signbit exec pointer
cdef object _isfinite_getws_ptr = None   # cached IsFinite getws pointer
cdef object _isfinite_exec_ptr = None    # cached IsFinite exec pointer
cdef object _isposinf_getws_ptr = None   # cached IsPosInf getws pointer
cdef object _isposinf_exec_ptr = None    # cached IsPosInf exec pointer
cdef object _isneginf_getws_ptr = None   # cached IsNegInf getws pointer
cdef object _isneginf_exec_ptr = None    # cached IsNegInf exec pointer
cdef object _logical_not_getws_ptr = None # cached LogicalNot getws pointer
cdef object _logical_not_exec_ptr = None  # cached LogicalNot exec pointer
cdef object _bitwise_not_getws_ptr = None # cached BitwiseNot getws pointer
cdef object _bitwise_not_exec_ptr = None  # cached BitwiseNot exec pointer
cdef object _square_getws_ptr = None     # cached Square getws pointer
cdef object _square_exec_ptr = None      # cached Square exec pointer
cdef object _exp_getws_ptr = None        # cached Exp getws pointer
cdef object _exp_exec_ptr = None         # cached Exp exec pointer
cdef object _expm1_getws_ptr = None      # cached Expm1 getws pointer
cdef object _expm1_exec_ptr = None       # cached Expm1 exec pointer
cdef object _log1p_getws_ptr = None      # cached Log1p getws pointer
cdef object _log1p_exec_ptr = None       # cached Log1p exec pointer
cdef object _log_getws_ptr = None        # cached Log getws pointer
cdef object _log_exec_ptr = None         # cached Log exec pointer
cdef object _log2_getws_ptr = None       # cached Log2 getws pointer
cdef object _log2_exec_ptr = None        # cached Log2 exec pointer
cdef object _log10_getws_ptr = None      # cached Log10 getws pointer
cdef object _log10_exec_ptr = None       # cached Log10 exec pointer
cdef object _exp2_getws_ptr = None       # cached Exp2 getws pointer
cdef object _exp2_exec_ptr = None        # cached Exp2 exec pointer
cdef object _asinh_getws_ptr = None      # cached Asinh getws pointer
cdef object _asinh_exec_ptr = None       # cached Asinh exec pointer
cdef object _acosh_getws_ptr = None      # cached Acosh getws pointer
cdef object _acosh_exec_ptr = None       # cached Acosh exec pointer
cdef object _atanh_getws_ptr = None      # cached Atanh getws pointer
cdef object _atanh_exec_ptr = None       # cached Atanh exec pointer
cdef object _atan_getws_ptr = None       # cached Atan getws pointer
cdef object _atan_exec_ptr = None        # cached Atan exec pointer
cdef object _asin_getws_ptr = None       # cached Asin getws pointer
cdef object _asin_exec_ptr = None        # cached Asin exec pointer
cdef object _acos_getws_ptr = None       # cached Acos getws pointer
cdef object _acos_exec_ptr = None        # cached Acos exec pointer
cdef object _rsqrt_getws_ptr = None      # cached Rsqrt getws pointer
cdef object _rsqrt_exec_ptr = None       # cached Rsqrt exec pointer
cdef object _sqrt_getws_ptr = None       # cached Sqrt getws pointer
cdef object _sqrt_exec_ptr = None        # cached Sqrt exec pointer
cdef object _sin_getws_ptr = None        # cached Sin getws pointer
cdef object _sin_exec_ptr = None         # cached Sin exec pointer
cdef object _cos_getws_ptr = None        # cached Cos getws pointer
cdef object _cos_exec_ptr = None         # cached Cos exec pointer
cdef object _tan_getws_ptr = None        # cached Tan getws pointer
cdef object _tan_exec_ptr = None         # cached Tan exec pointer
cdef object _tanh_getws_ptr = None       # cached Tanh getws pointer
cdef object _tanh_exec_ptr = None        # cached Tanh exec pointer
cdef object _sigmoid_getws_ptr = None    # cached Sigmoid getws pointer
cdef object _sigmoid_exec_ptr = None     # cached Sigmoid exec pointer
cdef object _relu_getws_ptr = None       # cached Relu getws pointer
cdef object _relu_exec_ptr = None        # cached Relu exec pointer
cdef object _leaky_relu_getws_ptr = None # cached LeakyRelu getws pointer
cdef object _leaky_relu_exec_ptr = None  # cached LeakyRelu exec pointer
cdef object _elu_getws_ptr = None        # cached Elu getws pointer
cdef object _elu_exec_ptr = None         # cached Elu exec pointer
cdef object _dropout_gen_mask_getws_ptr = None   # cached DropoutGenMask getws pointer
cdef object _dropout_gen_mask_exec_ptr = None    # cached DropoutGenMask exec pointer
cdef object _dropout_do_mask_getws_ptr = None    # cached DropoutDoMask getws pointer
cdef object _dropout_do_mask_exec_ptr = None     # cached DropoutDoMask exec pointer
cdef object _npu_mod_ref = None                  # cached candle.npu module

cdef object _embedding_getws_ptr = None  # cached Embedding getws pointer
cdef object _embedding_exec_ptr = None   # cached Embedding exec pointer
cdef object _prelu_getws_ptr = None      # cached Prelu getws pointer
cdef object _prelu_exec_ptr = None       # cached Prelu exec pointer
cdef object _softplus_getws_ptr = None   # cached Softplus getws pointer
cdef object _softplus_exec_ptr = None    # cached Softplus exec pointer
cdef object _softmax_getws_ptr = None    # cached Softmax getws pointer
cdef object _softmax_exec_ptr = None     # cached Softmax exec pointer
cdef object _log_softmax_getws_ptr = None # cached LogSoftmax getws pointer
cdef object _log_softmax_exec_ptr = None  # cached LogSoftmax exec pointer
cdef object _hardtanh_getws_ptr = None   # cached Hardtanh getws pointer
cdef object _hardtanh_exec_ptr = None    # cached Hardtanh exec pointer
cdef object _gelu_getws_ptr = None       # cached Gelu getws pointer
cdef object _gelu_exec_ptr = None        # cached Gelu exec pointer
cdef object _silu_getws_ptr = None       # cached Silu getws pointer
cdef object _silu_exec_ptr = None        # cached Silu exec pointer
cdef object _mish_getws_ptr = None       # cached Mish getws pointer
cdef object _mish_exec_ptr = None        # cached Mish exec pointer
cdef object _sinh_getws_ptr = None       # cached Sinh getws pointer
cdef object _sinh_exec_ptr = None        # cached Sinh exec pointer
cdef object _cosh_getws_ptr = None       # cached Cosh getws pointer
cdef object _cosh_exec_ptr = None        # cached Cosh exec pointer
cdef object _erf_getws_ptr = None        # cached Erf getws pointer
cdef object _erf_exec_ptr = None         # cached Erf exec pointer
cdef object _erfc_getws_ptr = None       # cached Erfc getws pointer
cdef object _erfc_exec_ptr = None        # cached Erfc exec pointer
cdef object _floor_getws_ptr = None      # cached Floor getws pointer
cdef object _floor_exec_ptr = None       # cached Floor exec pointer
cdef object _ceil_getws_ptr = None       # cached Ceil getws pointer
cdef object _ceil_exec_ptr = None        # cached Ceil exec pointer
cdef object _round_getws_ptr = None      # cached Round getws pointer
cdef object _round_exec_ptr = None       # cached Round exec pointer
cdef object _trunc_getws_ptr = None      # cached Trunc getws pointer
cdef object _trunc_exec_ptr = None       # cached Trunc exec pointer
cdef object _erfinv_getws_ptr = None     # cached Erfinv getws pointer
cdef object _erfinv_exec_ptr = None      # cached Erfinv exec pointer
cdef object _lerp_getws_ptr = None       # cached Lerp getws pointer
cdef object _lerp_exec_ptr = None        # cached Lerp exec pointer
cdef object _lerps_getws_ptr = None      # cached Lerps getws pointer
cdef object _lerps_exec_ptr = None       # cached Lerps exec pointer
cdef object _addcmul_getws_ptr = None    # cached Addcmul getws pointer
cdef object _addcmul_exec_ptr = None     # cached Addcmul exec pointer
cdef object _addcdiv_getws_ptr = None    # cached Addcdiv getws pointer
cdef object _addcdiv_exec_ptr = None     # cached Addcdiv exec pointer


cdef object _scalar_bytes_fn = None      # aclnn._scalar_bytes
cdef object _create_scalar_fn = None     # _aclnn_ffi.create_scalar
cdef object _destroy_scalar_fn = None    # _aclnn_ffi.destroy_scalar


cdef inline void _ensure_ffi_scalar_helpers() except *:
    global _scalar_bytes_fn, _create_scalar_fn, _destroy_scalar_fn
    if _scalar_bytes_fn is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    from candle._backends.npu.aclnn import _scalar_bytes as _sb
    _scalar_bytes_fn = _sb
    _create_scalar_fn = _ffi_ref.create_scalar
    _destroy_scalar_fn = _ffi_ref.destroy_scalar


cdef inline void _ensure_ffi_addcmul() except *:
    global _ffi_ref, _addcmul_getws_ptr, _addcmul_exec_ptr
    if _addcmul_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _addcmul_getws_ptr, _addcmul_exec_ptr = _ffi_ref.resolve_op("Addcmul")
    _ensure_ffi_scalar_helpers()


cdef inline void _ensure_ffi_addcdiv() except *:
    global _ffi_ref, _addcdiv_getws_ptr, _addcdiv_exec_ptr
    if _addcdiv_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _addcdiv_getws_ptr, _addcdiv_exec_ptr = _ffi_ref.resolve_op("Addcdiv")
    _ensure_ffi_scalar_helpers()




cdef inline void _ensure_ffi_where() except *:
    global _ffi_ref, _where_getws_ptr, _where_exec_ptr
    if _where_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _where_getws_ptr, _where_exec_ptr = _ffi_ref.resolve_op("SWhere")


cdef inline void _ensure_ffi_digamma() except *:
    global _ffi_ref, _digamma_getws_ptr, _digamma_exec_ptr
    if _digamma_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _digamma_getws_ptr, _digamma_exec_ptr = _ffi_ref.resolve_op("Digamma")


cdef inline void _ensure_ffi_lgamma() except *:
    global _ffi_ref, _lgamma_getws_ptr, _lgamma_exec_ptr
    if _lgamma_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _lgamma_getws_ptr, _lgamma_exec_ptr = _ffi_ref.resolve_op("Lgamma")


cdef inline void _ensure_ffi_sinc() except *:
    global _ffi_ref, _sinc_getws_ptr, _sinc_exec_ptr
    if _sinc_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _sinc_getws_ptr, _sinc_exec_ptr = _ffi_ref.resolve_op("Sinc")


cdef inline void _ensure_ffi_abs() except *:
    global _ffi_ref, _abs_getws_ptr, _abs_exec_ptr
    if _abs_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _abs_getws_ptr, _abs_exec_ptr = _ffi_ref.resolve_op("Abs")


cdef inline void _ensure_ffi_neg() except *:
    global _ffi_ref, _neg_getws_ptr, _neg_exec_ptr
    if _neg_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _neg_getws_ptr, _neg_exec_ptr = _ffi_ref.resolve_op("Neg")


cdef inline void _ensure_ffi_sign() except *:
    global _ffi_ref, _sign_getws_ptr, _sign_exec_ptr
    if _sign_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _sign_getws_ptr, _sign_exec_ptr = _ffi_ref.resolve_op("Sign")


cdef inline void _ensure_ffi_signbit() except *:
    global _ffi_ref, _signbit_getws_ptr, _signbit_exec_ptr
    if _signbit_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _signbit_getws_ptr, _signbit_exec_ptr = _ffi_ref.resolve_op("Signbit")


cdef inline void _ensure_ffi_isfinite() except *:
    global _ffi_ref, _isfinite_getws_ptr, _isfinite_exec_ptr
    if _isfinite_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _isfinite_getws_ptr, _isfinite_exec_ptr = _ffi_ref.resolve_op("IsFinite")


cdef inline void _ensure_ffi_isposinf() except *:
    global _ffi_ref, _isposinf_getws_ptr, _isposinf_exec_ptr
    if _isposinf_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _isposinf_getws_ptr, _isposinf_exec_ptr = _ffi_ref.resolve_op("IsPosInf")


cdef inline void _ensure_ffi_isneginf() except *:
    global _ffi_ref, _isneginf_getws_ptr, _isneginf_exec_ptr
    if _isneginf_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _isneginf_getws_ptr, _isneginf_exec_ptr = _ffi_ref.resolve_op("IsNegInf")


cdef inline void _ensure_ffi_logical_not() except *:
    global _ffi_ref, _logical_not_getws_ptr, _logical_not_exec_ptr
    if _logical_not_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _logical_not_getws_ptr, _logical_not_exec_ptr = _ffi_ref.resolve_op("LogicalNot")


cdef inline void _ensure_ffi_bitwise_not() except *:
    global _ffi_ref, _bitwise_not_getws_ptr, _bitwise_not_exec_ptr
    if _bitwise_not_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _bitwise_not_getws_ptr, _bitwise_not_exec_ptr = _ffi_ref.resolve_op("BitwiseNot")


cdef inline void _ensure_ffi_square() except *:
    global _ffi_ref, _square_getws_ptr, _square_exec_ptr
    if _square_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _square_getws_ptr, _square_exec_ptr = _ffi_ref.resolve_op("Square")


cdef inline void _ensure_ffi_exp() except *:
    global _ffi_ref, _exp_getws_ptr, _exp_exec_ptr
    if _exp_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _exp_getws_ptr, _exp_exec_ptr = _ffi_ref.resolve_op("Exp")


cdef inline void _ensure_ffi_expm1() except *:
    global _ffi_ref, _expm1_getws_ptr, _expm1_exec_ptr
    if _expm1_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _expm1_getws_ptr, _expm1_exec_ptr = _ffi_ref.resolve_op("Expm1")


cdef inline void _ensure_ffi_log1p() except *:
    global _ffi_ref, _log1p_getws_ptr, _log1p_exec_ptr
    if _log1p_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _log1p_getws_ptr, _log1p_exec_ptr = _ffi_ref.resolve_op("Log1p")


cdef inline void _ensure_ffi_log() except *:
    global _ffi_ref, _log_getws_ptr, _log_exec_ptr
    if _log_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _log_getws_ptr, _log_exec_ptr = _ffi_ref.resolve_op("Log")


cdef inline void _ensure_ffi_log2() except *:
    global _ffi_ref, _log2_getws_ptr, _log2_exec_ptr
    if _log2_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _log2_getws_ptr, _log2_exec_ptr = _ffi_ref.resolve_op("Log2")


cdef inline void _ensure_ffi_log10() except *:
    global _ffi_ref, _log10_getws_ptr, _log10_exec_ptr
    if _log10_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _log10_getws_ptr, _log10_exec_ptr = _ffi_ref.resolve_op("Log10")


cdef inline void _ensure_ffi_exp2() except *:
    global _ffi_ref, _exp2_getws_ptr, _exp2_exec_ptr
    if _exp2_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _exp2_getws_ptr, _exp2_exec_ptr = _ffi_ref.resolve_op("Exp2")


cdef inline void _ensure_ffi_asinh() except *:
    global _ffi_ref, _asinh_getws_ptr, _asinh_exec_ptr
    if _asinh_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _asinh_getws_ptr, _asinh_exec_ptr = _ffi_ref.resolve_op("Asinh")


cdef inline void _ensure_ffi_acosh() except *:
    global _ffi_ref, _acosh_getws_ptr, _acosh_exec_ptr
    if _acosh_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _acosh_getws_ptr, _acosh_exec_ptr = _ffi_ref.resolve_op("Acosh")


cdef inline void _ensure_ffi_atanh() except *:
    global _ffi_ref, _atanh_getws_ptr, _atanh_exec_ptr
    if _atanh_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _atanh_getws_ptr, _atanh_exec_ptr = _ffi_ref.resolve_op("Atanh")


cdef inline void _ensure_ffi_atan() except *:
    global _ffi_ref, _atan_getws_ptr, _atan_exec_ptr
    if _atan_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _atan_getws_ptr, _atan_exec_ptr = _ffi_ref.resolve_op("Atan")


cdef inline void _ensure_ffi_asin() except *:
    global _ffi_ref, _asin_getws_ptr, _asin_exec_ptr
    if _asin_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _asin_getws_ptr, _asin_exec_ptr = _ffi_ref.resolve_op("Asin")


cdef inline void _ensure_ffi_acos() except *:
    global _ffi_ref, _acos_getws_ptr, _acos_exec_ptr
    if _acos_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _acos_getws_ptr, _acos_exec_ptr = _ffi_ref.resolve_op("Acos")


cdef inline void _ensure_ffi_rsqrt() except *:
    global _ffi_ref, _rsqrt_getws_ptr, _rsqrt_exec_ptr
    if _rsqrt_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _rsqrt_getws_ptr, _rsqrt_exec_ptr = _ffi_ref.resolve_op("Rsqrt")


cdef inline void _ensure_ffi_sqrt() except *:
    global _ffi_ref, _sqrt_getws_ptr, _sqrt_exec_ptr
    if _sqrt_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _sqrt_getws_ptr, _sqrt_exec_ptr = _ffi_ref.resolve_op("Sqrt")


cdef inline void _ensure_ffi_sin() except *:
    global _ffi_ref, _sin_getws_ptr, _sin_exec_ptr
    if _sin_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _sin_getws_ptr, _sin_exec_ptr = _ffi_ref.resolve_op("Sin")


cdef inline void _ensure_ffi_cos() except *:
    global _ffi_ref, _cos_getws_ptr, _cos_exec_ptr
    if _cos_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _cos_getws_ptr, _cos_exec_ptr = _ffi_ref.resolve_op("Cos")


cdef inline void _ensure_ffi_tan() except *:
    global _ffi_ref, _tan_getws_ptr, _tan_exec_ptr
    if _tan_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _tan_getws_ptr, _tan_exec_ptr = _ffi_ref.resolve_op("Tan")


cdef inline void _ensure_ffi_tanh() except *:
    global _ffi_ref, _tanh_getws_ptr, _tanh_exec_ptr
    if _tanh_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _tanh_getws_ptr, _tanh_exec_ptr = _ffi_ref.resolve_op("Tanh")


cdef inline void _ensure_ffi_sigmoid() except *:
    global _ffi_ref, _sigmoid_getws_ptr, _sigmoid_exec_ptr
    if _sigmoid_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _sigmoid_getws_ptr, _sigmoid_exec_ptr = _ffi_ref.resolve_op("Sigmoid")


cdef inline void _ensure_ffi_relu() except *:
    global _ffi_ref, _relu_getws_ptr, _relu_exec_ptr
    if _relu_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _relu_getws_ptr, _relu_exec_ptr = _ffi_ref.resolve_op("Relu")


cdef inline void _ensure_ffi_leaky_relu() except *:
    global _ffi_ref, _leaky_relu_getws_ptr, _leaky_relu_exec_ptr
    if _leaky_relu_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _leaky_relu_getws_ptr, _leaky_relu_exec_ptr = _ffi_ref.resolve_op("LeakyRelu")
    _ensure_ffi_scalar_helpers()


cdef inline void _ensure_ffi_elu() except *:
    global _ffi_ref, _elu_getws_ptr, _elu_exec_ptr
    if _elu_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _elu_getws_ptr, _elu_exec_ptr = _ffi_ref.resolve_op("Elu")
    _ensure_ffi_scalar_helpers()


cdef inline void _ensure_ffi_dropout() except *:
    global _ffi_ref, _dropout_gen_mask_getws_ptr, _dropout_gen_mask_exec_ptr
    global _dropout_do_mask_getws_ptr, _dropout_do_mask_exec_ptr
    if _dropout_gen_mask_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _dropout_gen_mask_getws_ptr, _dropout_gen_mask_exec_ptr = _ffi_ref.resolve_op("DropoutGenMask")
    _dropout_do_mask_getws_ptr, _dropout_do_mask_exec_ptr = _ffi_ref.resolve_op("DropoutDoMask")


cdef inline void _ensure_npu_module() except *:
    global _npu_mod_ref
    if _npu_mod_ref is not None:
        return
    _npu_mod_ref = importlib.import_module("candle.npu")


cdef inline void _ensure_ffi_embedding() except *:
    global _ffi_ref, _embedding_getws_ptr, _embedding_exec_ptr
    if _embedding_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _embedding_getws_ptr, _embedding_exec_ptr = _ffi_ref.resolve_op("Embedding")


cdef inline void _ensure_ffi_prelu() except *:
    global _ffi_ref, _prelu_getws_ptr, _prelu_exec_ptr
    if _prelu_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _prelu_getws_ptr, _prelu_exec_ptr = _ffi_ref.resolve_op("Prelu")


cdef inline void _ensure_ffi_softplus() except *:
    global _ffi_ref, _softplus_getws_ptr, _softplus_exec_ptr
    if _softplus_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _softplus_getws_ptr, _softplus_exec_ptr = _ffi_ref.resolve_op("Softplus")
    _ensure_ffi_scalar_helpers()


cdef inline void _ensure_ffi_softmax() except *:
    global _ffi_ref, _softmax_getws_ptr, _softmax_exec_ptr
    if _softmax_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _softmax_getws_ptr, _softmax_exec_ptr = _ffi_ref.resolve_op("Softmax")


cdef inline void _ensure_ffi_log_softmax() except *:
    global _ffi_ref, _log_softmax_getws_ptr, _log_softmax_exec_ptr
    if _log_softmax_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _log_softmax_getws_ptr, _log_softmax_exec_ptr = _ffi_ref.resolve_op("LogSoftmax")


cdef inline void _ensure_ffi_hardtanh() except *:
    global _ffi_ref, _hardtanh_getws_ptr, _hardtanh_exec_ptr
    if _hardtanh_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _hardtanh_getws_ptr, _hardtanh_exec_ptr = _ffi_ref.resolve_op("Hardtanh")
    _ensure_ffi_scalar_helpers()


cdef inline void _ensure_ffi_gelu() except *:
    global _ffi_ref, _gelu_getws_ptr, _gelu_exec_ptr
    if _gelu_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _gelu_getws_ptr, _gelu_exec_ptr = _ffi_ref.resolve_op("Gelu")


cdef inline void _ensure_ffi_silu() except *:
    global _ffi_ref, _silu_getws_ptr, _silu_exec_ptr
    if _silu_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _silu_getws_ptr, _silu_exec_ptr = _ffi_ref.resolve_op("Silu")


cdef inline void _ensure_ffi_mish() except *:
    global _ffi_ref, _mish_getws_ptr, _mish_exec_ptr
    if _mish_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _mish_getws_ptr, _mish_exec_ptr = _ffi_ref.resolve_op("Mish")


cdef inline void _ensure_ffi_sinh() except *:
    global _ffi_ref, _sinh_getws_ptr, _sinh_exec_ptr
    if _sinh_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _sinh_getws_ptr, _sinh_exec_ptr = _ffi_ref.resolve_op("Sinh")


cdef inline void _ensure_ffi_cosh() except *:
    global _ffi_ref, _cosh_getws_ptr, _cosh_exec_ptr
    if _cosh_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _cosh_getws_ptr, _cosh_exec_ptr = _ffi_ref.resolve_op("Cosh")


cdef inline void _ensure_ffi_erf() except *:
    global _ffi_ref, _erf_getws_ptr, _erf_exec_ptr
    if _erf_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _erf_getws_ptr, _erf_exec_ptr = _ffi_ref.resolve_op("Erf")


cdef inline void _ensure_ffi_erfc() except *:
    global _ffi_ref, _erfc_getws_ptr, _erfc_exec_ptr
    if _erfc_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _erfc_getws_ptr, _erfc_exec_ptr = _ffi_ref.resolve_op("Erfc")


cdef inline void _ensure_ffi_floor() except *:
    global _ffi_ref, _floor_getws_ptr, _floor_exec_ptr
    if _floor_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _floor_getws_ptr, _floor_exec_ptr = _ffi_ref.resolve_op("Floor")


cdef inline void _ensure_ffi_ceil() except *:
    global _ffi_ref, _ceil_getws_ptr, _ceil_exec_ptr
    if _ceil_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _ceil_getws_ptr, _ceil_exec_ptr = _ffi_ref.resolve_op("Ceil")


cdef inline void _ensure_ffi_round() except *:
    global _ffi_ref, _round_getws_ptr, _round_exec_ptr
    if _round_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _round_getws_ptr, _round_exec_ptr = _ffi_ref.resolve_op("Round")


cdef inline void _ensure_ffi_trunc() except *:
    global _ffi_ref, _trunc_getws_ptr, _trunc_exec_ptr
    if _trunc_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _trunc_getws_ptr, _trunc_exec_ptr = _ffi_ref.resolve_op("Trunc")


cdef inline void _ensure_ffi_erfinv() except *:
    global _ffi_ref, _erfinv_getws_ptr, _erfinv_exec_ptr
    if _erfinv_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _erfinv_getws_ptr, _erfinv_exec_ptr = _ffi_ref.resolve_op("Erfinv")


cdef inline void _ensure_ffi_lerp() except *:
    global _ffi_ref, _lerp_getws_ptr, _lerp_exec_ptr
    if _lerp_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _lerp_getws_ptr, _lerp_exec_ptr = _ffi_ref.resolve_op("Lerp")


cdef inline void _ensure_ffi_lerps() except *:
    global _ffi_ref, _lerps_getws_ptr, _lerps_exec_ptr
    if _lerps_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _lerps_getws_ptr, _lerps_exec_ptr = _ffi_ref.resolve_op("Lerps")
    _ensure_ffi_scalar_helpers()



def fast_lerp_tensor(a, b, weight):
    """Optimized lerp(a, b, weight_tensor) that calls _ffi.four_tensor_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_lerp()

    cdef int dev_idx
    _validate_npu_binary(a, b, "lerp", &dev_idx)
    if weight.device.type != "npu":
        raise ValueError("NPU lerp expects NPU tensors")
    if weight.dtype != a.dtype:
        raise ValueError("NPU lerp requires matching dtypes")

    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    py_a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    py_b_shape = (<TensorImpl>b)._shape_tuple if isinstance(b, TensorImpl) else b.shape
    py_w_shape = (<TensorImpl>weight)._shape_tuple if isinstance(weight, TensorImpl) else weight.shape
    cdef int a_ndim = len(py_a_shape)
    cdef int b_ndim = len(py_b_shape)
    cdef int w_ndim = len(py_w_shape)

    if a_ndim > MAX_NDIM or b_ndim > MAX_NDIM or w_ndim > MAX_NDIM:
        raise ValueError(f"ndim exceeds MAX_NDIM ({MAX_NDIM})")

    cdef int64_t[MAX_NDIM] a_shape_buf, b_shape_buf, w_shape_buf
    cdef int64_t[MAX_NDIM] tmp_shape_buf, out_shape_buf, out_stride_buf
    _fill_shape(py_a_shape, a_shape_buf, a_ndim)
    _fill_shape(py_b_shape, b_shape_buf, b_ndim)
    _fill_shape(py_w_shape, w_shape_buf, w_ndim)

    cdef int tmp_ndim
    cdef int out_ndim
    cdef int64_t n
    with nogil:
        tmp_ndim = c_broadcast_shape(
            a_shape_buf, a_ndim, b_shape_buf, b_ndim, tmp_shape_buf)
        out_ndim = c_broadcast_shape(
            tmp_shape_buf, tmp_ndim, w_shape_buf, w_ndim, out_shape_buf)
        c_contiguous_stride(out_shape_buf, out_ndim, out_stride_buf)
        n = c_numel(out_shape_buf, out_ndim)

    out_shape = _to_tuple(out_shape_buf, out_ndim)
    out_stride = _to_tuple(out_stride_buf, out_ndim)

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_fl = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fl, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fl, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, b_ptr, w_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    if isinstance(b, TensorImpl):
        b_ptr = <uintptr_t>(<TensorImpl>b)._storage._untyped._device_ptr
    else:
        b_ptr = <uintptr_t>b.storage().data_ptr()
    if isinstance(weight, TensorImpl):
        w_ptr = <uintptr_t>(<TensorImpl>weight)._storage._untyped._device_ptr
    else:
        w_ptr = <uintptr_t>weight.storage().data_ptr()
    o_ptr = out_ptr

    cdef uintptr_t stream_raw = int(stream.stream)
    ws_size, executor = _ffi_ref.four_tensor_op(
        _lerp_getws_ptr, _lerp_exec_ptr,
        py_a_shape, a.stride,
        py_b_shape, b.stride,
        py_w_shape, weight.stride,
        out_shape, out_stride,
        dtype_code, dtype_code, dtype_code, dtype_code, 2,
        a_ptr, b_ptr, w_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _lerp_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnLerp execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)

    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)



def fast_where(cond, x, y):
    """Optimized where(cond, x, y) that calls _ffi.where_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_where()

    cdef int dev_idx
    if cond.device.type != "npu" or x.device.type != "npu" or y.device.type != "npu":
        raise ValueError("NPU where expects NPU tensors")
    if x.dtype != y.dtype:
        raise ValueError("NPU where requires matching dtypes")
    dev_idx = x.device.index or 0

    x_dev = _device_obj_fast(x)
    x_dtype = x.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    py_cond_shape = (<TensorImpl>cond)._shape_tuple if isinstance(cond, TensorImpl) else cond.shape
    py_x_shape = (<TensorImpl>x)._shape_tuple if isinstance(x, TensorImpl) else x.shape
    py_y_shape = (<TensorImpl>y)._shape_tuple if isinstance(y, TensorImpl) else y.shape

    out_shape = py_x_shape
    out_stride = x.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(x_dtype)
    cdef int64_t alloc_size_fw = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fw, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fw, stream=stream.stream)

    cdef uintptr_t cond_ptr, x_ptr, y_ptr, o_ptr
    if isinstance(cond, TensorImpl):
        cond_ptr = <uintptr_t>(<TensorImpl>cond)._storage._untyped._device_ptr
    else:
        cond_ptr = <uintptr_t>cond.storage().data_ptr()
    if isinstance(x, TensorImpl):
        x_ptr = <uintptr_t>(<TensorImpl>x)._storage._untyped._device_ptr
    else:
        x_ptr = <uintptr_t>x.storage().data_ptr()
    if isinstance(y, TensorImpl):
        y_ptr = <uintptr_t>(<TensorImpl>y)._storage._untyped._device_ptr
    else:
        y_ptr = <uintptr_t>y.storage().data_ptr()
    o_ptr = out_ptr

    cdef int x_dtype_code = _dtype_to_acl_code(x_dtype)
    cdef uintptr_t stream_raw = int(stream.stream)
    ws_size, executor = _ffi_ref.where_op(
        _where_getws_ptr, _where_exec_ptr,
        py_cond_shape, cond.stride,
        py_x_shape, x.stride,
        py_y_shape, y.stride,
        out_shape, out_stride,
        12, x_dtype_code, x_dtype_code, x_dtype_code, 2,
        cond_ptr, x_ptr, y_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _where_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnSWhere execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)

    return _cy_make_npu_tensor(out_ptr, n, x_dtype, x_dev, out_shape, out_stride)



def fast_digamma(a):
    """Optimized out-of-place digamma(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_digamma()

    if a.device.type != "npu":
        raise ValueError("NPU digamma expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_fd = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fd, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fd, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _digamma_getws_ptr, _digamma_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        dtype_code, dtype_code, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _digamma_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnDigamma execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_lgamma(a):
    """Optimized out-of-place lgamma(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_lgamma()

    if a.device.type != "npu":
        raise ValueError("NPU lgamma expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_flg = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_flg, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_flg, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _lgamma_getws_ptr, _lgamma_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        dtype_code, dtype_code, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _lgamma_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnLgamma execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_sinc(a):
    """Optimized out-of-place sinc(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_sinc()

    if a.device.type != "npu":
        raise ValueError("NPU sinc expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_fs = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fs, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fs, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _sinc_getws_ptr, _sinc_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        dtype_code, dtype_code, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _sinc_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnSinc execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_abs(a):
    """Optimized out-of-place abs(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_abs()

    if a.device.type != "npu":
        raise ValueError("NPU abs expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_fabs = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fabs, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fabs, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _abs_getws_ptr, _abs_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        dtype_code, dtype_code, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _abs_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnAbs execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_neg(a):
    """Optimized out-of-place neg(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_neg()

    if a.device.type != "npu":
        raise ValueError("NPU neg expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_fneg = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fneg, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fneg, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _neg_getws_ptr, _neg_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        dtype_code, dtype_code, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _neg_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnNeg execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_sign(a):
    """Optimized out-of-place sign(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_sign()

    if a.device.type != "npu":
        raise ValueError("NPU sign expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_fsign = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fsign, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fsign, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _sign_getws_ptr, _sign_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        dtype_code, dtype_code, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _sign_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnSign execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_signbit(a):
    """Optimized out-of-place signbit(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_signbit()

    if a.device.type != "npu":
        raise ValueError("NPU signbit expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    from candle._dtype import bool as _bool_dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(_bool_dtype)
    cdef int64_t alloc_size_fsignbit = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fsignbit, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fsignbit, stream=stream.stream)

    cdef int in_dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _signbit_getws_ptr, _signbit_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        in_dtype_code, 12, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _signbit_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnSignbit execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, _bool_dtype, a_dev, out_shape, out_stride)


def fast_isfinite(a):
    """Optimized out-of-place isfinite(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_isfinite()

    if a.device.type != "npu":
        raise ValueError("NPU isfinite expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    from candle._dtype import bool as _bool_dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(_bool_dtype)
    cdef int64_t alloc_size_fisfinite = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fisfinite, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fisfinite, stream=stream.stream)

    cdef int in_dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _isfinite_getws_ptr, _isfinite_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        in_dtype_code, 12, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _isfinite_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnIsFinite execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, _bool_dtype, a_dev, out_shape, out_stride)


def fast_isposinf(a):
    """Optimized out-of-place isposinf(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_isposinf()

    if a.device.type != "npu":
        raise ValueError("NPU isposinf expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    from candle._dtype import bool as _bool_dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(_bool_dtype)
    cdef int64_t alloc_size_fisposinf = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fisposinf, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fisposinf, stream=stream.stream)

    cdef int in_dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _isposinf_getws_ptr, _isposinf_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        in_dtype_code, 12, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _isposinf_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnIsPosInf execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, _bool_dtype, a_dev, out_shape, out_stride)


def fast_isneginf(a):
    """Optimized out-of-place isneginf(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_isneginf()

    if a.device.type != "npu":
        raise ValueError("NPU isneginf expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    from candle._dtype import bool as _bool_dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(_bool_dtype)
    cdef int64_t alloc_size_fisneginf = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fisneginf, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fisneginf, stream=stream.stream)

    cdef int in_dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _isneginf_getws_ptr, _isneginf_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        in_dtype_code, 12, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _isneginf_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnIsNegInf execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, _bool_dtype, a_dev, out_shape, out_stride)


def fast_logical_not(a):
    """Optimized out-of-place logical_not(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_logical_not()

    if a.device.type != "npu":
        raise ValueError("NPU logical_not expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    from candle._dtype import bool as _bool_dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(_bool_dtype)
    cdef int64_t alloc_size_flogicalnot = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_flogicalnot, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_flogicalnot, stream=stream.stream)

    cdef int in_dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _logical_not_getws_ptr, _logical_not_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        in_dtype_code, 12, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _logical_not_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnLogicalNot execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, _bool_dtype, a_dev, out_shape, out_stride)


def fast_bitwise_not(a):
    """Optimized out-of-place bitwise_not(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_bitwise_not()

    if a.device.type != "npu":
        raise ValueError("NPU bitwise_not expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_fbitwisenot = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fbitwisenot, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fbitwisenot, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _bitwise_not_getws_ptr, _bitwise_not_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        dtype_code, dtype_code, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _bitwise_not_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnBitwiseNot execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_square(a):
    """Optimized out-of-place square(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_square()

    if a.device.type != "npu":
        raise ValueError("NPU square expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_fsquare = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fsquare, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fsquare, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _square_getws_ptr, _square_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        dtype_code, dtype_code, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _square_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnSquare execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_exp(a):
    """Optimized out-of-place exp(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_exp()

    if a.device.type != "npu":
        raise ValueError("NPU exp expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_fx = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fx, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fx, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _exp_getws_ptr, _exp_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        dtype_code, dtype_code, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _exp_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnExp execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_expm1(a):
    """Optimized out-of-place expm1(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_expm1()

    if a.device.type != "npu":
        raise ValueError("NPU expm1 expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_fexpm1 = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fexpm1, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fexpm1, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _expm1_getws_ptr, _expm1_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        dtype_code, dtype_code, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _expm1_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnExpm1 execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_log1p(a):
    """Optimized out-of-place log1p(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_log1p()

    if a.device.type != "npu":
        raise ValueError("NPU log1p expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_flog1p = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_flog1p, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_flog1p, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _log1p_getws_ptr, _log1p_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        dtype_code, dtype_code, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _log1p_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnLog1p execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_log(a):
    """Optimized out-of-place log(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_log()

    if a.device.type != "npu":
        raise ValueError("NPU log expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_flg = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_flg, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_flg, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _log_getws_ptr, _log_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        dtype_code, dtype_code, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _log_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnLog execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_rsqrt(a):
    """Optimized out-of-place rsqrt(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_rsqrt()

    if a.device.type != "npu":
        raise ValueError("NPU rsqrt expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_frsqrt = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_frsqrt, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_frsqrt, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _rsqrt_getws_ptr, _rsqrt_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        dtype_code, dtype_code, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _rsqrt_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnRsqrt execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_sqrt(a):
    """Optimized out-of-place sqrt(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_sqrt()

    if a.device.type != "npu":
        raise ValueError("NPU sqrt expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_fsq = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fsq, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fsq, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _sqrt_getws_ptr, _sqrt_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        dtype_code, dtype_code, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _sqrt_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnSqrt execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_sin(a):
    """Optimized out-of-place sin(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_sin()

    if a.device.type != "npu":
        raise ValueError("NPU sin expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_fsin = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fsin, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fsin, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _sin_getws_ptr, _sin_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        dtype_code, dtype_code, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _sin_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnSin execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_cos(a):
    """Optimized out-of-place cos(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_cos()

    if a.device.type != "npu":
        raise ValueError("NPU cos expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_fcos = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fcos, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fcos, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _cos_getws_ptr, _cos_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        dtype_code, dtype_code, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _cos_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnCos execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_tan(a):
    """Optimized out-of-place tan(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_tan()

    if a.device.type != "npu":
        raise ValueError("NPU tan expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_ftan = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_ftan, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_ftan, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _tan_getws_ptr, _tan_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        dtype_code, dtype_code, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _tan_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnTan execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_tanh(a):
    """Optimized out-of-place tanh(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_tanh()

    if a.device.type != "npu":
        raise ValueError("NPU tanh expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_ftanh = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_ftanh, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_ftanh, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _tanh_getws_ptr, _tanh_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        dtype_code, dtype_code, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _tanh_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnTanh execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)



def fast_sigmoid(a):
    """Optimized out-of-place sigmoid(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_sigmoid()

    if a.device.type != "npu":
        raise ValueError("NPU sigmoid expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_fsigmoid = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fsigmoid, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fsigmoid, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _sigmoid_getws_ptr, _sigmoid_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        dtype_code, dtype_code, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _sigmoid_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnSigmoid execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_relu(a):
    """Optimized out-of-place relu(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_relu()

    if a.device.type != "npu":
        raise ValueError("NPU relu expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_frelu = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_frelu, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_frelu, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _relu_getws_ptr, _relu_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        dtype_code, dtype_code, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _relu_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnRelu execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_leaky_relu(a, negative_slope):
    """Optimized out-of-place leaky_relu(a) that calls _ffi.leaky_relu_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_leaky_relu()

    if a.device.type != "npu":
        raise ValueError("NPU leaky_relu expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_fleaky = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fleaky, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fleaky, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr

    scalar_handle = _create_scalar_fn(_scalar_bytes_fn(negative_slope, a_dtype), dtype_code)
    cdef uintptr_t stream_raw = int(stream.stream)
    try:
        ws_size, executor = _ffi_ref.leaky_relu_op(
            _leaky_relu_getws_ptr, _leaky_relu_exec_ptr,
            a.shape, a.stride,
            out_shape, out_stride,
            dtype_code, 2,
            a_ptr, o_ptr,
            scalar_handle,
            stream_raw)

        if ws_size:
            workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            try:
                ret = _ffi_ref.execute(
                    _leaky_relu_exec_ptr, int(workspace_ptr), ws_size,
                    executor, stream_raw)
                if ret != 0:
                    raise RuntimeError(f"aclnnLeakyRelu execute failed: {ret}")
            finally:
                runtime.defer_raw_free(workspace_ptr)

        _defer_executor_fn(executor)
    finally:
        _destroy_scalar_fn(int(scalar_handle))

    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_elu(a, alpha):
    """Optimized out-of-place elu(a) that calls _ffi.tensor_three_scalars_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_elu()

    if a.device.type != "npu":
        raise ValueError("NPU elu expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_felu = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_felu, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_felu, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr

    alpha_scalar = _create_scalar_fn(_scalar_bytes_fn(alpha, a_dtype), dtype_code)
    scale_scalar = _create_scalar_fn(_scalar_bytes_fn(1.0, a_dtype), dtype_code)
    input_scale_scalar = _create_scalar_fn(_scalar_bytes_fn(1.0, a_dtype), dtype_code)
    cdef uintptr_t stream_raw = int(stream.stream)
    try:
        ws_size, executor = _ffi_ref.tensor_three_scalars_op(
            _elu_getws_ptr, _elu_exec_ptr,
            a.shape, a.stride,
            out_shape, out_stride,
            dtype_code, dtype_code, 2,
            a_ptr, o_ptr,
            alpha_scalar, scale_scalar, input_scale_scalar,
            stream_raw)

        if ws_size:
            workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            try:
                ret = _ffi_ref.execute(
                    _elu_exec_ptr, int(workspace_ptr), ws_size,
                    executor, stream_raw)
                if ret != 0:
                    raise RuntimeError(f"aclnnElu execute failed: {ret}")
            finally:
                runtime.defer_raw_free(workspace_ptr)

        _defer_executor_fn(executor)
    finally:
        _destroy_scalar_fn(int(alpha_scalar))
        _destroy_scalar_fn(int(scale_scalar))
        _destroy_scalar_fn(int(input_scale_scalar))

    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_hardtanh(a, min_val, max_val):
    """Optimized out-of-place hardtanh(a) that calls _ffi.tensor_two_scalars_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_hardtanh()

    if a.device.type != "npu":
        raise ValueError("NPU hardtanh expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for size in out_shape:
        n *= size

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_fhardtanh = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fhardtanh, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fhardtanh, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr

    min_scalar = _create_scalar_fn(_scalar_bytes_fn(min_val, a_dtype), dtype_code)
    max_scalar = _create_scalar_fn(_scalar_bytes_fn(max_val, a_dtype), dtype_code)
    cdef uintptr_t stream_raw = int(stream.stream)
    try:
        ws_size, executor = _ffi_ref.tensor_two_scalars_op(
            _hardtanh_getws_ptr, _hardtanh_exec_ptr,
            a.shape, a.stride,
            out_shape, out_stride,
            dtype_code, dtype_code, 2,
            a_ptr, o_ptr,
            min_scalar, max_scalar,
            stream_raw)

        if ws_size:
            workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            try:
                ret = _ffi_ref.execute(
                    _hardtanh_exec_ptr, int(workspace_ptr), ws_size,
                    executor, stream_raw)
                if ret != 0:
                    raise RuntimeError(f"aclnnHardtanh execute failed: {ret}")
            finally:
                runtime.defer_raw_free(workspace_ptr)

        _defer_executor_fn(executor)
    finally:
        _destroy_scalar_fn(int(min_scalar))
        _destroy_scalar_fn(int(max_scalar))

    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_embedding(weight, indices):
    """Optimized out-of-place embedding(weight, indices) that calls _ffi.binary_two_inputs_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_embedding()

    if weight.device.type != "npu" or indices.device.type != "npu":
        raise ValueError("NPU embedding expects NPU tensors")
    cdef int dev_idx = weight.device.index or 0
    weight_dev = _device_obj_fast(weight)
    weight_dtype = weight.dtype
    indices_dtype = indices.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    weight_shape = (<TensorImpl>weight)._shape_tuple if isinstance(weight, TensorImpl) else weight.shape
    weight_stride = weight.stride
    indices_shape = (<TensorImpl>indices)._shape_tuple if isinstance(indices, TensorImpl) else indices.shape
    indices_stride = indices.stride
    embedding_dim = weight_shape[1] if len(weight_shape) > 1 else weight_shape[0]
    out_shape = indices_shape + (embedding_dim,)
    cdef int out_ndim = len(out_shape)
    cdef int64_t[MAX_NDIM] out_shape_buf, out_stride_buf
    if out_ndim > MAX_NDIM:
        raise ValueError(f"ndim exceeds MAX_NDIM ({MAX_NDIM})")
    _fill_shape(out_shape, out_shape_buf, out_ndim)
    with nogil:
        c_contiguous_stride(out_shape_buf, out_ndim, out_stride_buf)
    out_stride = _to_tuple(out_stride_buf, out_ndim)
    cdef int64_t n = c_numel(out_shape_buf, out_ndim)

    cdef int isize = c_dtype_itemsize(weight_dtype)
    cdef int64_t alloc_size_fembedding = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fembedding, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fembedding, stream=stream.stream)

    cdef int weight_dtype_code = _dtype_to_acl_code(weight_dtype)
    cdef int indices_dtype_code = _dtype_to_acl_code(indices_dtype)
    cdef uintptr_t weight_ptr, indices_ptr, o_ptr
    if isinstance(weight, TensorImpl):
        weight_ptr = <uintptr_t>(<TensorImpl>weight)._storage._untyped._device_ptr
    else:
        weight_ptr = <uintptr_t>weight.storage().data_ptr()
    if isinstance(indices, TensorImpl):
        indices_ptr = <uintptr_t>(<TensorImpl>indices)._storage._untyped._device_ptr
    else:
        indices_ptr = <uintptr_t>indices.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.binary_two_inputs_op(
        _embedding_getws_ptr, _embedding_exec_ptr,
        weight_shape, weight_stride,
        indices_shape, indices_stride,
        out_shape, out_stride,
        weight_dtype_code, indices_dtype_code, weight_dtype_code,
        2,
        weight_ptr, indices_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _embedding_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnEmbedding execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, weight_dtype, weight_dev, out_shape, out_stride)


def fast_dropout(a, p):
    """Optimized out-of-place dropout(a, p) that bypasses Python dropout wrappers."""
    _ensure_npu_imports()
    _ensure_ffi_dropout()
    _ensure_npu_module()

    if a.device.type != "npu":
        raise ValueError("NPU dropout expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int out_ndim = len(out_shape)
    cdef int64_t n = 1
    for size in out_shape:
        n *= size

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_fdropout = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fdropout, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fdropout, stream=stream.stream)

    mask_numel = (n + 127) // 128 * 128 // 8
    if dev_idx == 0:
        mask_ptr = _fast_allocator_dev0.malloc(mask_numel, stream=stream.stream)
    else:
        mask_ptr = _get_allocator_fn_ref(dev_idx).malloc(mask_numel, stream=stream.stream)

    seed, offset = _npu_mod_ref._get_and_advance_offset(device_index=dev_idx, increment=10)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef int mask_dtype_code = _dtype_to_acl_code("uint8")
    cdef uintptr_t a_ptr, o_ptr, m_ptr
    cdef tuple mask_shape = (mask_numel,)
    cdef tuple mask_stride = (1,)
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    m_ptr = mask_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.output_tensor_int_array_double_two_ints_op(
        _dropout_gen_mask_getws_ptr, _dropout_gen_mask_exec_ptr,
        mask_shape, mask_stride,
        tuple(out_shape), float(p), int(seed), int(offset),
        mask_dtype_code, 2,
        m_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _dropout_gen_mask_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnDropoutGenMask execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)
    _defer_executor_fn(executor)

    ws_size, executor = _ffi_ref.two_tensor_one_double_op(
        _dropout_do_mask_getws_ptr, _dropout_do_mask_exec_ptr,
        out_shape, out_stride,
        mask_shape, mask_stride,
        out_shape, out_stride,
        float(p),
        dtype_code, mask_dtype_code, dtype_code,
        2,
        a_ptr, m_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _dropout_do_mask_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnDropoutDoMask execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)
    _defer_executor_fn(executor)

    out = _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)
    return out, int(mask_ptr), int(mask_numel)


def fast_prelu(a, weight):
    """Optimized out-of-place prelu(a, weight) that calls _ffi.binary_two_inputs_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_prelu()

    if a.device.type != "npu" or weight.device.type != "npu":
        raise ValueError("NPU prelu expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    weight_shape = (<TensorImpl>weight)._shape_tuple if isinstance(weight, TensorImpl) else weight.shape
    weight_stride = weight.stride
    cdef int64_t n = 1
    for size in out_shape:
        n *= size

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_fprelu = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fprelu, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fprelu, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, w_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    if isinstance(weight, TensorImpl):
        w_ptr = <uintptr_t>(<TensorImpl>weight)._storage._untyped._device_ptr
    else:
        w_ptr = <uintptr_t>weight.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.binary_two_inputs_op(
        _prelu_getws_ptr, _prelu_exec_ptr,
        a.shape, a.stride,
        weight_shape, weight_stride,
        out_shape, out_stride,
        dtype_code, dtype_code, dtype_code,
        2,
        a_ptr, w_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _prelu_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnPrelu execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_softplus(a, beta, threshold):

    """Optimized out-of-place softplus(a, beta, threshold) that calls _ffi.tensor_two_scalars_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_softplus()

    if a.device.type != "npu":
        raise ValueError("NPU softplus expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for size in out_shape:
        n *= size

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_fsoftplus = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fsoftplus, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fsoftplus, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr

    beta_scalar = _create_scalar_fn(_scalar_bytes_fn(beta, a_dtype), dtype_code)
    threshold_scalar = _create_scalar_fn(_scalar_bytes_fn(threshold, a_dtype), dtype_code)
    cdef uintptr_t stream_raw = int(stream.stream)
    try:
        ws_size, executor = _ffi_ref.tensor_two_scalars_op(
            _softplus_getws_ptr, _softplus_exec_ptr,
            a.shape, a.stride,
            out_shape, out_stride,
            dtype_code, dtype_code, 2,
            a_ptr, o_ptr,
            beta_scalar, threshold_scalar,
            stream_raw)

        if ws_size:
            workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            try:
                ret = _ffi_ref.execute(
                    _softplus_exec_ptr, int(workspace_ptr), ws_size,
                    executor, stream_raw)
                if ret != 0:
                    raise RuntimeError(f"aclnnSoftplus execute failed: {ret}")
            finally:
                runtime.defer_raw_free(workspace_ptr)

        _defer_executor_fn(executor)
    finally:
        _destroy_scalar_fn(int(beta_scalar))
        _destroy_scalar_fn(int(threshold_scalar))

    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_softmax(a, dim):
    """Optimized out-of-place softmax(a, dim) that calls _ffi.axis_unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_softmax()

    if a.device.type != "npu":
        raise ValueError("NPU softmax expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    if dim < 0:
        dim += len(out_shape)
    cdef int64_t n = 1
    for size in out_shape:
        n *= size

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_fsoftmax = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fsoftmax, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fsoftmax, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.axis_unary_op(
        _softmax_getws_ptr, _softmax_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        int(dim), dtype_code, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _softmax_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnSoftmax execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_log_softmax(a, dim):
    """Optimized out-of-place log_softmax(a, dim) that calls _ffi.axis_unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_log_softmax()

    if a.device.type != "npu":
        raise ValueError("NPU log_softmax expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    if dim < 0:
        dim += len(out_shape)
    cdef int64_t n = 1
    for size in out_shape:
        n *= size

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_flogsoftmax = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_flogsoftmax, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_flogsoftmax, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.axis_unary_op(
        _log_softmax_getws_ptr, _log_softmax_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        int(dim), dtype_code, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _log_softmax_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnLogSoftmax execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_gelu(a):
    """Optimized out-of-place gelu(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_gelu()

    if a.device.type != "npu":
        raise ValueError("NPU gelu expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_fgelu = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fgelu, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fgelu, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _gelu_getws_ptr, _gelu_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        dtype_code, dtype_code, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _gelu_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnGelu execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_silu(a):
    """Optimized out-of-place silu(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_silu()

    if a.device.type != "npu":
        raise ValueError("NPU silu expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_fsilu = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fsilu, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fsilu, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _silu_getws_ptr, _silu_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        dtype_code, dtype_code, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _silu_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnSilu execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_mish(a):
    """Optimized out-of-place mish(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_mish()

    if a.device.type != "npu":
        raise ValueError("NPU mish expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_fmish = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fmish, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fmish, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _mish_getws_ptr, _mish_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        dtype_code, dtype_code, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _mish_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnMish execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_sinh(a):
    """Optimized out-of-place sinh(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_sinh()

    if a.device.type != "npu":
        raise ValueError("NPU sinh expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_fsinh = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fsinh, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fsinh, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _sinh_getws_ptr, _sinh_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        dtype_code, dtype_code, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _sinh_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnSinh execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_cosh(a):
    """Optimized out-of-place cosh(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_cosh()

    if a.device.type != "npu":
        raise ValueError("NPU cosh expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_fcosh = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fcosh, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fcosh, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _cosh_getws_ptr, _cosh_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        dtype_code, dtype_code, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _cosh_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnCosh execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_erf(a):
    """Optimized out-of-place erf(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_erf()

    if a.device.type != "npu":
        raise ValueError("NPU erf expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_ferf = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_ferf, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_ferf, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _erf_getws_ptr, _erf_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        dtype_code, dtype_code, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _erf_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnErf execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_erfc(a):
    """Optimized out-of-place erfc(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_erfc()

    if a.device.type != "npu":
        raise ValueError("NPU erfc expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_ferfc = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_ferfc, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_ferfc, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _erfc_getws_ptr, _erfc_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        dtype_code, dtype_code, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _erfc_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnErfc execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_floor(a):
    """Optimized out-of-place floor(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_floor()

    if a.device.type != "npu":
        raise ValueError("NPU floor expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_ffloor = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_ffloor, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_ffloor, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _floor_getws_ptr, _floor_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        dtype_code, dtype_code, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _floor_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnFloor execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_ceil(a):
    """Optimized out-of-place ceil(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_ceil()

    if a.device.type != "npu":
        raise ValueError("NPU ceil expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_fceil = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fceil, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fceil, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _ceil_getws_ptr, _ceil_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        dtype_code, dtype_code, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _ceil_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnCeil execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_round(a):
    """Optimized out-of-place round(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_round()

    if a.device.type != "npu":
        raise ValueError("NPU round expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_fround = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fround, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fround, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _round_getws_ptr, _round_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        dtype_code, dtype_code, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _round_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnRound execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)



def fast_trunc(a):
    """Optimized out-of-place trunc(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_trunc()

    if a.device.type != "npu":
        raise ValueError("NPU trunc expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_ftrunc = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_ftrunc, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_ftrunc, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _trunc_getws_ptr, _trunc_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        dtype_code, dtype_code, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _trunc_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnTrunc execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_log2(a):
    """Optimized out-of-place log2(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_log2()

    if a.device.type != "npu":
        raise ValueError("NPU log2 expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_flog2 = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_flog2, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_flog2, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _log2_getws_ptr, _log2_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        dtype_code, dtype_code, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _log2_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnLog2 execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_log10(a):
    """Optimized out-of-place log10(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_log10()

    if a.device.type != "npu":
        raise ValueError("NPU log10 expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_flog10 = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_flog10, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_flog10, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _log10_getws_ptr, _log10_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        dtype_code, dtype_code, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _log10_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnLog10 execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_exp2(a):
    """Optimized out-of-place exp2(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_exp2()

    if a.device.type != "npu":
        raise ValueError("NPU exp2 expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_fexp2 = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fexp2, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fexp2, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _exp2_getws_ptr, _exp2_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        dtype_code, dtype_code, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _exp2_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnExp2 execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_asinh(a):
    """Optimized out-of-place asinh(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_asinh()

    if a.device.type != "npu":
        raise ValueError("NPU asinh expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_fasinh = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fasinh, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fasinh, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _asinh_getws_ptr, _asinh_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        dtype_code, dtype_code, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _asinh_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnAsinh execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_acosh(a):
    """Optimized out-of-place acosh(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_acosh()

    if a.device.type != "npu":
        raise ValueError("NPU acosh expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_facosh = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_facosh, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_facosh, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _acosh_getws_ptr, _acosh_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        dtype_code, dtype_code, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _acosh_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnAcosh execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_atanh(a):
    """Optimized out-of-place atanh(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_atanh()

    if a.device.type != "npu":
        raise ValueError("NPU atanh expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_fatanh = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fatanh, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fatanh, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _atanh_getws_ptr, _atanh_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        dtype_code, dtype_code, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _atanh_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnAtanh execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_atan(a):
    """Optimized out-of-place atan(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_atan()

    if a.device.type != "npu":
        raise ValueError("NPU atan expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_fatan = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fatan, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fatan, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _atan_getws_ptr, _atan_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        dtype_code, dtype_code, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _atan_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnAtan execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_asin(a):
    """Optimized out-of-place asin(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_asin()

    if a.device.type != "npu":
        raise ValueError("NPU asin expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_fasin = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fasin, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fasin, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _asin_getws_ptr, _asin_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        dtype_code, dtype_code, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _asin_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnAsin execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_acos(a):
    """Optimized out-of-place acos(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_acos()

    if a.device.type != "npu":
        raise ValueError("NPU acos expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_facos = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_facos, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_facos, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _acos_getws_ptr, _acos_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        dtype_code, dtype_code, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _acos_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnAcos execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_erfinv(a):
    """Optimized out-of-place erfinv(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_erfinv()

    if a.device.type != "npu":
        raise ValueError("NPU erfinv expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = a.stride
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_fe = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fe, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fe, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _erfinv_getws_ptr, _erfinv_exec_ptr,
        a.shape, a.stride,
        out_shape, out_stride,
        dtype_code, dtype_code, 2,
        a_ptr, o_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _erfinv_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnErfinv execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_erfinv_(a):
    """Optimized in-place erfinv(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_erfinv()

    if a.device.type != "npu":
        raise ValueError("NPU erfinv_ expects NPU tensors")
    cdef int dev_idx = a.device.index or 0
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    cdef int dtype_code = _dtype_to_acl_code(a.dtype)
    cdef uintptr_t a_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.unary_op(
        _erfinv_getws_ptr, _erfinv_exec_ptr,
        a.shape, a.stride,
        a.shape, a.stride,
        dtype_code, dtype_code, 2,
        a_ptr, a_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_ref.execute(
                _erfinv_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnErfinv execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a



def fast_lerp_scalar(a, b, value):
    """Optimized lerp(a, b, scalar) that calls _ffi.two_tensor_scalar_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_lerps()

    cdef int dev_idx
    _validate_npu_binary(a, b, "lerp", &dev_idx)

    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    py_a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    py_b_shape = (<TensorImpl>b)._shape_tuple if isinstance(b, TensorImpl) else b.shape
    cdef int a_ndim = len(py_a_shape)
    cdef int b_ndim = len(py_b_shape)

    if a_ndim > MAX_NDIM or b_ndim > MAX_NDIM:
        raise ValueError(f"ndim exceeds MAX_NDIM ({MAX_NDIM})")

    cdef int64_t[MAX_NDIM] a_shape_buf, b_shape_buf, out_shape_buf, out_stride_buf
    _fill_shape(py_a_shape, a_shape_buf, a_ndim)
    _fill_shape(py_b_shape, b_shape_buf, b_ndim)

    cdef int out_ndim
    cdef int64_t n
    with nogil:
        out_ndim = c_broadcast_shape(
            a_shape_buf, a_ndim, b_shape_buf, b_ndim, out_shape_buf)
        c_contiguous_stride(out_shape_buf, out_ndim, out_stride_buf)
        n = c_numel(out_shape_buf, out_ndim)

    out_shape = _to_tuple(out_shape_buf, out_ndim)
    out_stride = _to_tuple(out_stride_buf, out_ndim)

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_fl = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fl, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fl, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, b_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    if isinstance(b, TensorImpl):
        b_ptr = <uintptr_t>(<TensorImpl>b)._storage._untyped._device_ptr
    else:
        b_ptr = <uintptr_t>b.storage().data_ptr()
    o_ptr = out_ptr

    scalar_handle = _create_scalar_fn(_scalar_bytes_fn(value, a_dtype), dtype_code)
    cdef uintptr_t stream_raw = int(stream.stream)
    try:
        ws_size, executor = _ffi_ref.two_tensor_scalar_op(
            _lerps_getws_ptr, _lerps_exec_ptr,
            py_a_shape, a.stride,
            py_b_shape, b.stride,
            out_shape, out_stride,
            dtype_code, dtype_code, dtype_code, 2,
            a_ptr, b_ptr, o_ptr,
            scalar_handle,
            stream_raw)

        if ws_size:
            workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            try:
                ret = _ffi_ref.execute(
                    _lerps_exec_ptr, int(workspace_ptr), ws_size,
                    executor, stream_raw)
                if ret != 0:
                    raise RuntimeError(f"aclnnLerps execute failed: {ret}")
            finally:
                runtime.defer_raw_free(workspace_ptr)

        _defer_executor_fn(executor)
    finally:
        _destroy_scalar_fn(int(scalar_handle))

    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)



def fast_addcmul(a, b, c, value):
    """Optimized addcmul(a, b, c, value) that calls _ffi.three_tensor_scalar_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_addcmul()

    cdef int dev_idx
    _validate_npu_binary(a, b, "addcmul", &dev_idx)
    if c.device.type != "npu":
        raise ValueError("NPU addcmul expects NPU tensors")
    if c.dtype != a.dtype:
        raise ValueError("NPU addcmul requires matching dtypes")

    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    py_a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    py_b_shape = (<TensorImpl>b)._shape_tuple if isinstance(b, TensorImpl) else b.shape
    py_c_shape = (<TensorImpl>c)._shape_tuple if isinstance(c, TensorImpl) else c.shape
    cdef int a_ndim = len(py_a_shape)
    cdef int b_ndim = len(py_b_shape)
    cdef int c_ndim = len(py_c_shape)

    if a_ndim > MAX_NDIM or b_ndim > MAX_NDIM or c_ndim > MAX_NDIM:
        raise ValueError(f"ndim exceeds MAX_NDIM ({MAX_NDIM})")

    cdef int64_t[MAX_NDIM] a_shape_buf, b_shape_buf, c_shape_buf
    cdef int64_t[MAX_NDIM] tmp_shape_buf, out_shape_buf, out_stride_buf
    _fill_shape(py_a_shape, a_shape_buf, a_ndim)
    _fill_shape(py_b_shape, b_shape_buf, b_ndim)
    _fill_shape(py_c_shape, c_shape_buf, c_ndim)

    cdef int tmp_ndim
    cdef int out_ndim
    cdef int64_t n
    with nogil:
        tmp_ndim = c_broadcast_shape(
            a_shape_buf, a_ndim, b_shape_buf, b_ndim, tmp_shape_buf)
        out_ndim = c_broadcast_shape(
            tmp_shape_buf, tmp_ndim, c_shape_buf, c_ndim, out_shape_buf)
        c_contiguous_stride(out_shape_buf, out_ndim, out_stride_buf)
        n = c_numel(out_shape_buf, out_ndim)

    out_shape = _to_tuple(out_shape_buf, out_ndim)
    out_stride = _to_tuple(out_stride_buf, out_ndim)

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_fa = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fa, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fa, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, b_ptr, c_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    if isinstance(b, TensorImpl):
        b_ptr = <uintptr_t>(<TensorImpl>b)._storage._untyped._device_ptr
    else:
        b_ptr = <uintptr_t>b.storage().data_ptr()
    if isinstance(c, TensorImpl):
        c_ptr = <uintptr_t>(<TensorImpl>c)._storage._untyped._device_ptr
    else:
        c_ptr = <uintptr_t>c.storage().data_ptr()
    o_ptr = out_ptr

    scalar_handle = _create_scalar_fn(_scalar_bytes_fn(value, a_dtype), dtype_code)
    cdef uintptr_t stream_raw = int(stream.stream)
    try:
        ws_size, executor = _ffi_ref.three_tensor_scalar_op(
            _addcmul_getws_ptr, _addcmul_exec_ptr,
            py_a_shape, a.stride,
            py_b_shape, b.stride,
            py_c_shape, c.stride,
            out_shape, out_stride,
            dtype_code, dtype_code, dtype_code, dtype_code, 2,
            a_ptr, b_ptr, c_ptr, o_ptr,
            scalar_handle,
            stream_raw)

        if ws_size:
            workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            try:
                ret = _ffi_ref.execute(
                    _addcmul_exec_ptr, int(workspace_ptr), ws_size,
                    executor, stream_raw)
                if ret != 0:
                    raise RuntimeError(f"aclnnAddcmul execute failed: {ret}")
            finally:
                runtime.defer_raw_free(workspace_ptr)

        _defer_executor_fn(executor)
    finally:
        _destroy_scalar_fn(int(scalar_handle))

    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)



def fast_addcdiv(a, b, c, value):
    """Optimized addcdiv(a, b, c, value) that calls _ffi.three_tensor_scalar_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_addcdiv()

    cdef int dev_idx
    _validate_npu_binary(a, b, "addcdiv", &dev_idx)
    if c.device.type != "npu":
        raise ValueError("NPU addcdiv expects NPU tensors")
    if c.dtype != a.dtype:
        raise ValueError("NPU addcdiv requires matching dtypes")

    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    py_a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    py_b_shape = (<TensorImpl>b)._shape_tuple if isinstance(b, TensorImpl) else b.shape
    py_c_shape = (<TensorImpl>c)._shape_tuple if isinstance(c, TensorImpl) else c.shape
    cdef int a_ndim = len(py_a_shape)
    cdef int b_ndim = len(py_b_shape)
    cdef int c_ndim = len(py_c_shape)

    if a_ndim > MAX_NDIM or b_ndim > MAX_NDIM or c_ndim > MAX_NDIM:
        raise ValueError(f"ndim exceeds MAX_NDIM ({MAX_NDIM})")

    cdef int64_t[MAX_NDIM] a_shape_buf, b_shape_buf, c_shape_buf
    cdef int64_t[MAX_NDIM] tmp_shape_buf, out_shape_buf, out_stride_buf
    _fill_shape(py_a_shape, a_shape_buf, a_ndim)
    _fill_shape(py_b_shape, b_shape_buf, b_ndim)
    _fill_shape(py_c_shape, c_shape_buf, c_ndim)

    cdef int tmp_ndim
    cdef int out_ndim
    cdef int64_t n
    with nogil:
        tmp_ndim = c_broadcast_shape(
            a_shape_buf, a_ndim, b_shape_buf, b_ndim, tmp_shape_buf)
        out_ndim = c_broadcast_shape(
            tmp_shape_buf, tmp_ndim, c_shape_buf, c_ndim, out_shape_buf)
        c_contiguous_stride(out_shape_buf, out_ndim, out_stride_buf)
        n = c_numel(out_shape_buf, out_ndim)

    out_shape = _to_tuple(out_shape_buf, out_ndim)
    out_stride = _to_tuple(out_stride_buf, out_ndim)

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_fa = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fa, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fa, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, b_ptr, c_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    if isinstance(b, TensorImpl):
        b_ptr = <uintptr_t>(<TensorImpl>b)._storage._untyped._device_ptr
    else:
        b_ptr = <uintptr_t>b.storage().data_ptr()
    if isinstance(c, TensorImpl):
        c_ptr = <uintptr_t>(<TensorImpl>c)._storage._untyped._device_ptr
    else:
        c_ptr = <uintptr_t>c.storage().data_ptr()
    o_ptr = out_ptr

    scalar_handle = _create_scalar_fn(_scalar_bytes_fn(value, a_dtype), dtype_code)
    cdef uintptr_t stream_raw = int(stream.stream)
    try:
        ws_size, executor = _ffi_ref.three_tensor_scalar_op(
            _addcdiv_getws_ptr, _addcdiv_exec_ptr,
            py_a_shape, a.stride,
            py_b_shape, b.stride,
            py_c_shape, c.stride,
            out_shape, out_stride,
            dtype_code, dtype_code, dtype_code, dtype_code, 2,
            a_ptr, b_ptr, c_ptr, o_ptr,
            scalar_handle,
            stream_raw)

        if ws_size:
            workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            try:
                ret = _ffi_ref.execute(
                    _addcdiv_exec_ptr, int(workspace_ptr), ws_size,
                    executor, stream_raw)
                if ret != 0:
                    raise RuntimeError(f"aclnnAddcdiv execute failed: {ret}")
            finally:
                runtime.defer_raw_free(workspace_ptr)

        _defer_executor_fn(executor)
    finally:
        _destroy_scalar_fn(int(scalar_handle))

    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)



def cy_npu_synchronize(int device_id=0):
    """Fast NPU synchronize: skip Python imports, Device construction, activate().

    Equivalent to runtime.synchronize() for the given device but avoids:
    - Lazy imports of runtime/allocator/aclnn on each call
    - Device object construction
    - Two activate() round-trips (set_device + set_context)

    All callables are cached at first call via _ensure_npu_imports().
    """
    _ensure_npu_imports()

    # 1. Cached runtime — already initialized, no activate() needed
    runtime = _get_runtime_fast(device_id)

    # 2. Direct aclrtSynchronizeStream (~1.25 us)
    _aclrt_sync_stream_fn(runtime.stream)

    # 3. Allocator drain: pending events + return cached blocks
    alloc = _get_allocator_fn_ref(device_id)
    alloc.synchronize()

    # 4. Flush deferred executors (usually empty, ~0.2 us)
    _flush_executors_fn()

    # 5. Process all three deferred free lists from runtime
    frees = runtime._deferred_frees
    if frees:
        runtime._deferred_frees = []
        for ptr in frees:
            alloc.free(ptr, None)

    raw_frees = runtime._deferred_raw_frees
    if raw_frees:
        runtime._deferred_raw_frees = []
        from candle._backends.npu import runtime as _rt_mod
        for ptr in raw_frees:
            _rt_mod.acl.rt.free(ptr)

    host_frees = runtime._deferred_host_frees
    if host_frees:
        runtime._deferred_host_frees = []
        from candle._backends.npu import runtime as _rt_mod
        for ptr in host_frees:
            _rt_mod.acl.rt.free_host(ptr)
