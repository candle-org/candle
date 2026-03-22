# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython fast-path for _binary_op helper.

Replaces Python-level shape computation (broadcast, stride, numel) with
C-level loops, reducing per-op overhead by ~0.05-0.10ms on the hot path.

The heavy operations (device malloc, aclnn kernel, output wrapping) remain
in Python — this module only accelerates the metadata computation.
"""

from libc.stdint cimport int64_t

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
                raise ValueError("matmul shape mismatch")
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

    # 1. Validate device/dtype (direct attribute access)
    a_dev = a.device
    b_dev = b.device
    if a_dev.type != "npu" or b_dev.type != "npu":
        raise ValueError(f"NPU {name} expects NPU tensors")
    a_dtype = a.dtype
    if a_dtype != b.dtype:
        raise ValueError(f"NPU {name} requires matching dtypes")

    # 2. Get runtime + stream (fast path: skip activate() and TLS lock)
    cdef int dev_idx = a_dev.index or 0
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    # 3. Extract shapes into C arrays
    py_a_shape = a.shape
    py_b_shape = b.shape
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

    # 6. Allocate output
    cdef int isize = c_dtype_itemsize(a_dtype)
    out_ptr = _npu_runtime._alloc_device(n * isize, runtime=runtime)

    # 7. Get data pointers via storage
    a_storage = a.storage()
    b_storage = b.storage()

    # 8. Call aclnn
    fn(
        a_storage.data_ptr(),
        b_storage.data_ptr(),
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
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


# ---------------------------------------------------------------------------
# cy_npu_synchronize — fast synchronize bypassing Python dispatch overhead
# ---------------------------------------------------------------------------

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
