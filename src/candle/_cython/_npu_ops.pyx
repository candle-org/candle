# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython fast-path for _binary_op helper.

Replaces Python-level shape computation (broadcast, stride, numel) with
C-level loops, reducing per-op overhead by ~0.05-0.10ms on the hot path.

The heavy operations (device malloc, aclnn kernel, output wrapping) remain
in Python — this module only accelerates the metadata computation.
"""

from libc.stdint cimport int64_t, int32_t, uint64_t
from libc.stdint cimport uintptr_t

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

    # 6. Allocate output — bypass _alloc_device (avoids 2 lazy imports +
    #    current_stream() Python call per op). stream.stream is the raw ACL
    #    stream pointer (int) that FastNpuAllocator.malloc expects.
    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size, stream=stream.stream)
    else:
        # Multi-device fallback: still avoids the two lazy imports inside
        # _alloc_device by going directly to get_allocator().
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size, stream=stream.stream)

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
# fast_add — hardwired add(a, b, alpha=1) that skips aclnn.py wrapper
# ---------------------------------------------------------------------------

cdef object _ffi_ref = None              # _aclnn_ffi module
cdef object _add_getws_ptr = None        # cached Add getws pointer
cdef object _add_exec_ptr = None         # cached Add exec pointer
cdef object _defer_executor_fn = None    # aclnn._defer_executor
cdef object _acl_rt_malloc_fn = None     # acl.rt.malloc
cdef object _acl_rt_free_fn = None       # acl.rt.free (for workspace)
cdef dict _alpha_one_handles = {}        # dtype_code -> alpha=1 scalar handle (int)
cdef dict _alpha_one_bytes_cache = {}    # dtype_code -> (bytes, alpha_dtype_code) for PTA hash
cdef object _pta_cache_begin_fn = None   # _aclnn_ffi.pta_begin_add_cache_lookup
cdef object _pta_cache_end_fn = None     # _aclnn_ffi.pta_end_cache_lookup


cdef inline void _ensure_ffi_add() except *:
    global _ffi_ref, _add_getws_ptr, _add_exec_ptr
    global _defer_executor_fn, _acl_rt_malloc_fn, _acl_rt_free_fn
    global _pta_cache_begin_fn, _pta_cache_end_fn
    if _ffi_ref is not None:
        return
    from candle._cython import _aclnn_ffi as _f  # pylint: disable=import-error,no-name-in-module
    from candle._backends.npu.aclnn import _defer_executor as _def_ex, ensure_acl as _eacl
    _ffi_ref = _f
    _add_getws_ptr, _add_exec_ptr = _f.resolve_op("Add")
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
    _ensure_ffi_add()

    # 1. Validate device/dtype
    a_dev = a.device
    b_dev = b.device
    if a_dev.type != "npu" or b_dev.type != "npu":
        raise ValueError("fast_add expects NPU tensors")
    a_dtype = a.dtype
    if a_dtype != b.dtype:
        raise ValueError("fast_add requires matching dtypes")

    # 2. Get runtime + stream
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
    cdef str dtype_name = getattr(a_dtype, 'name', str(a_dtype))
    cdef int dtype_code
    if dtype_name == 'float32':
        dtype_code = 0
    elif dtype_name == 'float16':
        dtype_code = 1
    elif dtype_name == 'bfloat16':
        dtype_code = 27
    elif dtype_name == 'int32':
        dtype_code = 3
    elif dtype_name == 'int64':
        dtype_code = 9
    elif dtype_name == 'float64':
        dtype_code = 11
    elif dtype_name == 'int8':
        dtype_code = 2
    elif dtype_name == 'uint8':
        dtype_code = 4
    elif dtype_name == 'int16':
        dtype_code = 6
    elif dtype_name == 'bool':
        dtype_code = 12
    else:
        dtype_code = 0  # fallback to float32

    # 8. Get data pointers — direct C attribute access (no Python method calls)
    cdef uintptr_t a_ptr, b_ptr, o_ptr
    a_ptr = a._storage._untyped._device_ptr
    b_ptr = b._storage._untyped._device_ptr
    o_ptr = out_ptr

    cdef uintptr_t stream_raw = int(stream.stream)
    cdef bint pta_active = False
    cdef uintptr_t alpha_handle

    # 9. Try PTA executor cache (torch_npu-aligned hit_cache_v2 path)
    if _pta_cache_begin_fn is not None:
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
