# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython fast-path for _binary_op helper.

Replaces Python-level shape computation (broadcast, stride, numel) with
C-level loops, reducing per-op overhead by ~0.05-0.10ms on the hot path.

The heavy operations (device malloc, aclnn kernel, output wrapping) remain
in Python — this module only accelerates the metadata computation.
"""

from cpython.list cimport PyList_Append
from libc.stdint cimport int64_t, int32_t, uint64_t
from libc.stdint cimport uintptr_t
from candle._C._tensor_impl cimport (
    TensorImpl,
    cy_make_tensor_from_storage,
    cy_make_tensor_from_storage_trusted,
    cy_make_view_tensor,
)
from candle._C._storage_impl cimport StorageImpl
from candle._C._npu_storage cimport FastNPUStorage, FastTypedStorage
from candle._C._allocator cimport FastNpuAllocator
from candle._C._aclnn_ffi cimport (
    binary_op_no_alpha as _ffi_binary_op_no_alpha,
    binary_op_with_alpha as _ffi_binary_op_with_alpha,
    tensor_scalar_op_with_alpha as _ffi_tensor_scalar_op_with_alpha,
    tensor_scalar_op_no_alpha as _ffi_tensor_scalar_op_no_alpha,
    inplace_tensor_scalar_op_with_alpha as _ffi_inplace_tensor_scalar_op_with_alpha,
    inplace_fill_scalar_op as _ffi_inplace_fill_scalar_op,
    binary_two_inputs_with_int8_op as _ffi_binary_two_inputs_with_int8_op,
    four_tensor_two_scalars_one_int8_op as _ffi_four_tensor_two_scalars_one_int8_op,
    reduce_sum_op as _ffi_reduce_sum_op,
    layer_norm_op as _ffi_layer_norm_op,
    split_with_size_op as _ffi_split_with_size_op,
    execute as _ffi_execute,
    pta_begin_add_cache_lookup as _ffi_pta_begin_add_cache_lookup,
    pta_begin_add_cache_lookup_raw as _ffi_pta_begin_add_cache_lookup_raw,
    pta_begin_addmm_cache_lookup_raw as _ffi_pta_begin_addmm_cache_lookup_raw,
    pta_begin_reduce_sum_cache_lookup_raw as _ffi_pta_begin_reduce_sum_cache_lookup_raw,
    pta_begin_sdpa_flash_cache_lookup_raw as _ffi_pta_begin_sdpa_flash_cache_lookup_raw,
    pta_begin_sdpa_flash_grad_cache_lookup_raw as _ffi_pta_begin_sdpa_flash_grad_cache_lookup_raw,
    pta_begin_sdpa_flash_grad_v2_cache_lookup_raw as _ffi_pta_begin_sdpa_flash_grad_v2_cache_lookup_raw,
    pta_begin_sdpa_flash_grad_storage_cache_lookup_raw as _ffi_pta_begin_sdpa_flash_grad_storage_cache_lookup_raw,
    pta_begin_binary_cache_lookup as _ffi_pta_begin_binary_cache_lookup,
    pta_begin_binary_cache_lookup_raw as _ffi_pta_begin_binary_cache_lookup_raw,
    pta_begin_binary_alpha_cache_lookup_raw as _ffi_pta_begin_binary_alpha_cache_lookup_raw,
    pta_begin_tensor_scalar_cache_lookup_raw as _ffi_pta_begin_tensor_scalar_cache_lookup_raw,
    pta_begin_unary_cache_lookup as _ffi_pta_begin_unary_cache_lookup,
    pta_begin_unary_cache_lookup_raw as _ffi_pta_begin_unary_cache_lookup_raw,
    pta_begin_inplace_copy_cache_lookup as _ffi_pta_begin_inplace_copy_cache_lookup,
    pta_begin_split_with_size_cache_lookup as _ffi_pta_begin_split_with_size_cache_lookup,
    pta_begin_stack_cache_lookup as _ffi_pta_begin_stack_cache_lookup,
    pta_end_cache_lookup as _ffi_pta_end_cache_lookup,
    unary_op as _ffi_unary_op,
    unary_op_with_input_storage as _ffi_unary_op_with_input_storage,
    cast_op as _ffi_cast_op,
    create_tensor_raw as _ffi_create_tensor_raw,
    create_tensor_raw_with_storage as _ffi_create_tensor_raw_with_storage,
    destroy_tensor_raw as _ffi_destroy_tensor_raw,
)
from candle._C._aclrt_ffi cimport memcpy_d2d as _cy_memcpy_d2d
import importlib

DEF MAX_NDIM = 16
DEF MAX_FAST_CAT_INPUTS = 64
DEF SMALL_INNER_CONTIGUOUS_COPY_MAX_BLOCKS = 1024
DEF SMALL_INNER_CONTIGUOUS_COPY_MAX_BYTES = 1048576

ctypedef int32_t (*FlashAttentionScoreGetWorkspaceSize_t)(
    void*, void*, void*, void*, void*, void*, void*, void*,
    double, double, int64_t, int64_t, int64_t, char*, int64_t, int64_t,
    void*, void*, void*, void*, uint64_t*, void**) noexcept nogil

ctypedef int32_t (*FlashAttentionScoreGradGetWorkspaceSize_t)(
    void*, void*, void*, void*, void*, void*, void*, void*,
    void*, void*, void*, void*, void*, double, double, int64_t, int64_t,
    int64_t, char*, int64_t, int64_t, void*, void*, void*, void*,
    uint64_t*, void**) noexcept nogil

ctypedef int32_t (*FlashAttentionScoreGradV2GetWorkspaceSize_t)(
    void*, void*, void*, void*, void*, void*, void*, void*,
    void*, void*, void*, void*, void*, void*, void*, double, double,
    int64_t, int64_t, int64_t, char*, int64_t, int64_t, int64_t,
    void*, void*, void*, void*, uint64_t*, void**) noexcept nogil

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


cdef inline int _tensor_dtype_to_acl_code(TensorImpl t) noexcept:
    """Map TensorImpl's cached dtype code to ACL's dtype enum."""
    if t._dtype_code == 0:      # float32
        return 0
    if t._dtype_code == 1:      # float16
        return 1
    if t._dtype_code == 2:      # float64
        return 11
    if t._dtype_code == 3:      # bfloat16
        return 27
    if t._dtype_code == 4:      # int32
        return 3
    if t._dtype_code == 5:      # int64
        return 9
    if t._dtype_code == 6:      # int16
        return 6
    if t._dtype_code == 7:      # int8
        return 2
    if t._dtype_code == 8:      # uint8
        return 4
    if t._dtype_code == 9:      # bool
        return 12
    return 0


cdef inline int _dtype_to_tensor_code(object dtype):
    """Map dtype object to TensorImpl's cached dtype code."""
    cdef str name = getattr(dtype, 'name', None)
    if name is None:
        name = str(dtype)
    if name == 'float32':
        return 0
    if name == 'float16':
        return 1
    if name == 'float64':
        return 2
    if name == 'bfloat16':
        return 3
    if name == 'int32':
        return 4
    if name == 'int64':
        return 5
    if name == 'int16':
        return 6
    if name == 'int8':
        return 7
    if name == 'uint8':
        return 8
    if name == 'bool':
        return 9
    return -1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

cdef inline void _fill_shape(object py_tuple, int64_t* buf, int ndim):
    """Copy Python tuple of ints into a C array."""
    cdef int i
    for i in range(ndim):
        buf[i] = <int64_t>py_tuple[i]


cdef inline bint _tensor_has_strict_contiguous_stride(TensorImpl t) noexcept:
    """True when t's cached stride is the canonical contiguous stride."""
    cdef int i
    cdef int j
    cdef int64_t acc = 1
    for j in range(t._ndim):
        i = t._ndim - 1 - j
        if t._c_stride[i] != acc:
            return False
        acc = acc * t._c_shape[i]
    return True


cdef inline tuple _to_tuple(const int64_t* arr, int n):
    """Convert C int64 array to Python tuple."""
    return tuple([arr[i] for i in range(n)])


cdef inline tuple _contiguous_stride_tuple(object shape):
    """Compute contiguous strides for a Python shape tuple/list."""
    cdef Py_ssize_t ndim = len(shape)
    cdef list strides = [1] * ndim
    cdef int64_t acc = 1
    cdef Py_ssize_t j
    cdef Py_ssize_t i
    for j in range(ndim):
        i = ndim - 1 - j
        strides[i] = acc
        acc *= <int64_t>shape[i]
    return tuple(strides)


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


cdef inline uintptr_t _npu_data_ptr_fast(object t, int itemsize):
    """Return effective NPU device pointer, including TensorImpl storage offset."""
    cdef FastTypedStorage typed
    if isinstance(t, TensorImpl):
        typed = <FastTypedStorage>(<TensorImpl>t)._storage
        return <uintptr_t>typed._untyped._device_ptr + <uintptr_t>((<TensorImpl>t)._c_offset * itemsize)
    return <uintptr_t>int(t.storage().data_ptr()) + <uintptr_t>(int(t.offset) * itemsize)


cdef inline uintptr_t _npu_storage_base_ptr_fast(object t):
    """Return the base storage pointer without applying TensorImpl offset."""
    cdef FastTypedStorage typed
    if isinstance(t, TensorImpl):
        typed = <FastTypedStorage>(<TensorImpl>t)._storage
        return <uintptr_t>typed._untyped._device_ptr
    return <uintptr_t>int(t.storage().data_ptr())


cdef inline int64_t _npu_storage_numel_fast(object t, int itemsize):
    """Return the backing storage size in elements for ACL storage metadata."""
    cdef FastTypedStorage typed
    if isinstance(t, TensorImpl):
        typed = <FastTypedStorage>(<TensorImpl>t)._storage
        return typed._untyped._nbytes // itemsize
    return int(t.storage().nbytes()) // itemsize


cdef inline bint _can_use_native_offset_unary_descriptor(TensorImpl t) noexcept:
    """True when ACLNN can describe the view by storage offset instead of copying."""
    cdef int i
    if t._ndim == 0 or t._c_offset == 0:
        return False
    for i in range(t._ndim):
        if t._c_stride[i] < 0:
            return False
    return True


cdef inline bint _is_small_inner_contiguous_view(object t) noexcept:
    """True for small positive-stride views whose innermost dimension is contiguous."""
    cdef TensorImpl ti
    cdef int i
    cdef int64_t outer = 1
    cdef int64_t numel = 1
    if not isinstance(t, TensorImpl):
        return False
    ti = <TensorImpl>t
    if ti._ndim == 0 or _tensor_has_strict_contiguous_stride(ti):
        return False
    if ti._c_stride[ti._ndim - 1] != 1:
        return False
    if ti._c_shape[ti._ndim - 1] == 0:
        return False
    for i in range(ti._ndim):
        if ti._c_stride[i] < 0:
            return False
        numel *= ti._c_shape[i]
        if i < ti._ndim - 1:
            outer *= ti._c_shape[i]
    if outer > SMALL_INNER_CONTIGUOUS_COPY_MAX_BLOCKS:
        return False
    if numel * ti._itemsize > SMALL_INNER_CONTIGUOUS_COPY_MAX_BYTES:
        return False
    return True


cdef inline int _copy_small_inner_contiguous_view_to_ptr(
    TensorImpl src,
    uintptr_t dst_base,
    uintptr_t stream_raw,
    int itemsize,
) except -1:
    """Copy a small inner-contiguous view into an already-allocated contiguous buffer."""
    cdef int i, j
    cdef int64_t outer = 1
    cdef int64_t linear_idx, rem, coord, src_offset
    cdef int64_t block_bytes = src._c_shape[src._ndim - 1] * itemsize
    cdef FastTypedStorage typed = <FastTypedStorage>src._storage
    cdef uintptr_t src_base = <uintptr_t>typed._untyped._device_ptr + <uintptr_t>(src._c_offset * itemsize)

    for i in range(src._ndim - 1):
        outer *= src._c_shape[i]

    if src._ndim == 1:
        _cy_memcpy_d2d(dst_base, <uint64_t>block_bytes, src_base, stream_raw, True)
        return 0

    for linear_idx in range(outer):
        rem = linear_idx
        src_offset = 0
        for j in range(src._ndim - 1):
            i = src._ndim - 2 - j
            coord = rem % src._c_shape[i]
            rem = rem // src._c_shape[i]
            src_offset += coord * src._c_stride[i]
        _cy_memcpy_d2d(
            dst_base + <uintptr_t>(linear_idx * block_bytes),
            <uint64_t>block_bytes,
            src_base + <uintptr_t>(src_offset * itemsize),
            stream_raw,
            True,
        )
    return 0


# ---------------------------------------------------------------------------
# fast_binary_op — drop-in replacement for _binary_op in _helpers.py
# ---------------------------------------------------------------------------

# Cached module references/classes (loaded once)
cdef object _npu_runtime = None
cdef object _npu_state = None
cdef object _cy_make_npu_tensor = None
cdef object _FastNPUStorage_cls = None
cdef object _FastTypedStorage_cls = None
cdef object _StrideTuple_cls = None
cdef object _float32_dtype_obj = None

cdef object _get_runtime_fast = None
cdef object _get_stream_fast = None
cdef object _aclrt_sync_stream_fn = None   # _aclrt_ffi.synchronize_stream
cdef object _flush_executors_fn = None     # aclnn.flush_deferred_executors
cdef object _get_allocator_fn_ref = None   # allocator.get_allocator (for sync path)
cdef object _stream_raw_obj_dev0 = None    # cached Python int raw stream for device 0
cdef uintptr_t _stream_raw_dev0 = 0
cdef bint _stream_cache_dev0_valid = False

# Per-device allocator cache: avoids get_allocator() dict lookup on hot path.
# Index 0 covers the overwhelmingly common single-device case.
cdef FastNpuAllocator _fast_allocator_dev0 = None    # FastNpuAllocator for device 0

cdef inline void _ensure_allocator_dev0():
    """Populate _fast_allocator_dev0 on first call (device 0 only)."""
    global _fast_allocator_dev0
    if _fast_allocator_dev0 is not None:
        return
    _ensure_npu_imports()
    _fast_allocator_dev0 = <FastNpuAllocator>_get_allocator_fn_ref(0)


cdef inline object _get_stream_obj_cached_dev0():
    global _stream_raw_obj_dev0, _stream_raw_dev0, _stream_cache_dev0_valid
    cdef object stream
    if _stream_cache_dev0_valid:
        return _stream_raw_obj_dev0
    _ensure_npu_imports()
    stream = _get_stream_fast(0)
    _stream_raw_obj_dev0 = stream.stream
    _stream_raw_dev0 = <uintptr_t>int(_stream_raw_obj_dev0)
    _stream_cache_dev0_valid = True
    return _stream_raw_obj_dev0


cdef inline object _get_stream_obj_fast(int dev_idx):
    if dev_idx == 0:
        return _get_stream_obj_cached_dev0()
    return _get_stream_fast(dev_idx).stream


cdef inline uintptr_t _get_stream_raw_fast(int dev_idx):
    if dev_idx == 0:
        _get_stream_obj_cached_dev0()
        return _stream_raw_dev0
    return <uintptr_t>int(_get_stream_fast(dev_idx).stream)


cpdef void invalidate_stream_cache_dev0():
    global _stream_raw_obj_dev0, _stream_raw_dev0, _stream_cache_dev0_valid
    _stream_raw_obj_dev0 = None
    _stream_raw_dev0 = 0
    _stream_cache_dev0_valid = False


cpdef void invalidate_allocator_cache_dev0():
    """Drop the cached device-0 allocator reference for tests/runtime reset."""
    global _fast_allocator_dev0
    _fast_allocator_dev0 = None


cdef inline void _ensure_npu_imports():
    global _npu_runtime, _npu_state, _cy_make_npu_tensor
    global _FastNPUStorage_cls, _FastTypedStorage_cls, _StrideTuple_cls
    global _get_runtime_fast, _get_stream_fast
    global _aclrt_sync_stream_fn, _flush_executors_fn, _get_allocator_fn_ref
    if _npu_runtime is not None:
        return
    from candle._backends.npu import runtime as rt
    from candle._backends.npu import state as st
    from candle._backends.npu import allocator as _alloc_mod
    from candle._C import _StrideTuple
    from candle._C._storage import cy_make_npu_tensor as _cymt  # pylint: disable=import-error,no-name-in-module
    from candle._C._npu_storage import FastNPUStorage, FastTypedStorage  # pylint: disable=import-error,no-name-in-module
    from candle._C._aclrt_ffi import synchronize_stream as _ssf  # pylint: disable=import-error,no-name-in-module
    from candle._backends.npu.aclnn import flush_deferred_executors as _fef
    _npu_runtime = rt
    _npu_state = st
    _cy_make_npu_tensor = _cymt
    _FastNPUStorage_cls = FastNPUStorage
    _FastTypedStorage_cls = FastTypedStorage
    _StrideTuple_cls = _StrideTuple
    _get_runtime_fast = rt.get_runtime_fast
    _get_stream_fast = st.current_stream_fast
    _aclrt_sync_stream_fn = _ssf
    _flush_executors_fn = _fef
    _get_allocator_fn_ref = _alloc_mod.get_allocator


cdef inline object _get_float32_dtype():
    global _float32_dtype_obj
    if _float32_dtype_obj is None:
        from candle._dtype import float32 as _f32
        _float32_dtype_obj = _f32
    return _float32_dtype_obj


cdef inline object _make_npu_tensor_fast(int64_t device_ptr, int64_t n_elements,
                                         object dtype, object device,
                                         tuple shape, object stride,
                                         int itemsize):
    """Inline NPU Tensor wrapper for eager kernels.

    This mirrors candle._C._storage.cy_make_npu_tensor but uses the runtime truth
    already known by the NPU kernel wrapper, avoiding storage->device/dtype
    round-trips inside TensorImpl initialization.
    """
    cdef int64_t nbytes = n_elements * itemsize
    cdef FastNPUStorage untyped = FastNPUStorage(device_ptr, nbytes, device)
    cdef FastTypedStorage typed = FastTypedStorage(untyped, dtype, n_elements)
    cdef object idx = getattr(device, "index", None)
    cdef int device_index = <int>(idx if idx is not None else -1)
    return cy_make_tensor_from_storage_trusted(
        typed,
        shape,
        stride,
        0,
        device,
        1,
        device_index,
        dtype,
        _dtype_to_tensor_code(dtype),
        itemsize,
    )



cdef inline object _make_npu_tensor_fast_large(int64_t device_ptr, int64_t n_elements,
                                               object dtype, object device,
                                               tuple shape, object stride,
                                               int itemsize, int device_index,
                                               int tensor_dtype_code):
    """Inline NPU Tensor wrapper whose storage can use large-pool fast free."""
    cdef int64_t nbytes = n_elements * itemsize
    cdef FastNPUStorage untyped = FastNPUStorage(device_ptr, nbytes, device, True)
    cdef FastTypedStorage typed = FastTypedStorage(untyped, dtype, n_elements)
    return cy_make_tensor_from_storage_trusted(
        typed,
        shape,
        stride,
        0,
        device,
        1,
        device_index,
        dtype,
        tensor_dtype_code,
        itemsize,
    )


cpdef object fast_last_dim_slice_view(object tensor, object key):
    """Fast path for Ellipsis + step-1 last-dim slices, e.g. x[..., :h]."""
    _ensure_npu_imports()
    if not isinstance(tensor, TensorImpl):
        return None
    cdef TensorImpl base = <TensorImpl>tensor
    if base._ndim == 0:
        return None
    if not isinstance(key, tuple):
        return None
    cdef int key_len = len(key)
    if key_len == 0 or key_len > 2:
        return None

    cdef object last
    if key_len == 1:
        last = key[0]
    else:
        if key[0] is not Ellipsis:
            return None
        last = key[1]
    if not isinstance(last, slice):
        return None

    cdef Py_ssize_t start_obj, stop_obj, step_obj
    start_obj, stop_obj, step_obj = last.indices(base._c_shape[base._ndim - 1])
    cdef int64_t start = <int64_t>start_obj
    cdef int64_t stop = <int64_t>stop_obj
    cdef int64_t step = <int64_t>step_obj
    if step != 1:
        return None

    cdef int64_t length = stop - start
    if length < 0:
        length = 0
    cdef int i
    cdef int64_t offset = base._c_offset + start * base._c_stride[base._ndim - 1]
    cdef list shape_list = [0] * base._ndim
    cdef list stride_list = [0] * base._ndim
    for i in range(base._ndim):
        if i == base._ndim - 1:
            shape_list[i] = length
        else:
            shape_list[i] = base._c_shape[i]
        stride_list[i] = base._c_stride[i]
    return cy_make_view_tensor(base, base._storage, tuple(shape_list), tuple(stride_list), offset)


cpdef object fast_copy_small_inner_contiguous_view(object view):
    """Materialize a small inner-contiguous NPU view using Cython D2D copies."""
    _ensure_npu_imports()

    cdef tuple shape = (<TensorImpl>view)._shape_tuple if isinstance(view, TensorImpl) else tuple(view.shape)
    cdef object stride_obj = (<TensorImpl>view)._stride_tuple if isinstance(view, TensorImpl) else view.stride
    cdef int ndim = len(shape)
    if ndim == 0:
        return None
    if <int64_t>stride_obj[ndim - 1] != 1:
        return None

    cdef int i, j
    for i in range(ndim):
        if <int64_t>stride_obj[i] < 0:
            return None

    cdef int64_t inner = <int64_t>shape[ndim - 1]
    if inner == 0:
        return None

    cdef int64_t outer = 1
    for i in range(ndim - 1):
        outer *= <int64_t>shape[i]
    if outer > SMALL_INNER_CONTIGUOUS_COPY_MAX_BLOCKS:
        return None

    cdef int dev_idx = (<TensorImpl>view)._device_index if isinstance(view, TensorImpl) else (view.device.index or 0)
    cdef object dtype = (<TensorImpl>view)._dtype_obj if isinstance(view, TensorImpl) else view.dtype
    cdef object device = _device_obj_fast(view)
    cdef int itemsize = (<TensorImpl>view)._itemsize if isinstance(view, TensorImpl) else c_dtype_itemsize(dtype)
    cdef int64_t out_numel = 1
    for i in range(ndim):
        out_numel *= <int64_t>shape[i]
    cdef int64_t total_bytes = out_numel * itemsize
    if total_bytes > SMALL_INNER_CONTIGUOUS_COPY_MAX_BYTES:
        return None

    cdef object out_stride = _contiguous_stride_tuple(shape)
    cdef object stream_obj = _get_stream_obj_fast(dev_idx)
    cdef uintptr_t stream_raw = _get_stream_raw_fast(dev_idx)
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(max(total_bytes, itemsize), stream=stream_obj)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(max(total_bytes, itemsize), stream=stream_obj)

    cdef uintptr_t dst_base = <uintptr_t>out_ptr
    cdef uintptr_t src_base = _npu_data_ptr_fast(view, itemsize)
    cdef int64_t block_bytes = inner * itemsize
    cdef int64_t linear_idx, rem, coord, src_offset

    if ndim == 1:
        _cy_memcpy_d2d(dst_base, <uint64_t>block_bytes, src_base, stream_raw, True)
    else:
        for linear_idx in range(outer):
            rem = linear_idx
            src_offset = 0
            for j in range(ndim - 1):
                i = ndim - 2 - j
                coord = rem % <int64_t>shape[i]
                rem = rem // <int64_t>shape[i]
                src_offset += coord * <int64_t>stride_obj[i]
            _cy_memcpy_d2d(
                dst_base + <uintptr_t>(linear_idx * block_bytes),
                <uint64_t>block_bytes,
                src_base + <uintptr_t>(src_offset * itemsize),
                stream_raw,
                True,
            )

    cdef int dtype_code = (<TensorImpl>view)._dtype_code if isinstance(view, TensorImpl) else _dtype_to_tensor_code(dtype)
    return _make_npu_tensor_fast_large(<int64_t>dst_base, out_numel, dtype, device, shape, out_stride, itemsize, dev_idx, dtype_code)


cpdef object fast_cat_small_last_dim(object tensors, int64_t dim, object out_shape,
                                     object out_stride, int64_t out_ptr, object first):
    """Concatenate small inner-contiguous tensors with Cython D2D row copies."""
    _ensure_npu_imports()

    cdef int ndim = len(out_shape)
    cdef int ntensors = len(tensors)
    if ndim == 0 or dim != ndim - 1 or ntensors <= 0 or ntensors > MAX_FAST_CAT_INPUTS:
        return None

    cdef int dev_idx = (<TensorImpl>first)._device_index if isinstance(first, TensorImpl) else (first.device.index or 0)
    cdef object dtype = (<TensorImpl>first)._dtype_obj if isinstance(first, TensorImpl) else first.dtype
    cdef object device = _device_obj_fast(first)
    cdef int itemsize = (<TensorImpl>first)._itemsize if isinstance(first, TensorImpl) else c_dtype_itemsize(dtype)
    cdef int dtype_code = (<TensorImpl>first)._dtype_code if isinstance(first, TensorImpl) else _dtype_to_tensor_code(dtype)
    cdef int64_t out_numel = 1
    cdef int i, j, tensor_idx
    for i in range(ndim):
        out_numel *= <int64_t>out_shape[i]
    cdef int64_t total_bytes = out_numel * itemsize
    if total_bytes > SMALL_INNER_CONTIGUOUS_COPY_MAX_BYTES:
        return None

    cdef int64_t outer = 1
    for i in range(ndim - 1):
        outer *= <int64_t>out_shape[i]
    if outer > SMALL_INNER_CONTIGUOUS_COPY_MAX_BLOCKS:
        return None

    cdef uintptr_t src_bases[MAX_FAST_CAT_INPUTS]
    cdef int64_t src_strides[MAX_FAST_CAT_INPUTS][MAX_NDIM]
    cdef int64_t inners[MAX_FAST_CAT_INPUTS]
    cdef int64_t row_offsets[MAX_FAST_CAT_INPUTS]
    cdef object t, stride_obj, shape_obj
    cdef int64_t row_offset = 0

    for tensor_idx in range(ntensors):
        t = tensors[tensor_idx]
        if t.dtype != dtype or t.device != device:
            return None
        shape_obj = (<TensorImpl>t)._shape_tuple if isinstance(t, TensorImpl) else t.shape
        stride_obj = (<TensorImpl>t)._stride_tuple if isinstance(t, TensorImpl) else t.stride
        if len(shape_obj) != ndim or <int64_t>stride_obj[ndim - 1] != 1:
            return None
        for i in range(ndim):
            if <int64_t>stride_obj[i] < 0:
                return None
            if i != ndim - 1 and <int64_t>shape_obj[i] != <int64_t>out_shape[i]:
                return None
            src_strides[tensor_idx][i] = <int64_t>stride_obj[i]
        inners[tensor_idx] = <int64_t>shape_obj[ndim - 1]
        row_offsets[tensor_idx] = row_offset
        row_offset += inners[tensor_idx]
        src_bases[tensor_idx] = _npu_data_ptr_fast(t, itemsize)
    if row_offset != <int64_t>out_shape[ndim - 1]:
        return None

    cdef uintptr_t dst_base = <uintptr_t>out_ptr
    cdef uintptr_t stream_raw = _get_stream_raw_fast(dev_idx)
    cdef int64_t linear_idx, rem, coord, src_offset, inner
    for linear_idx in range(outer):
        for tensor_idx in range(ntensors):
            inner = inners[tensor_idx]
            if inner == 0:
                continue
            rem = linear_idx
            src_offset = 0
            for j in range(ndim - 1):
                i = ndim - 2 - j
                coord = rem % <int64_t>out_shape[i]
                rem = rem // <int64_t>out_shape[i]
                src_offset += coord * src_strides[tensor_idx][i]
            _cy_memcpy_d2d(
                dst_base + <uintptr_t>((linear_idx * <int64_t>out_shape[ndim - 1] + row_offsets[tensor_idx]) * itemsize),
                <uint64_t>(inner * itemsize),
                src_bases[tensor_idx] + <uintptr_t>(src_offset * itemsize),
                stream_raw,
                True,
            )

    return _make_npu_tensor_fast_large(out_ptr, max(out_numel, 1), dtype, device, tuple(out_shape), out_stride, itemsize, dev_idx, dtype_code)


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
    if name in ("atan2", "logaddexp", "logaddexp2", "remainder", "fmod", "pow", "floor_divide", "eq", "ne", "lt", "le", "gt", "ge", "logical_and", "logical_or", "logical_xor", "bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_left_shift", "bitwise_right_shift", "max", "maximum", "min", "minimum"):
        from candle._C import _aclnn_ffi as _ffi  # pylint: disable=import-error,no-name-in-module
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
        elif name == "bitwise_left_shift":
            pretty = "aclnnLeftShift"
            getws_ptr, exec_ptr = _ffi.resolve_op("LeftShift")
            ws_size, executor = _ffi.binary_two_inputs_op(
                getws_ptr, exec_ptr,
                py_a_shape, a.stride,
                py_b_shape, b.stride,
                out_shape, out_stride,
                dtype_code, dtype_code, dtype_code,
                2,
                int(a_ptr), int(b_ptr), int(out_ptr),
                int(stream.stream))
        elif name == "bitwise_right_shift":
            pretty = "aclnnRightShift"
            getws_ptr, exec_ptr = _ffi.resolve_op("RightShift")
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
cdef object _adds_getws_ptr = None       # cached Adds getws pointer
cdef object _adds_exec_ptr = None        # cached Adds exec pointer
cdef object _inplace_adds_getws_ptr = None  # cached InplaceAdds getws pointer
cdef object _inplace_adds_exec_ptr = None   # cached InplaceAdds exec pointer
cdef object _mul_getws_ptr = None        # cached Mul getws pointer
cdef object _mul_exec_ptr = None         # cached Mul exec pointer
cdef object _muls_getws_ptr = None       # cached Muls getws pointer
cdef object _muls_exec_ptr = None        # cached Muls exec pointer
cdef object _sub_getws_ptr = None        # cached Sub getws pointer
cdef object _sub_exec_ptr = None         # cached Sub exec pointer
cdef object _subs_getws_ptr = None       # cached Subs getws pointer
cdef object _subs_exec_ptr = None        # cached Subs exec pointer
cdef object _div_getws_ptr = None        # cached Div getws pointer
cdef object _div_exec_ptr = None         # cached Div exec pointer
cdef object _divs_getws_ptr = None       # cached Divs getws pointer
cdef object _divs_exec_ptr = None        # cached Divs exec pointer
cdef object _fill_scalar_getws_ptr = None  # cached InplaceFillScalar getws pointer
cdef object _fill_scalar_exec_ptr = None   # cached InplaceFillScalar exec pointer
cdef object _matmul_getws_ptr = None     # cached Matmul getws pointer
cdef object _matmul_exec_ptr = None      # cached Matmul exec pointer
cdef object _sdpa_flash_getws_ptr = None  # cached FlashAttentionScore getws pointer
cdef object _sdpa_flash_exec_ptr = None   # cached FlashAttentionScore exec pointer
cdef object _sdpa_flash_grad_getws_ptr = None  # cached FlashAttentionScoreGrad getws pointer
cdef object _sdpa_flash_grad_exec_ptr = None   # cached FlashAttentionScoreGrad exec pointer
cdef object _sdpa_flash_grad_v2_getws_ptr = None  # cached FlashAttentionScoreGradV2 getws pointer
cdef object _sdpa_flash_grad_v2_exec_ptr = None   # cached FlashAttentionScoreGradV2 exec pointer
cdef object _addmm_getws_ptr = None      # cached Addmm getws pointer
cdef object _addmm_exec_ptr = None       # cached Addmm exec pointer
cdef object _reduce_sum_getws_ptr = None  # cached ReduceSum getws pointer
cdef object _reduce_sum_exec_ptr = None   # cached ReduceSum exec pointer
cdef object _layer_norm_getws_ptr = None  # cached LayerNorm getws pointer
cdef object _layer_norm_exec_ptr = None   # cached LayerNorm exec pointer
cdef object _layer_norm_backward_getws_ptr = None  # cached LayerNormBackward getws pointer
cdef object _layer_norm_backward_exec_ptr = None   # cached LayerNormBackward exec pointer
cdef object _bitwise_and_getws_ptr = None  # cached BitwiseAndTensor getws pointer
cdef object _bitwise_and_exec_ptr = None   # cached BitwiseAndTensor exec pointer
cdef object _bitwise_or_getws_ptr = None   # cached BitwiseOrTensor getws pointer
cdef object _bitwise_or_exec_ptr = None    # cached BitwiseOrTensor exec pointer
cdef object _bitwise_xor_getws_ptr = None  # cached BitwiseXorTensor getws pointer
cdef object _bitwise_xor_exec_ptr = None   # cached BitwiseXorTensor exec pointer
cdef object _pow_getws_ptr = None          # cached PowTensorTensor getws pointer
cdef object _pow_exec_ptr = None           # cached PowTensorTensor exec pointer
cdef object _cast_getws_ptr = None         # cached Cast getws pointer
cdef object _cast_exec_ptr = None          # cached Cast exec pointer
cdef object _defer_executor_fn = None    # aclnn._defer_executor
cdef object _deferred_executors_ref = None  # aclnn._DEFERRED_EXECUTORS
cdef object _acl_rt_malloc_fn = None     # acl.rt.malloc
cdef object _acl_rt_free_fn = None       # acl.rt.free (for workspace)
cdef dict _alpha_one_handles = {}        # dtype_code -> alpha=1 scalar handle (int)
cdef dict _alpha_one_bytes_cache = {}    # dtype_code -> (bytes, alpha_dtype_code) for PTA hash
cdef dict _scalar_handle_cache = {}      # (dtype_code, scalar value) -> scalar handle (int)
cdef dict _scalar_tensor_cache = {}      # (device, dtype, scalar value) -> NPU scalar tensor
cdef object _capture_check_fn = None      # candle.npu._is_in_graph_capture
cdef object _pta_cache_begin_fn = None   # _aclnn_ffi.pta_begin_add_cache_lookup
cdef object _pta_binary_begin_fn = None  # _aclnn_ffi.pta_begin_binary_cache_lookup
cdef object _pta_unary_begin_fn = None   # _aclnn_ffi.pta_begin_unary_cache_lookup
cdef object _pta_cache_end_fn = None     # _aclnn_ffi.pta_end_cache_lookup
# PTA cached-executor path follows torch_npu's hit_cache_v2 flow.  CANN usually
# rebinds current tensor addresses during lookup via AddTensorAddrToCachedList;
# however, Mul has shown stale binding when a cache entry created with aliased
# inputs (e.g. square via mul(a, a)) is later reused for non-aliased inputs of
# the same metadata signature.  The binary PTA hash includes an input-alias
# discriminator so ordinary same-shape tensor Mul still reuses cached executors
# while square-style and non-aliased Mul entries stay separate.
cdef bint _use_add_pta_cache = True
cdef bint _use_sub_pta_cache = True
cdef bint _use_div_pta_cache = True
cdef bint _use_mul_pta_cache = True
cdef bint _use_matmul_pta_cache = True
cdef bint _use_addmm_pta_cache = True
cdef bint _use_reduce_sum_pta_cache = True
cdef bint _use_sdpa_flash_pta_cache = True
cdef bint _use_sdpa_flash_grad_pta_cache = True
# TODO: re-enable when CANN FlashAttentionScoreGradV2 avoids its ~45MB workspace
# and multi-second latency on the default BNSD training shape.
cdef bint _use_sdpa_flash_grad_v2 = False
cdef bint _use_silu_pta_cache = True
cdef bint _use_gelu_pta_cache = True
cdef bint _use_cast_pta_cache = True
cdef bint _use_silu_backward_pta_cache = True
cdef bint _use_gelu_backward_pta_cache = True
cdef dict _mul_pta_pointer_keys = {}
cdef dict _div_pta_pointer_keys = {}
cdef dict _silu_backward_pta_pointer_keys = {}


cdef inline void _defer_executor_handle(uintptr_t executor) except *:
    if executor == 0:
        return
    if PyList_Append(_deferred_executors_ref, executor) != 0:
        raise MemoryError("failed to defer ACLNN executor handle")


cdef inline void _ensure_ffi_binary() except *:
    global _ffi_ref, _add_getws_ptr, _add_exec_ptr
    global _adds_getws_ptr, _adds_exec_ptr
    global _inplace_adds_getws_ptr, _inplace_adds_exec_ptr
    global _mul_getws_ptr, _mul_exec_ptr
    global _muls_getws_ptr, _muls_exec_ptr
    global _sub_getws_ptr, _sub_exec_ptr
    global _subs_getws_ptr, _subs_exec_ptr
    global _div_getws_ptr, _div_exec_ptr
    global _divs_getws_ptr, _divs_exec_ptr
    global _fill_scalar_getws_ptr, _fill_scalar_exec_ptr
    global _matmul_getws_ptr, _matmul_exec_ptr
    global _defer_executor_fn, _deferred_executors_ref
    global _acl_rt_malloc_fn, _acl_rt_free_fn
    global _pta_cache_begin_fn, _pta_binary_begin_fn, _pta_unary_begin_fn, _pta_cache_end_fn
    if _ffi_ref is not None:
        return
    from candle._C import _aclnn_ffi as _f  # pylint: disable=import-error,no-name-in-module
    from candle._backends.npu import aclnn as _aclnn_mod
    _ffi_ref = _f
    _add_getws_ptr, _add_exec_ptr = _f.resolve_op("Add")
    _adds_getws_ptr, _adds_exec_ptr = _f.resolve_op("Adds")
    _inplace_adds_getws_ptr, _inplace_adds_exec_ptr = _f.resolve_op("InplaceAdds")
    _mul_getws_ptr, _mul_exec_ptr = _f.resolve_op("Mul")
    _muls_getws_ptr, _muls_exec_ptr = _f.resolve_op("Muls")
    _sub_getws_ptr, _sub_exec_ptr = _f.resolve_op("Sub")
    _subs_getws_ptr, _subs_exec_ptr = _f.resolve_op("Subs")
    _div_getws_ptr, _div_exec_ptr = _f.resolve_op("Div")
    _divs_getws_ptr, _divs_exec_ptr = _f.resolve_op("Divs")
    _fill_scalar_getws_ptr, _fill_scalar_exec_ptr = _f.resolve_op("InplaceFillScalar")
    _matmul_getws_ptr, _matmul_exec_ptr = _f.resolve_op("Matmul")
    _defer_executor_fn = _aclnn_mod._defer_executor
    _deferred_executors_ref = _aclnn_mod._DEFERRED_EXECUTORS
    # Direct-append fast path bypasses _defer_executor's Python normalization, so
    # register the atexit drain that _defer_executor would otherwise install.
    _aclnn_mod._register_cleanup()
    _acl = _aclnn_mod.ensure_acl()
    _acl_rt_malloc_fn = _acl.rt.malloc
    _acl_rt_free_fn = _acl.rt.free
    if _f.is_pta_cache_available():
        _pta_cache_begin_fn = _f.pta_begin_add_cache_lookup
        _pta_binary_begin_fn = _f.pta_begin_binary_cache_lookup
        _pta_unary_begin_fn = _f.pta_begin_unary_cache_lookup
        _pta_cache_end_fn = _f.pta_end_cache_lookup


cdef inline void _ensure_ffi_sdpa_flash() except *:
    global _sdpa_flash_getws_ptr, _sdpa_flash_exec_ptr
    global _sdpa_flash_grad_getws_ptr, _sdpa_flash_grad_exec_ptr
    global _sdpa_flash_grad_v2_getws_ptr, _sdpa_flash_grad_v2_exec_ptr
    _ensure_ffi_binary()
    if _sdpa_flash_getws_ptr is not None:
        return
    _sdpa_flash_getws_ptr, _sdpa_flash_exec_ptr = _ffi_ref.resolve_op("FlashAttentionScore")
    _sdpa_flash_grad_getws_ptr, _sdpa_flash_grad_exec_ptr = _ffi_ref.resolve_op("FlashAttentionScoreGrad")
    grad_v2 = _ffi_ref.resolve_op_optional("FlashAttentionScoreGradV2")
    if grad_v2 is not None:
        _sdpa_flash_grad_v2_getws_ptr, _sdpa_flash_grad_v2_exec_ptr = grad_v2


cdef inline void _ensure_ffi_layer_norm() except *:
    global _layer_norm_getws_ptr, _layer_norm_exec_ptr
    global _layer_norm_backward_getws_ptr, _layer_norm_backward_exec_ptr
    _ensure_ffi_binary()
    if _layer_norm_getws_ptr is not None:
        return
    _layer_norm_getws_ptr, _layer_norm_exec_ptr = _ffi_ref.resolve_op("LayerNorm")
    _layer_norm_backward_getws_ptr, _layer_norm_backward_exec_ptr = _ffi_ref.resolve_op("LayerNormBackward")


cdef inline void _ensure_ffi_bitwise() except *:
    """Resolve and cache BitwiseAnd/Or/Xor getws/exec ptrs (lazy)."""
    global _bitwise_and_getws_ptr, _bitwise_and_exec_ptr
    global _bitwise_or_getws_ptr, _bitwise_or_exec_ptr
    global _bitwise_xor_getws_ptr, _bitwise_xor_exec_ptr
    if _bitwise_and_getws_ptr is not None:
        return
    _ensure_ffi_binary()
    _bitwise_and_getws_ptr, _bitwise_and_exec_ptr = _ffi_ref.resolve_op("BitwiseAndTensor")
    _bitwise_or_getws_ptr, _bitwise_or_exec_ptr = _ffi_ref.resolve_op("BitwiseOrTensor")
    _bitwise_xor_getws_ptr, _bitwise_xor_exec_ptr = _ffi_ref.resolve_op("BitwiseXorTensor")


cdef inline void _ensure_ffi_pow() except *:
    """Resolve and cache PowTensorTensor getws/exec ptrs (lazy)."""
    global _pow_getws_ptr, _pow_exec_ptr
    if _pow_getws_ptr is not None:
        return
    _ensure_ffi_binary()
    _pow_getws_ptr, _pow_exec_ptr = _ffi_ref.resolve_op("PowTensorTensor")


cdef inline void _ensure_ffi_cast() except *:
    """Resolve and cache Cast getws/exec ptrs (lazy)."""
    global _cast_getws_ptr, _cast_exec_ptr
    if _cast_getws_ptr is not None:
        return
    _ensure_ffi_binary()
    _cast_getws_ptr, _cast_exec_ptr = _ffi_ref.resolve_op("Cast")


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


cdef object _scalar_bytes_for_dtype_code(int dtype_code, object value):
    """Encode a Python scalar for aclCreateScalar without Python backend wrappers."""
    import struct
    cdef int bits
    cdef int lsb
    cdef int rounded
    if dtype_code == 0:    # float32
        return struct.pack('<f', float(value))
    if dtype_code == 1:    # float16
        return struct.pack('<e', float(value))
    if dtype_code == 27:   # bfloat16
        bits = struct.unpack('<I', struct.pack('<f', float(value)))[0]
        lsb = (bits >> 16) & 1
        rounded = bits + 0x7FFF + lsb
        return int((rounded >> 16) & 0xFFFF).to_bytes(2, byteorder='little', signed=False)
    if dtype_code == 3:    # int32
        return int(value).to_bytes(4, byteorder='little', signed=True)
    if dtype_code == 9:    # int64
        return int(value).to_bytes(8, byteorder='little', signed=True)
    if dtype_code == 11:   # float64
        return struct.pack('<d', float(value))
    if dtype_code == 2:    # int8
        return int(value).to_bytes(1, byteorder='little', signed=True)
    if dtype_code == 4:    # uint8
        return int(value).to_bytes(1, byteorder='little', signed=False)
    if dtype_code == 6:    # int16
        return int(value).to_bytes(2, byteorder='little', signed=True)
    if dtype_code == 12:   # bool
        return (1 if bool(value) else 0).to_bytes(1, byteorder='little', signed=False)
    return struct.pack('<f', float(value))


cdef uintptr_t _get_cached_scalar_handle(int dtype_code, object value) except? 0:
    global _scalar_handle_cache
    cdef object normalized
    cdef object key
    cdef object existing
    cdef object scalar_bytes
    cdef uintptr_t handle
    if dtype_code in (0, 1, 11, 27):
        normalized = float(value)
    elif dtype_code == 12:
        normalized = bool(value)
    else:
        normalized = int(value)
    key = (dtype_code, normalized)
    existing = _scalar_handle_cache.get(key)
    if existing is not None:
        return <uintptr_t>existing
    scalar_bytes = _scalar_bytes_for_dtype_code(dtype_code, normalized)
    handle = <uintptr_t>_ffi_ref.create_scalar(scalar_bytes, dtype_code)
    _scalar_handle_cache[key] = handle
    return handle


cdef inline object _normalize_scalar_cache_value(int dtype_code, object value):
    if dtype_code in (0, 1, 11, 27):
        return float(value)
    if dtype_code == 12:
        return bool(value)
    return int(value)


cdef inline bint _npu_in_graph_capture():
    global _capture_check_fn
    if _capture_check_fn is None:
        try:
            from candle import npu as _npu_mod
            _capture_check_fn = _npu_mod._is_in_graph_capture
        except Exception:
            _capture_check_fn = False
    if _capture_check_fn is False:
        return False
    return bool(_capture_check_fn())


cdef object _get_cached_scalar_tensor(TensorImpl a, object value):
    """Return a persistent 0-d NPU tensor for scalar eager arithmetic.

    Direct ACLNN scalar kernels measure materially slower than tensor-tensor PTA
    for Add/Mul/Sub/Div on this CANN build.  Outside graph capture, materialize
    each (device, dtype, value) once with on-device InplaceFillScalar, then route
    arithmetic through the normal tensor-tensor exact kernels.  During capture,
    keep using direct scalar ACLNN so no new scalar tensor/H2D work is recorded.
    """
    global _scalar_tensor_cache
    _ensure_npu_imports()
    _ensure_ffi_binary()

    cdef int dtype_code = _tensor_dtype_to_acl_code(a)
    cdef object normalized = _normalize_scalar_cache_value(dtype_code, value)
    cdef int dev_idx = a._device_index
    if dev_idx < 0:
        dev_idx = 0
    cdef object key = (dev_idx, dtype_code, normalized)
    cdef object cached = _scalar_tensor_cache.get(key)
    if cached is not None:
        return cached

    cdef object stream_obj = _get_stream_obj_fast(dev_idx)
    cdef object out_ptr
    cdef int isize = a._itemsize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc_large_cached(isize, stream_obj)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(isize, stream=stream_obj)

    cdef object shape = ()
    cdef object stride = ()
    cdef uintptr_t scalar_handle = _get_cached_scalar_handle(dtype_code, normalized)
    cdef uintptr_t stream_raw = _get_stream_raw_fast(dev_idx)
    cdef object ws_size
    cdef object executor
    cdef object workspace_ptr
    cdef object ret
    cdef int ret_i
    try:
        ws_size, executor = _ffi_inplace_fill_scalar_op(
            <uintptr_t>_fill_scalar_getws_ptr,
            <uintptr_t>_fill_scalar_exec_ptr,
            shape,
            stride,
            dtype_code,
            2,
            <uintptr_t>out_ptr,
            scalar_handle,
            stream_raw,
        )
        if ws_size:
            workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            try:
                ret_i = _ffi_execute(<uintptr_t>_fill_scalar_exec_ptr, <uintptr_t>int(workspace_ptr), ws_size, executor, stream_raw)
                if ret_i != 0:
                    raise RuntimeError(f"aclnnInplaceFillScalar execute failed: {ret_i}")
            finally:
                _get_runtime_fast(dev_idx).defer_raw_free(workspace_ptr)
        _defer_executor_handle(<uintptr_t>int(executor))
    except Exception:
        _get_allocator_fn_ref(dev_idx).free(<int64_t>out_ptr)
        raise

    cached = _make_npu_tensor_fast_large(<int64_t>out_ptr, 1, a._dtype_obj, a._device_obj,
                                         shape, stride, isize, dev_idx, a._dtype_code)
    _scalar_tensor_cache[key] = cached
    return cached


cdef object _fast_tensor_scalar_exact(TensorImpl a, object value, object getws_ptr, object exec_ptr,
                                      bint with_alpha, str pretty_name):
    """Run a covered NPU tensor/scalar op entirely through Cython _C."""
    _ensure_npu_imports()
    _ensure_ffi_binary()

    cdef int dev_idx = a._device_index
    if dev_idx < 0:
        dev_idx = 0
    cdef object a_dev = a._device_obj
    cdef object a_dtype = a._dtype_obj
    cdef object py_shape = a._shape_tuple
    cdef object py_stride = a._stride_tuple
    cdef int ndim = a._ndim
    if ndim > MAX_NDIM:
        raise ValueError(f"ndim exceeds MAX_NDIM ({MAX_NDIM})")

    cdef int64_t[MAX_NDIM] shape_buf, out_stride_buf
    _fill_shape(py_shape, shape_buf, ndim)
    with nogil:
        c_contiguous_stride(shape_buf, ndim, out_stride_buf)
    cdef object out_stride = _to_tuple(out_stride_buf, ndim)
    cdef int64_t n = a._c_numel
    cdef int64_t alloc_numel = n if n > 0 else 1
    cdef int isize = a._itemsize
    cdef int64_t alloc_size = alloc_numel * isize
    cdef object stream_obj = _get_stream_obj_fast(dev_idx)
    cdef object out_ptr
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc_large_cached(alloc_size, stream_obj)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size, stream=stream_obj)

    cdef int dtype_code = _tensor_dtype_to_acl_code(a)
    cdef uintptr_t a_ptr = a._storage._untyped._device_ptr + <uintptr_t>(a._c_offset * a._itemsize)
    cdef uintptr_t o_ptr = <uintptr_t>out_ptr
    cdef uintptr_t stream_raw = _get_stream_raw_fast(dev_idx)
    cdef uintptr_t scalar_handle = _get_cached_scalar_handle(dtype_code, value)
    cdef uintptr_t alpha_handle
    cdef object ws_size
    cdef object executor
    cdef object workspace_ptr
    cdef object ret

    # CANN 9.0 reports scalar ops as PTA-cacheable, but measured hit-cache lookups
    # for Adds/Muls/Subs/Divs are slower (~100us) than the direct ACLNN scalar
    # GetWorkspaceSize path (~40us) on this platform. Keep scalar ops on the
    # direct Cython ACLNN path until the native PTA scalar path is proven faster.
    try:
        if with_alpha:
            alpha_handle = _get_alpha_one(dtype_code)
            ws_size, executor = _ffi_tensor_scalar_op_with_alpha(
                <uintptr_t>getws_ptr, <uintptr_t>exec_ptr,
                py_shape, py_stride,
                py_shape, out_stride,
                dtype_code, 2,
                a_ptr, o_ptr,
                scalar_handle, alpha_handle,
                stream_raw,
            )
        else:
            ws_size, executor = _ffi_tensor_scalar_op_no_alpha(
                <uintptr_t>getws_ptr, <uintptr_t>exec_ptr,
                py_shape, py_stride,
                py_shape, out_stride,
                dtype_code, 2,
                a_ptr, o_ptr,
                scalar_handle,
                stream_raw,
            )

        if ws_size:
            workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            try:
                ret = _ffi_execute(<uintptr_t>exec_ptr, int(workspace_ptr), ws_size, executor, stream_raw)
                if ret != 0:
                    raise RuntimeError(f"{pretty_name} execute failed: {ret}")
            finally:
                _get_runtime_fast(dev_idx).defer_raw_free(workspace_ptr)
        _defer_executor_handle(<uintptr_t>int(executor))
    finally:
        pass
    return _make_npu_tensor_fast_large(<int64_t>out_ptr, alloc_numel, a_dtype, a_dev,
                                       py_shape, out_stride, isize, dev_idx, a._dtype_code)


cpdef object fast_add_scalar_exact(TensorImpl a, object value):
    cdef object scalar_tensor
    _ensure_ffi_binary()
    if not _npu_in_graph_capture():
        scalar_tensor = _get_cached_scalar_tensor(a, value)
        return fast_add_exact(a, <TensorImpl>scalar_tensor)
    return _fast_tensor_scalar_exact(a, value, _adds_getws_ptr, _adds_exec_ptr, True, "aclnnAdds")


cdef object _fast_tensor_scalar_inplace_exact(TensorImpl a, object value, object getws_ptr,
                                              object exec_ptr, str pretty_name):
    """Run a covered NPU in-place tensor/scalar op entirely through Cython _C."""
    cdef int dev_idx = a._device_index
    if dev_idx < 0:
        dev_idx = 0
    cdef object py_shape = a._shape_tuple
    cdef object py_stride = a._stride_tuple
    cdef int ndim = a._ndim
    if ndim > MAX_NDIM:
        raise ValueError(f"ndim exceeds MAX_NDIM ({MAX_NDIM})")

    cdef int dtype_code = _tensor_dtype_to_acl_code(a)
    cdef uintptr_t a_ptr = a._storage._untyped._device_ptr + <uintptr_t>(a._c_offset * a._itemsize)
    cdef uintptr_t stream_raw = _get_stream_raw_fast(dev_idx)
    cdef uintptr_t scalar_handle = _get_cached_scalar_handle(dtype_code, value)
    cdef uintptr_t alpha_handle = _get_alpha_one(dtype_code)
    cdef object ws_size
    cdef object executor
    cdef object workspace_ptr
    cdef object ret

    ws_size, executor = _ffi_inplace_tensor_scalar_op_with_alpha(
        <uintptr_t>getws_ptr, <uintptr_t>exec_ptr,
        py_shape, py_stride,
        dtype_code, 2,
        a_ptr,
        scalar_handle, alpha_handle,
        stream_raw,
    )
    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(<uintptr_t>exec_ptr, int(workspace_ptr), ws_size, executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"{pretty_name} execute failed: {ret}")
        finally:
            _get_runtime_fast(dev_idx).defer_raw_free(workspace_ptr)
    _defer_executor_handle(<uintptr_t>int(executor))
    return a


cpdef object fast_add_scalar_inplace_exact(TensorImpl a, object value):
    """In-place add_(a, scalar) entirely through Cython _C; capture-safe.

    Outside graph capture, reuse the cached 0-d scalar tensor and the
    tensor-tensor in-place Add kernel (same rationale as fast_add_scalar_exact).
    During capture, run aclnnInplaceAdds directly so no scalar-tensor fill or
    H2D copy is recorded into the graph.
    """
    cdef object scalar_tensor
    _ensure_npu_imports()
    _ensure_ffi_binary()
    if not _npu_in_graph_capture():
        scalar_tensor = _get_cached_scalar_tensor(a, value)
        return fast_add_inplace(a, scalar_tensor)
    return _fast_tensor_scalar_inplace_exact(a, value, _inplace_adds_getws_ptr,
                                             _inplace_adds_exec_ptr, "aclnnInplaceAdds")


cpdef object fast_mul_scalar_exact(TensorImpl a, object value):
    cdef object scalar_tensor
    _ensure_ffi_binary()
    if not _npu_in_graph_capture():
        scalar_tensor = _get_cached_scalar_tensor(a, value)
        return fast_mul_exact(a, <TensorImpl>scalar_tensor)
    return _fast_tensor_scalar_exact(a, value, _muls_getws_ptr, _muls_exec_ptr, False, "aclnnMuls")


cpdef object fast_sub_scalar_exact(TensorImpl a, object value):
    cdef object scalar_tensor
    _ensure_ffi_binary()
    if not _npu_in_graph_capture():
        scalar_tensor = _get_cached_scalar_tensor(a, value)
        return fast_sub_exact(a, <TensorImpl>scalar_tensor)
    # TODO: re-enable native aclnnSubs when CANN 9.x no longer segfaults on the
    # direct scalar-subtract GetWorkspaceSize/Execute path.  Keep the covered
    # capture-safe path entirely in Cython/on-device by using equivalent Adds(-value).
    return _fast_tensor_scalar_exact(a, -value, _adds_getws_ptr, _adds_exec_ptr, True, "aclnnAdds")


cpdef object fast_div_scalar_exact(TensorImpl a, object value):
    cdef object scalar_tensor
    _ensure_ffi_binary()
    if not _npu_in_graph_capture():
        scalar_tensor = _get_cached_scalar_tensor(a, value)
        return fast_div_exact(a, <TensorImpl>scalar_tensor)
    return _fast_tensor_scalar_exact(a, value, _divs_getws_ptr, _divs_exec_ptr, False, "aclnnDivs")


cpdef object fast_rsub_scalar_exact(TensorImpl a, object value):
    cdef object scalar_tensor
    _ensure_ffi_binary()
    scalar_tensor = _get_cached_scalar_tensor(a, value)
    return fast_sub_exact(<TensorImpl>scalar_tensor, a)


cpdef object fast_rdiv_scalar_exact(TensorImpl a, object value):
    cdef object scalar_tensor
    _ensure_ffi_binary()
    scalar_tensor = _get_cached_scalar_tensor(a, value)
    return fast_div_exact(<TensorImpl>scalar_tensor, a)


def fast_eq(a, b):
    return fast_binary_op(a, b, None, "eq")


def fast_ne(a, b):
    return fast_binary_op(a, b, None, "ne")


def fast_le(a, b):
    return fast_binary_op(a, b, None, "le")


def fast_lt(a, b):
    return fast_binary_op(a, b, None, "lt")


def fast_gt(a, b):
    return fast_binary_op(a, b, None, "gt")


def fast_ge(a, b):
    return fast_binary_op(a, b, None, "ge")


def fast_logical_and(a, b):
    return fast_binary_op(a, b, None, "logical_and")


def fast_logical_or(a, b):
    return fast_binary_op(a, b, None, "logical_or")


def fast_logical_xor(a, b):
    return fast_binary_op(a, b, None, "logical_xor")


def fast_bitwise_and(a, b):
    return fast_binary_op(a, b, None, "bitwise_and")


def fast_bitwise_or(a, b):
    return fast_binary_op(a, b, None, "bitwise_or")


def fast_bitwise_xor(a, b):
    return fast_binary_op(a, b, None, "bitwise_xor")


def fast_bitwise_left_shift(a, b):
    return fast_binary_op(a, b, None, "bitwise_left_shift")


def fast_bitwise_right_shift(a, b):
    return fast_binary_op(a, b, None, "bitwise_right_shift")


cpdef object fast_add(object a, object b):
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
    if isinstance(a, TensorImpl):
        a_dev = (<TensorImpl>a)._device_obj
        a_dtype = (<TensorImpl>a)._dtype_obj
    else:
        a_dev = a.device
        a_dtype = a.dtype

    # 2. Get runtime + stream
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    # 3. Extract shapes/strides into C arrays
    py_a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    py_b_shape = (<TensorImpl>b)._shape_tuple if isinstance(b, TensorImpl) else b.shape
    py_a_stride = (<TensorImpl>a)._stride_tuple if isinstance(a, TensorImpl) else a.stride
    py_b_stride = (<TensorImpl>b)._stride_tuple if isinstance(b, TensorImpl) else b.stride
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
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fa, stream.stream)
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
        state = _ffi_pta_begin_add_cache_lookup(
            py_a_shape, py_a_stride,
            py_b_shape, py_b_stride,
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
                    # CANN's PTA hit_cache_v2 path rebinds the current tensor
                    # addresses through AddTensorAddrToCachedList while building
                    # the lookup key.  Therefore an executor hit is valid for the
                    # current input/output pointers even when they differ from the
                    # pointers that originally populated the cache.
                    if ws_size:
                        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
                        if ret != 0:
                            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
                        try:
                            ret = _ffi_execute(
                                _add_exec_ptr, int(workspace_ptr), ws_size,
                                executor, stream_raw)
                            if ret != 0:
                                raise RuntimeError(f"aclnnAdd execute failed: {ret}")
                        finally:
                            runtime.defer_raw_free(workspace_ptr)
                    else:
                        ret = _ffi_execute(_add_exec_ptr, 0, 0, executor, stream_raw)
                        if ret != 0:
                            raise RuntimeError(f"aclnnAdd execute failed: {ret}")
                    return _make_npu_tensor_fast(out_ptr, n, a_dtype, a_dev, out_shape, out_stride, isize)
                finally:
                    if pta_active:
                        _ffi_pta_end_cache_lookup()
                        pta_active = False
            # else: PTA miss — fall through with pta_active still set so the
            # GetWorkspaceSize path below runs inside the open PTA context and
            # the miss-path finally closes it.

    try:
        # 10. Cache miss: full GetWorkspaceSize + Execute path
        alpha_handle = _get_alpha_one(dtype_code)
        ws_size, executor = _ffi_binary_op_with_alpha(
            _add_getws_ptr, _add_exec_ptr,
            py_a_shape, py_a_stride,
            py_b_shape, py_b_stride,
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
                ret = _ffi_execute(
                    _add_exec_ptr, int(workspace_ptr), ws_size,
                    executor, stream_raw)
                if ret != 0:
                    raise RuntimeError(f"aclnnAdd execute failed: {ret}")
            finally:
                runtime.defer_raw_free(workspace_ptr)

        # 12. Defer executor cleanup by raw handle; fast paths already register
        #     descriptor cleanup in _aclnn_ffi, so the Python wrapper adds only
        #     type normalization overhead here.
        _defer_executor_handle(executor)
    finally:
        if pta_active:
            _ffi_pta_end_cache_lookup()

    # 13. Wrap output — construct Tensor entirely in Cython (same as fast_binary_op)
    return _make_npu_tensor_fast(out_ptr, n, a_dtype, a_dev, out_shape, out_stride, isize)


cpdef object fast_add_exact(TensorImpl a, TensorImpl b):
    """Add for already-validated exact base NPU tensors.

    Called only from the public eager wrapper after it has checked the exact
    Candle Tensor type and NPU/global-dispatch guards.  Keep the dtype check here
    because the public exact guard does not prove it.
    """
    _ensure_npu_imports()
    _ensure_ffi_binary()

    cdef int dev_idx = a._device_index
    cdef object a_dev = a._device_obj
    cdef object a_dtype = a._dtype_obj
    cdef object runtime = None
    cdef object stream_obj = _get_stream_obj_fast(dev_idx)
    cdef object py_a_shape = a._shape_tuple
    cdef object py_b_shape = b._shape_tuple
    cdef object py_a_stride = a._stride_tuple
    cdef object py_b_stride = b._stride_tuple
    cdef int a_ndim = len(py_a_shape)
    cdef int b_ndim = len(py_b_shape)
    cdef int out_ndim
    cdef int64_t n
    cdef int64_t[MAX_NDIM] a_shape_buf, b_shape_buf
    cdef int64_t[MAX_NDIM] out_shape_buf, out_stride_buf
    cdef int isize
    cdef int64_t alloc_size_fa
    cdef int dtype_code
    cdef uintptr_t a_ptr, b_ptr, o_ptr
    cdef uintptr_t stream_raw
    cdef bint pta_active = False
    cdef uintptr_t alpha_handle
    cdef uint64_t ws_size_raw = 0
    cdef uintptr_t executor_raw = 0
    cdef int pta_lookup
    cdef int ret_i
    cdef object out_shape
    cdef object out_stride
    cdef object out_ptr
    cdef bint same_contiguous_layout
    cdef object alpha_bytes_pair
    cdef object ws_size
    cdef object executor
    cdef object workspace_ptr
    cdef object ret

    if a._device_index != b._device_index:
        raise ValueError("NPU add requires tensors on the same device")
    if a._dtype_code != b._dtype_code:
        raise ValueError("NPU add requires matching dtypes")
    if a_ndim > MAX_NDIM or b_ndim > MAX_NDIM:
        raise ValueError(f"ndim exceeds MAX_NDIM ({MAX_NDIM})")

    same_contiguous_layout = (
        a._ndim == b._ndim
        and a._c_numel == b._c_numel
        and a._shape_tuple == b._shape_tuple
        and _tensor_has_strict_contiguous_stride(a)
        and _tensor_has_strict_contiguous_stride(b)
    )
    if same_contiguous_layout:
        out_ndim = a_ndim
        n = a._c_numel
        out_shape = py_a_shape
        out_stride = py_a_stride
    else:
        _fill_shape(py_a_shape, a_shape_buf, a_ndim)
        _fill_shape(py_b_shape, b_shape_buf, b_ndim)
        with nogil:
            out_ndim = c_broadcast_shape(
                a_shape_buf, a_ndim, b_shape_buf, b_ndim, out_shape_buf)
            c_contiguous_stride(out_shape_buf, out_ndim, out_stride_buf)
            n = c_numel(out_shape_buf, out_ndim)
        out_shape = _to_tuple(out_shape_buf, out_ndim)
        out_stride = _to_tuple(out_stride_buf, out_ndim)
    isize = a._itemsize
    alloc_size_fa = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc_large_cached(alloc_size_fa, stream_obj)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fa, stream=stream_obj)

    dtype_code = _tensor_dtype_to_acl_code(a)
    a_ptr = a._storage._untyped._device_ptr
    b_ptr = b._storage._untyped._device_ptr
    o_ptr = out_ptr
    stream_raw = _get_stream_raw_fast(dev_idx)

    if _use_add_pta_cache and _pta_cache_begin_fn is not None:
        alpha_bytes_pair = _get_alpha_one_bytes(dtype_code)
        pta_lookup = _ffi_pta_begin_add_cache_lookup_raw(
            py_a_shape, py_a_stride,
            py_b_shape, py_b_stride,
            out_shape, out_stride,
            dtype_code,
            a_ptr, b_ptr, o_ptr,
            alpha_bytes_pair[0], alpha_bytes_pair[1],
            stream_raw,
            &pta_active,
            &ws_size_raw,
            &executor_raw)
        if pta_lookup and executor_raw != 0:
            try:
                if ws_size_raw:
                    workspace_ptr, ret = _acl_rt_malloc_fn(ws_size_raw, 0)
                    if ret != 0:
                        raise RuntimeError(f"acl.rt.malloc failed: {ret}")
                    try:
                        ret_i = _ffi_execute(
                            _add_exec_ptr, <uintptr_t>int(workspace_ptr), ws_size_raw,
                            executor_raw, stream_raw)
                        if ret_i != 0:
                            raise RuntimeError(f"aclnnAdd execute failed: {ret_i}")
                    finally:
                        if runtime is None:
                            runtime = _get_runtime_fast(dev_idx)
                        runtime.defer_raw_free(workspace_ptr)
                else:
                    ret_i = _ffi_execute(_add_exec_ptr, 0, 0, executor_raw, stream_raw)
                    if ret_i != 0:
                        raise RuntimeError(f"aclnnAdd execute failed: {ret_i}")
                return _make_npu_tensor_fast_large(out_ptr, n, a_dtype, a_dev, out_shape, out_stride, isize, dev_idx, a._dtype_code)
            finally:
                if pta_active:
                    _ffi_pta_end_cache_lookup()
                    pta_active = False

    try:
        alpha_handle = _get_alpha_one(dtype_code)
        ws_size, executor = _ffi_binary_op_with_alpha(
            _add_getws_ptr, _add_exec_ptr,
            py_a_shape, py_a_stride,
            py_b_shape, py_b_stride,
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
                ret = _ffi_execute(
                    _add_exec_ptr, int(workspace_ptr), ws_size,
                    executor, stream_raw)
                if ret != 0:
                    raise RuntimeError(f"aclnnAdd execute failed: {ret}")
            finally:
                if runtime is None:
                    runtime = _get_runtime_fast(dev_idx)
                runtime.defer_raw_free(workspace_ptr)
        _defer_executor_handle(executor)
    finally:
        if pta_active:
            _ffi_pta_end_cache_lookup()

    return _make_npu_tensor_fast_large(out_ptr, n, a_dtype, a_dev, out_shape, out_stride, isize, dev_idx, a._dtype_code)


cpdef object fast_sub_exact(TensorImpl a, TensorImpl b):
    """Sub for already-validated exact base NPU tensors."""
    _ensure_npu_imports()
    _ensure_ffi_binary()

    cdef int dev_idx = a._device_index
    cdef object a_dev = a._device_obj
    cdef object a_dtype = a._dtype_obj
    cdef object runtime = None
    cdef object stream_obj = _get_stream_obj_fast(dev_idx)
    cdef object py_a_shape = a._shape_tuple
    cdef object py_b_shape = b._shape_tuple
    cdef object py_a_stride = a._stride_tuple
    cdef object py_b_stride = b._stride_tuple
    cdef int a_ndim = len(py_a_shape)
    cdef int b_ndim = len(py_b_shape)
    cdef int out_ndim
    cdef int64_t n
    cdef int64_t[MAX_NDIM] a_shape_buf, b_shape_buf
    cdef int64_t[MAX_NDIM] out_shape_buf, out_stride_buf
    cdef int isize
    cdef int64_t alloc_size_fs
    cdef int dtype_code
    cdef uintptr_t a_ptr, b_ptr, o_ptr
    cdef uintptr_t stream_raw
    cdef bint pta_active = False
    cdef uintptr_t alpha_handle
    cdef uint64_t ws_size_raw = 0
    cdef uintptr_t executor_raw = 0
    cdef int pta_lookup
    cdef int ret_i
    cdef object out_shape
    cdef object out_stride
    cdef object out_ptr
    cdef bint same_contiguous_layout
    cdef object alpha_bytes_pair
    cdef object ws_size
    cdef object executor
    cdef object workspace_ptr
    cdef object ret

    if a._device_index != b._device_index:
        raise ValueError("NPU sub requires tensors on the same device")
    if a._dtype_code != b._dtype_code:
        raise ValueError("NPU sub requires matching dtypes")
    if a_ndim > MAX_NDIM or b_ndim > MAX_NDIM:
        raise ValueError(f"ndim exceeds MAX_NDIM ({MAX_NDIM})")

    same_contiguous_layout = (
        a._ndim == b._ndim
        and a._c_numel == b._c_numel
        and a._shape_tuple == b._shape_tuple
        and _tensor_has_strict_contiguous_stride(a)
        and _tensor_has_strict_contiguous_stride(b)
    )
    if same_contiguous_layout:
        out_ndim = a_ndim
        n = a._c_numel
        out_shape = py_a_shape
        out_stride = py_a_stride
    else:
        _fill_shape(py_a_shape, a_shape_buf, a_ndim)
        _fill_shape(py_b_shape, b_shape_buf, b_ndim)
        with nogil:
            out_ndim = c_broadcast_shape(
                a_shape_buf, a_ndim, b_shape_buf, b_ndim, out_shape_buf)
            c_contiguous_stride(out_shape_buf, out_ndim, out_stride_buf)
            n = c_numel(out_shape_buf, out_ndim)
        out_shape = _to_tuple(out_shape_buf, out_ndim)
        out_stride = _to_tuple(out_stride_buf, out_ndim)

    isize = a._itemsize
    alloc_size_fs = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc_large_cached(alloc_size_fs, stream_obj)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fs, stream=stream_obj)

    dtype_code = _tensor_dtype_to_acl_code(a)
    a_ptr = a._storage._untyped._device_ptr
    b_ptr = b._storage._untyped._device_ptr
    o_ptr = out_ptr
    stream_raw = _get_stream_raw_fast(dev_idx)

    if _use_sub_pta_cache and _pta_cache_end_fn is not None:
        alpha_bytes_pair = _get_alpha_one_bytes(dtype_code)
        pta_lookup = _ffi_pta_begin_binary_alpha_cache_lookup_raw(
            b"aclnnSub",
            py_a_shape, py_a_stride,
            py_b_shape, py_b_stride,
            out_shape, out_stride,
            dtype_code,
            a_ptr, b_ptr, o_ptr,
            alpha_bytes_pair[0], alpha_bytes_pair[1],
            stream_raw,
            &pta_active,
            &ws_size_raw,
            &executor_raw)
        if pta_lookup and executor_raw != 0:
            try:
                if ws_size_raw:
                    workspace_ptr, ret = _acl_rt_malloc_fn(ws_size_raw, 0)
                    if ret != 0:
                        raise RuntimeError(f"acl.rt.malloc failed: {ret}")
                    try:
                        ret_i = _ffi_execute(
                            _sub_exec_ptr, <uintptr_t>int(workspace_ptr), ws_size_raw,
                            executor_raw, stream_raw)
                        if ret_i != 0:
                            raise RuntimeError(f"aclnnSub execute failed: {ret_i}")
                    finally:
                        if runtime is None:
                            runtime = _get_runtime_fast(dev_idx)
                        runtime.defer_raw_free(workspace_ptr)
                else:
                    ret_i = _ffi_execute(_sub_exec_ptr, 0, 0, executor_raw, stream_raw)
                    if ret_i != 0:
                        raise RuntimeError(f"aclnnSub execute failed: {ret_i}")
                return _make_npu_tensor_fast_large(out_ptr, n, a_dtype, a_dev, out_shape, out_stride, isize, dev_idx, a._dtype_code)
            finally:
                if pta_active:
                    _ffi_pta_end_cache_lookup()
                    pta_active = False

    try:
        alpha_handle = _get_alpha_one(dtype_code)
        ws_size, executor = _ffi_binary_op_with_alpha(
            _sub_getws_ptr, _sub_exec_ptr,
            py_a_shape, py_a_stride,
            py_b_shape, py_b_stride,
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
                ret = _ffi_execute(
                    _sub_exec_ptr, int(workspace_ptr), ws_size,
                    executor, stream_raw)
                if ret != 0:
                    raise RuntimeError(f"aclnnSub execute failed: {ret}")
            finally:
                if runtime is None:
                    runtime = _get_runtime_fast(dev_idx)
                runtime.defer_raw_free(workspace_ptr)
        _defer_executor_handle(executor)
    finally:
        if pta_active:
            _ffi_pta_end_cache_lookup()

    return _make_npu_tensor_fast_large(out_ptr, n, a_dtype, a_dev, out_shape, out_stride, isize, dev_idx, a._dtype_code)


cpdef object fast_div_exact(TensorImpl a, TensorImpl b):
    """Div for already-validated exact base NPU tensors."""
    _ensure_npu_imports()
    _ensure_ffi_binary()

    cdef int dev_idx = a._device_index
    cdef object a_dev = a._device_obj
    cdef object a_dtype = a._dtype_obj
    cdef object runtime = None
    cdef object stream_obj = _get_stream_obj_fast(dev_idx)
    cdef object py_a_shape = a._shape_tuple
    cdef object py_b_shape = b._shape_tuple
    cdef object py_a_stride = a._stride_tuple
    cdef object py_b_stride = b._stride_tuple
    cdef int a_ndim = len(py_a_shape)
    cdef int b_ndim = len(py_b_shape)
    cdef int out_ndim
    cdef int64_t n
    cdef int64_t[MAX_NDIM] a_shape_buf, b_shape_buf
    cdef int64_t[MAX_NDIM] out_shape_buf, out_stride_buf
    cdef int isize
    cdef int64_t alloc_size_fd
    cdef int dtype_code
    cdef uintptr_t a_ptr, b_ptr, o_ptr
    cdef uintptr_t stream_raw
    cdef bint pta_active = False
    cdef bint pta_cache_miss = False
    cdef bint pta_pointer_guard = False
    cdef bint inputs_alias
    cdef object pta_key = None
    cdef object pointer_key = None
    cdef object out_shape
    cdef object out_stride
    cdef object out_ptr
    cdef bint same_contiguous_layout
    cdef uint64_t ws_size_raw = 0
    cdef uintptr_t executor_raw = 0
    cdef int pta_lookup
    cdef int ret_i
    cdef object ws_size
    cdef object executor
    cdef object workspace_ptr
    cdef object ret

    if a._device_index != b._device_index:
        raise ValueError("NPU div requires tensors on the same device")
    if a._dtype_code != b._dtype_code:
        raise ValueError("NPU div requires matching dtypes")
    if a_ndim > MAX_NDIM or b_ndim > MAX_NDIM:
        raise ValueError(f"ndim exceeds MAX_NDIM ({MAX_NDIM})")

    same_contiguous_layout = (
        a._ndim == b._ndim
        and a._c_numel == b._c_numel
        and a._shape_tuple == b._shape_tuple
        and _tensor_has_strict_contiguous_stride(a)
        and _tensor_has_strict_contiguous_stride(b)
    )
    if same_contiguous_layout:
        out_ndim = a_ndim
        n = a._c_numel
        out_shape = py_a_shape
        out_stride = py_a_stride
    else:
        _fill_shape(py_a_shape, a_shape_buf, a_ndim)
        _fill_shape(py_b_shape, b_shape_buf, b_ndim)
        with nogil:
            out_ndim = c_broadcast_shape(
                a_shape_buf, a_ndim, b_shape_buf, b_ndim, out_shape_buf)
            c_contiguous_stride(out_shape_buf, out_ndim, out_stride_buf)
            n = c_numel(out_shape_buf, out_ndim)
        out_shape = _to_tuple(out_shape_buf, out_ndim)
        out_stride = _to_tuple(out_stride_buf, out_ndim)

    isize = a._itemsize
    alloc_size_fd = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc_large_cached(alloc_size_fd, stream_obj)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fd, stream=stream_obj)

    dtype_code = _tensor_dtype_to_acl_code(a)
    a_ptr = a._storage._untyped._device_ptr
    b_ptr = b._storage._untyped._device_ptr
    o_ptr = out_ptr
    stream_raw = _get_stream_raw_fast(dev_idx)
    inputs_alias = a_ptr == b_ptr
    if inputs_alias or not same_contiguous_layout:
        pta_pointer_guard = True

    if _use_div_pta_cache and _pta_binary_begin_fn is not None:
        pta_lookup = _ffi_pta_begin_binary_cache_lookup_raw(
            b"aclnnDiv",
            py_a_shape, py_a_stride,
            py_b_shape, py_b_stride,
            out_shape, out_stride,
            dtype_code,
            a_ptr, b_ptr, o_ptr,
            stream_raw,
            &pta_active,
            &ws_size_raw,
            &executor_raw)
        if pta_lookup:
            pta_key = (py_a_shape, py_a_stride, py_b_shape, py_b_stride,
                       out_shape, out_stride, dtype_code)
            pointer_key = (a_ptr, b_ptr, o_ptr)
            if executor_raw != 0:
                if (not pta_pointer_guard) or _div_pta_pointer_keys.get(pta_key) == pointer_key:
                    try:
                        if ws_size_raw:
                            workspace_ptr, ret = _acl_rt_malloc_fn(ws_size_raw, 0)
                            if ret != 0:
                                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
                            try:
                                ret_i = _ffi_execute(
                                    _div_exec_ptr, <uintptr_t>int(workspace_ptr), ws_size_raw,
                                    executor_raw, stream_raw)
                                if ret_i != 0:
                                    raise RuntimeError(f"aclnnDiv execute failed: {ret_i}")
                            finally:
                                if runtime is None:
                                    runtime = _get_runtime_fast(dev_idx)
                                runtime.defer_raw_free(workspace_ptr)
                        else:
                            ret_i = _ffi_execute(_div_exec_ptr, 0, 0, executor_raw, stream_raw)
                            if ret_i != 0:
                                raise RuntimeError(f"aclnnDiv execute failed: {ret_i}")
                        return _make_npu_tensor_fast_large(out_ptr, n, a_dtype, a_dev, out_shape, out_stride, isize, dev_idx, a._dtype_code)
                    finally:
                        if pta_active:
                            _ffi_pta_end_cache_lookup()
                            pta_active = False
                if pta_active:
                    _ffi_pta_end_cache_lookup()
                    pta_active = False
            else:
                pta_cache_miss = pta_active

    try:
        ws_size, executor = _ffi_binary_op_no_alpha(
            _div_getws_ptr, _div_exec_ptr,
            py_a_shape, py_a_stride,
            py_b_shape, py_b_stride,
            out_shape, out_stride,
            dtype_code, 2,
            a_ptr, b_ptr, o_ptr,
            stream_raw)
        if ws_size:
            workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            try:
                ret = _ffi_execute(
                    _div_exec_ptr, int(workspace_ptr), ws_size,
                    executor, stream_raw)
                if ret != 0:
                    raise RuntimeError(f"aclnnDiv execute failed: {ret}")
            finally:
                if runtime is None:
                    runtime = _get_runtime_fast(dev_idx)
                runtime.defer_raw_free(workspace_ptr)
        if pta_pointer_guard and pta_cache_miss and pta_key is not None:
            _div_pta_pointer_keys[pta_key] = pointer_key
        _defer_executor_handle(executor)
    finally:
        if pta_active:
            _ffi_pta_end_cache_lookup()

    return _make_npu_tensor_fast_large(out_ptr, n, a_dtype, a_dev, out_shape, out_stride, isize, dev_idx, a._dtype_code)


# ---------------------------------------------------------------------------
# fast_mul — hardwired mul(a, b) that skips aclnn.py wrapper
# ---------------------------------------------------------------------------

cpdef object fast_mul(object a, object b):
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
    if isinstance(a, TensorImpl):
        a_dev = (<TensorImpl>a)._device_obj
        a_dtype = (<TensorImpl>a)._dtype_obj
    else:
        a_dev = a.device
        a_dtype = a.dtype

    # 2. Get runtime + stream
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    # 3. Extract shapes/strides into C arrays
    py_a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    py_b_shape = (<TensorImpl>b)._shape_tuple if isinstance(b, TensorImpl) else b.shape
    py_a_stride = (<TensorImpl>a)._stride_tuple if isinstance(a, TensorImpl) else a.stride
    py_b_stride = (<TensorImpl>b)._stride_tuple if isinstance(b, TensorImpl) else b.stride
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
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fm, stream.stream)
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
    cdef bint pta_active = False
    cdef bint pta_cache_miss = False
    cdef bint pta_pointer_guard = False
    cdef object pta_key = None
    cdef object pointer_key = None
    cdef bint inputs_alias = a_ptr == b_ptr
    if py_a_shape != py_b_shape or py_a_stride != py_b_stride or inputs_alias:
        pta_pointer_guard = True

    # 9. Try PTA executor cache (torch_npu-aligned hit_cache_v2 path)
    if _use_mul_pta_cache and _pta_binary_begin_fn is not None:
        state = _ffi_pta_begin_binary_cache_lookup(
            b"aclnnMul",
            py_a_shape, py_a_stride,
            py_b_shape, py_b_stride,
            out_shape, out_stride,
            dtype_code,
            a_ptr, b_ptr, o_ptr,
            stream_raw)
        if state is not None:
            pta_active = bool(state[0])
            ws_size = state[1]
            executor = state[2]
            pta_key = (py_a_shape, py_a_stride, py_b_shape, py_b_stride,
                       out_shape, out_stride, dtype_code)
            pointer_key = (a_ptr, b_ptr, o_ptr)
            if executor:
                if (not pta_pointer_guard) or _mul_pta_pointer_keys.get(pta_key) == pointer_key:
                    try:
                        # Same-shape non-aliased tensor Mul follows torch_npu's
                        # hit_cache_v2 path and relies on the alias-aware PTA hash
                        # plus AddTensorAddrToCachedList rebinding current addresses.
                        # Broadcast and aliased-input Mul keep a pointer guard because
                        # this CANN build can reuse stale bound tensor addresses after
                        # prior native traffic.
                        if ws_size:
                            workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
                            if ret != 0:
                                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
                            try:
                                ret = _ffi_execute(
                                    _mul_exec_ptr, int(workspace_ptr), ws_size,
                                    executor, stream_raw)
                                if ret != 0:
                                    raise RuntimeError(f"aclnnMul execute failed: {ret}")
                            finally:
                                runtime.defer_raw_free(workspace_ptr)
                        else:
                            ret = _ffi_execute(_mul_exec_ptr, 0, 0, executor, stream_raw)
                            if ret != 0:
                                raise RuntimeError(f"aclnnMul execute failed: {ret}")
                        return _make_npu_tensor_fast(out_ptr, n, a_dtype, a_dev, out_shape, out_stride, isize)
                    finally:
                        if pta_active:
                            _ffi_pta_end_cache_lookup()
                            pta_active = False
                if pta_active:
                    _ffi_pta_end_cache_lookup()
                    pta_active = False
            else:
                pta_cache_miss = pta_active

    try:
        # 10. Cache miss: full GetWorkspaceSize + Execute path
        ws_size, executor = _ffi_binary_op_no_alpha(
            _mul_getws_ptr, _mul_exec_ptr,
            py_a_shape, py_a_stride,
            py_b_shape, py_b_stride,
            out_shape, out_stride,
            dtype_code, 2,  # ACL_FORMAT_ND = 2
            a_ptr, b_ptr, o_ptr,
            stream_raw)

        # 11. Handle workspace (rare: ws_size > 0 means execute not yet called)
        if ws_size:
            workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            try:
                ret = _ffi_execute(
                    _mul_exec_ptr, int(workspace_ptr), ws_size,
                    executor, stream_raw)
                if ret != 0:
                    raise RuntimeError(f"aclnnMul execute failed: {ret}")
            finally:
                if runtime is None:
                    runtime = _get_runtime_fast(dev_idx)
                runtime.defer_raw_free(workspace_ptr)

        if pta_pointer_guard and pta_cache_miss and pta_key is not None:
            _mul_pta_pointer_keys[pta_key] = pointer_key

        # 12. Defer executor cleanup by raw handle; descriptor cleanup is registered
        #     in _aclnn_ffi for this executor.
        _defer_executor_handle(executor)
    finally:
        if pta_active:
            _ffi_pta_end_cache_lookup()

    # 13. Wrap output
    return _make_npu_tensor_fast(out_ptr, n, a_dtype, a_dev, out_shape, out_stride, isize)


cpdef object fast_mul_exact(TensorImpl a, TensorImpl b):
    """Mul for already-validated exact base NPU tensors."""
    _ensure_npu_imports()
    _ensure_ffi_binary()

    cdef int dev_idx = a._device_index
    cdef object a_dev = a._device_obj
    cdef object a_dtype = a._dtype_obj
    cdef object runtime = None
    cdef object stream_obj = _get_stream_obj_fast(dev_idx)
    cdef object py_a_shape = a._shape_tuple
    cdef object py_b_shape = b._shape_tuple
    cdef object py_a_stride = a._stride_tuple
    cdef object py_b_stride = b._stride_tuple
    cdef int a_ndim = len(py_a_shape)
    cdef int b_ndim = len(py_b_shape)
    cdef int out_ndim
    cdef int64_t n
    cdef int64_t[MAX_NDIM] a_shape_buf, b_shape_buf
    cdef int64_t[MAX_NDIM] out_shape_buf, out_stride_buf
    cdef int isize
    cdef int64_t alloc_size_fm
    cdef int dtype_code
    cdef uintptr_t a_ptr, b_ptr, o_ptr
    cdef uintptr_t stream_raw
    cdef bint pta_active = False
    cdef bint pta_cache_miss = False
    cdef bint pta_pointer_guard = False
    cdef bint inputs_alias
    cdef object pta_key = None
    cdef object pointer_key = None
    cdef object out_shape
    cdef object out_stride
    cdef object out_ptr
    cdef bint same_contiguous_layout
    cdef uint64_t ws_size_raw = 0
    cdef uintptr_t executor_raw = 0
    cdef int pta_lookup
    cdef int ret_i
    cdef object ws_size
    cdef object executor
    cdef object workspace_ptr
    cdef object ret

    if a._device_index != b._device_index:
        raise ValueError("NPU mul requires tensors on the same device")
    if a._dtype_code != b._dtype_code:
        raise ValueError("NPU mul requires matching dtypes")
    if a_ndim > MAX_NDIM or b_ndim > MAX_NDIM:
        raise ValueError(f"ndim exceeds MAX_NDIM ({MAX_NDIM})")

    same_contiguous_layout = (
        a._ndim == b._ndim
        and a._c_numel == b._c_numel
        and a._shape_tuple == b._shape_tuple
        and _tensor_has_strict_contiguous_stride(a)
        and _tensor_has_strict_contiguous_stride(b)
    )
    if same_contiguous_layout:
        out_ndim = a_ndim
        n = a._c_numel
        out_shape = py_a_shape
        out_stride = py_a_stride
    else:
        _fill_shape(py_a_shape, a_shape_buf, a_ndim)
        _fill_shape(py_b_shape, b_shape_buf, b_ndim)
        with nogil:
            out_ndim = c_broadcast_shape(
                a_shape_buf, a_ndim, b_shape_buf, b_ndim, out_shape_buf)
            c_contiguous_stride(out_shape_buf, out_ndim, out_stride_buf)
            n = c_numel(out_shape_buf, out_ndim)
        out_shape = _to_tuple(out_shape_buf, out_ndim)
        out_stride = _to_tuple(out_stride_buf, out_ndim)
    isize = a._itemsize
    alloc_size_fm = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc_large_cached(alloc_size_fm, stream_obj)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fm, stream=stream_obj)

    dtype_code = _tensor_dtype_to_acl_code(a)
    a_ptr = a._storage._untyped._device_ptr
    b_ptr = b._storage._untyped._device_ptr
    o_ptr = out_ptr
    stream_raw = _get_stream_raw_fast(dev_idx)
    inputs_alias = a_ptr == b_ptr
    if inputs_alias or not same_contiguous_layout:
        pta_pointer_guard = True

    if _use_mul_pta_cache and _pta_binary_begin_fn is not None:
        pta_lookup = _ffi_pta_begin_binary_cache_lookup_raw(
            b"aclnnMul",
            py_a_shape, py_a_stride,
            py_b_shape, py_b_stride,
            out_shape, out_stride,
            dtype_code,
            a_ptr, b_ptr, o_ptr,
            stream_raw,
            &pta_active,
            &ws_size_raw,
            &executor_raw)
        if pta_lookup:
            pta_key = (py_a_shape, py_a_stride, py_b_shape, py_b_stride,
                       out_shape, out_stride, dtype_code)
            pointer_key = (a_ptr, b_ptr, o_ptr)
            if executor_raw != 0:
                if (not pta_pointer_guard) or _mul_pta_pointer_keys.get(pta_key) == pointer_key:
                    try:
                        if ws_size_raw:
                            workspace_ptr, ret = _acl_rt_malloc_fn(ws_size_raw, 0)
                            if ret != 0:
                                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
                            try:
                                ret_i = _ffi_execute(
                                    _mul_exec_ptr, <uintptr_t>int(workspace_ptr), ws_size_raw,
                                    executor_raw, stream_raw)
                                if ret_i != 0:
                                    raise RuntimeError(f"aclnnMul execute failed: {ret_i}")
                            finally:
                                if runtime is None:
                                    runtime = _get_runtime_fast(dev_idx)
                                runtime.defer_raw_free(workspace_ptr)
                        else:
                            ret_i = _ffi_execute(_mul_exec_ptr, 0, 0, executor_raw, stream_raw)
                            if ret_i != 0:
                                raise RuntimeError(f"aclnnMul execute failed: {ret_i}")
                        return _make_npu_tensor_fast_large(out_ptr, n, a_dtype, a_dev, out_shape, out_stride, isize, dev_idx, a._dtype_code)
                    finally:
                        if pta_active:
                            _ffi_pta_end_cache_lookup()
                            pta_active = False
                if pta_active:
                    _ffi_pta_end_cache_lookup()
                    pta_active = False
            else:
                pta_cache_miss = pta_active

    try:
        ws_size, executor = _ffi_binary_op_no_alpha(
            _mul_getws_ptr, _mul_exec_ptr,
            py_a_shape, py_a_stride,
            py_b_shape, py_b_stride,
            out_shape, out_stride,
            dtype_code, 2,
            a_ptr, b_ptr, o_ptr,
            stream_raw)
        if ws_size:
            workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            try:
                ret = _ffi_execute(
                    _mul_exec_ptr, int(workspace_ptr), ws_size,
                    executor, stream_raw)
                if ret != 0:
                    raise RuntimeError(f"aclnnMul execute failed: {ret}")
            finally:
                if runtime is None:
                    runtime = _get_runtime_fast(dev_idx)
                runtime.defer_raw_free(workspace_ptr)

        if pta_pointer_guard and pta_cache_miss and pta_key is not None:
            _mul_pta_pointer_keys[pta_key] = pointer_key
        _defer_executor_handle(executor)
    finally:
        if pta_active:
            _ffi_pta_end_cache_lookup()

    return _make_npu_tensor_fast_large(out_ptr, n, a_dtype, a_dev, out_shape, out_stride, isize, dev_idx, a._dtype_code)


cdef inline tuple _shape_tuple_from_buf(const int64_t* buf, int ndim):
    return tuple([buf[i] for i in range(ndim)])


cdef inline bint _sdpa_flash_valid_layout(TensorImpl t, int64_t B, int64_t H, int64_t S, int64_t D) noexcept:
    if t._ndim != 4:
        return False
    if t._c_shape[0] != B or t._c_shape[1] != H or t._c_shape[2] != S or t._c_shape[3] != D:
        return False
    if t._c_stride[3] != 1:
        return False
    return True


cdef inline bint _sdpa_flash_valid_grad_layout(TensorImpl t, int64_t B, int64_t H, int64_t S, int64_t D) noexcept:
    if t._ndim != 4:
        return False
    if t._c_shape[0] != B or t._c_shape[1] != H or t._c_shape[2] != S or t._c_shape[3] != D:
        return False
    if t._c_stride[3] == 1:
        return True
    return t._c_stride[0] == 0 and t._c_stride[1] == 0 and t._c_stride[2] == 0 and t._c_stride[3] == 0


cdef inline object _run_executor(uintptr_t exec_ptr, uint64_t ws_size, uintptr_t executor,
                                uintptr_t stream_raw, int dev_idx, object runtime,
                                str op_name):
    cdef object workspace_ptr
    cdef int ret_i
    if ws_size:
        if dev_idx == 0:
            _ensure_allocator_dev0()
            workspace_ptr = _fast_allocator_dev0.malloc_large_cached(<int64_t>ws_size, _get_stream_obj_fast(dev_idx))
        else:
            workspace_ptr = _get_allocator_fn_ref(dev_idx).malloc(<int64_t>ws_size, stream=_get_stream_obj_fast(dev_idx))
        try:
            ret_i = _ffi_execute(exec_ptr, <uintptr_t>int(workspace_ptr), ws_size, executor, stream_raw)
            if ret_i != 0:
                raise RuntimeError(f"{op_name} execute failed: {ret_i}")
        finally:
            if runtime is None:
                runtime = _get_runtime_fast(dev_idx)
            runtime.defer_free(workspace_ptr)
    else:
        ret_i = _ffi_execute(exec_ptr, 0, 0, executor, stream_raw)
        if ret_i != 0:
            raise RuntimeError(f"{op_name} execute failed: {ret_i}")
    return runtime


cpdef object fast_layer_norm(object input, object normalized_shape, object weight=None, object bias=None, object eps=1e-5):
    """Cython NPU LayerNorm forward wrapper with native ACLNN execution."""
    _ensure_npu_imports()
    _ensure_ffi_layer_norm()

    if not isinstance(input, TensorImpl):
        raise ValueError("fast_layer_norm expects a base TensorImpl input")
    cdef TensorImpl x = <TensorImpl>input
    if x._device_type != 1:
        raise ValueError("fast_layer_norm expects an NPU tensor")
    cdef TensorImpl w
    cdef TensorImpl b
    cdef bint has_weight = weight is not None
    cdef bint has_bias = bias is not None
    if has_weight:
        if not isinstance(weight, TensorImpl):
            raise ValueError("fast_layer_norm expects TensorImpl weight")
        w = <TensorImpl>weight
        if w._device_type != 1 or w._device_index != x._device_index or w._dtype_code != x._dtype_code:
            raise ValueError("fast_layer_norm requires matching NPU weight")
    if has_bias:
        if not isinstance(bias, TensorImpl):
            raise ValueError("fast_layer_norm expects TensorImpl bias")
        b = <TensorImpl>bias
        if b._device_type != 1 or b._device_index != x._device_index or b._dtype_code != x._dtype_code:
            raise ValueError("fast_layer_norm requires matching NPU bias")

    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    else:
        normalized_shape = tuple(normalized_shape)
    cdef int norm_ndim = len(normalized_shape)
    if norm_ndim <= 0 or norm_ndim > x._ndim:
        raise ValueError("fast_layer_norm expects a non-empty normalized_shape suffix")
    cdef int i
    cdef int lead = x._ndim - norm_ndim
    for i in range(norm_ndim):
        if x._c_shape[lead + i] != <int64_t>normalized_shape[i]:
            raise ValueError("fast_layer_norm normalized_shape does not match input suffix")
    if has_weight:
        if w._ndim != norm_ndim:
            raise ValueError("fast_layer_norm weight rank mismatch")
        for i in range(norm_ndim):
            if w._c_shape[i] != <int64_t>normalized_shape[i]:
                raise ValueError("fast_layer_norm weight shape mismatch")
    if has_bias:
        if b._ndim != norm_ndim:
            raise ValueError("fast_layer_norm bias rank mismatch")
        for i in range(norm_ndim):
            if b._c_shape[i] != <int64_t>normalized_shape[i]:
                raise ValueError("fast_layer_norm bias shape mismatch")

    cdef int dev_idx = x._device_index
    cdef object x_dev = x._device_obj
    cdef object x_dtype = x._dtype_obj
    cdef int dtype_code = _tensor_dtype_to_acl_code(x)
    cdef int isize = x._itemsize
    cdef object stream_obj = _get_stream_obj_fast(dev_idx)
    cdef uintptr_t stream_raw = _get_stream_raw_fast(dev_idx)
    cdef int64_t out_numel = x._c_numel
    cdef int64_t out_alloc = max(out_numel, 1) * isize
    cdef int64_t stats_numel = 1
    cdef int64_t[MAX_NDIM] out_shape_buf
    cdef int64_t[MAX_NDIM] out_stride_buf
    cdef int64_t[MAX_NDIM] stats_shape_buf
    cdef int64_t[MAX_NDIM] stats_stride_buf
    for i in range(x._ndim):
        out_shape_buf[i] = x._c_shape[i]
    c_contiguous_stride(out_shape_buf, x._ndim, out_stride_buf)
    for i in range(x._ndim):
        if i < lead:
            stats_shape_buf[i] = x._c_shape[i]
        else:
            stats_shape_buf[i] = 1
        stats_numel = stats_numel * stats_shape_buf[i]
    c_contiguous_stride(stats_shape_buf, x._ndim, stats_stride_buf)
    cdef tuple out_shape = _shape_tuple_from_buf(out_shape_buf, x._ndim)
    cdef tuple out_stride = _shape_tuple_from_buf(out_stride_buf, x._ndim)
    cdef tuple stats_shape = _shape_tuple_from_buf(stats_shape_buf, x._ndim)
    cdef tuple stats_stride = _shape_tuple_from_buf(stats_stride_buf, x._ndim)

    cdef object out_ptr
    cdef object mean_ptr
    cdef object rstd_ptr
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc_large_cached(out_alloc, stream_obj)
        mean_ptr = _fast_allocator_dev0.malloc_large_cached(max(stats_numel, 1) * 4, stream_obj)
        rstd_ptr = _fast_allocator_dev0.malloc_large_cached(max(stats_numel, 1) * 4, stream_obj)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(out_alloc, stream=stream_obj)
        mean_ptr = _get_allocator_fn_ref(dev_idx).malloc(max(stats_numel, 1) * 4, stream=stream_obj)
        rstd_ptr = _get_allocator_fn_ref(dev_idx).malloc(max(stats_numel, 1) * 4, stream=stream_obj)

    cdef uintptr_t input_ptr = <uintptr_t>x._storage._untyped._device_ptr + <uintptr_t>(x._c_offset * isize)
    cdef uintptr_t weight_ptr = 0
    cdef uintptr_t bias_ptr = 0
    if has_weight:
        weight_ptr = <uintptr_t>w._storage._untyped._device_ptr + <uintptr_t>(w._c_offset * w._itemsize)
    if has_bias:
        bias_ptr = <uintptr_t>b._storage._untyped._device_ptr + <uintptr_t>(b._c_offset * b._itemsize)
    cdef uintptr_t out_raw = <uintptr_t>int(out_ptr)
    cdef uintptr_t mean_raw = <uintptr_t>int(mean_ptr)
    cdef uintptr_t rstd_raw = <uintptr_t>int(rstd_ptr)
    cdef object ws_size
    cdef object executor = 0
    cdef object runtime = None
    try:
        ws_size, executor = _ffi_layer_norm_op(
            <uintptr_t>int(_layer_norm_getws_ptr),
            <uintptr_t>int(_layer_norm_exec_ptr),
            x._shape_tuple,
            x._stride_tuple,
            out_shape,
            out_stride,
            stats_shape,
            stats_stride,
            w._shape_tuple if has_weight else (),
            w._stride_tuple if has_weight else (),
            b._shape_tuple if has_bias else (),
            b._stride_tuple if has_bias else (),
            normalized_shape,
            float(eps),
            dtype_code,
            2,
            input_ptr,
            weight_ptr,
            bias_ptr,
            out_raw,
            mean_raw,
            rstd_raw,
            stream_raw,
        )
        runtime = _run_executor(<uintptr_t>int(_layer_norm_exec_ptr), ws_size, <uintptr_t>int(executor), stream_raw, dev_idx, runtime, "aclnnLayerNorm")
        _defer_executor_handle(<uintptr_t>int(executor))
        executor = 0
    finally:
        if executor:
            _ffi_ref.destroy_executor(<uintptr_t>int(executor))

    cdef object out = _make_npu_tensor_fast_large(out_raw, out_numel, x_dtype, x_dev, out_shape, out_stride, isize, dev_idx, x._dtype_code)
    cdef object mean = _make_npu_tensor_fast_large(mean_raw, max(stats_numel, 1), _get_float32_dtype(), x_dev, stats_shape, stats_stride, 4, dev_idx, 0)
    cdef object rstd = _make_npu_tensor_fast_large(rstd_raw, max(stats_numel, 1), _get_float32_dtype(), x_dev, stats_shape, stats_stride, 4, dev_idx, 0)
    out._backward_data = {
        "mean_ptr": mean_raw,
        "rstd_ptr": rstd_raw,
        "mean_storage": (<TensorImpl>mean)._storage,
        "rstd_storage": (<TensorImpl>rstd)._storage,
        "stats_shape": stats_shape,
        "stats_stride": stats_stride,
        "normalized_shape": normalized_shape,
    }
    return out


cpdef object fast_layer_norm_backward(object grad, object saved_input, object backward_data,
                                      object normalized_shape, object weight=None, object bias=None):
    """Cython NPU LayerNorm backward wrapper around aclnnLayerNormBackward."""
    _ensure_npu_imports()
    _ensure_ffi_layer_norm()

    if not (isinstance(grad, TensorImpl) and isinstance(saved_input, TensorImpl)):
        raise ValueError("fast_layer_norm_backward expects TensorImpl grad/input")
    cdef TensorImpl g = <TensorImpl>grad
    cdef TensorImpl x = <TensorImpl>saved_input
    if g._device_type != 1 or x._device_type != 1:
        raise ValueError("fast_layer_norm_backward expects NPU tensors")
    if g._device_index != x._device_index or g._dtype_code != x._dtype_code:
        raise ValueError("fast_layer_norm_backward requires matching grad/input")
    cdef TensorImpl w
    cdef TensorImpl b
    cdef bint need_weight = weight is not None and getattr(weight, "requires_grad", False)
    cdef bint need_bias = bias is not None and getattr(bias, "requires_grad", False)
    cdef bint has_weight = weight is not None
    cdef bint has_bias = bias is not None
    if has_weight:
        if not isinstance(weight, TensorImpl):
            raise ValueError("fast_layer_norm_backward expects TensorImpl weight")
        w = <TensorImpl>weight
        if w._device_type != 1 or w._device_index != x._device_index or w._dtype_code != x._dtype_code:
            raise ValueError("fast_layer_norm_backward requires matching weight")
    if has_bias:
        if not isinstance(bias, TensorImpl):
            raise ValueError("fast_layer_norm_backward expects TensorImpl bias")
        b = <TensorImpl>bias
        if b._device_type != 1 or b._device_index != x._device_index or b._dtype_code != x._dtype_code:
            raise ValueError("fast_layer_norm_backward requires matching bias")
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    else:
        normalized_shape = tuple(normalized_shape)

    cdef int dev_idx = x._device_index
    cdef object x_dev = x._device_obj
    cdef object x_dtype = x._dtype_obj
    cdef int dtype_code = _tensor_dtype_to_acl_code(x)
    cdef int isize = x._itemsize
    cdef object stream_obj = _get_stream_obj_fast(dev_idx)
    cdef uintptr_t stream_raw = _get_stream_raw_fast(dev_idx)
    cdef tuple gi_shape = x._shape_tuple
    cdef tuple gi_stride = _contiguous_stride_tuple(gi_shape)
    cdef int64_t gi_numel = x._c_numel
    cdef object gi_ptr
    cdef object gw_ptr = None
    cdef object gb_ptr = None
    if dev_idx == 0:
        _ensure_allocator_dev0()
        gi_ptr = _fast_allocator_dev0.malloc_large_cached(max(gi_numel, 1) * isize, stream_obj)
        if need_weight:
            gw_ptr = _fast_allocator_dev0.malloc_large_cached(max(w._c_numel, 1) * isize, stream_obj)
        if need_bias:
            gb_ptr = _fast_allocator_dev0.malloc_large_cached(max(b._c_numel, 1) * isize, stream_obj)
    else:
        gi_ptr = _get_allocator_fn_ref(dev_idx).malloc(max(gi_numel, 1) * isize, stream=stream_obj)
        if need_weight:
            gw_ptr = _get_allocator_fn_ref(dev_idx).malloc(max(w._c_numel, 1) * isize, stream=stream_obj)
        if need_bias:
            gb_ptr = _get_allocator_fn_ref(dev_idx).malloc(max(b._c_numel, 1) * isize, stream=stream_obj)

    cdef uintptr_t grad_ptr = <uintptr_t>g._storage._untyped._device_ptr + <uintptr_t>(g._c_offset * g._itemsize)
    cdef uintptr_t input_ptr_bwd = <uintptr_t>x._storage._untyped._device_ptr + <uintptr_t>(x._c_offset * isize)
    cdef uintptr_t weight_ptr_bwd = 0
    cdef uintptr_t bias_ptr_bwd = 0
    if has_weight:
        weight_ptr_bwd = <uintptr_t>w._storage._untyped._device_ptr + <uintptr_t>(w._c_offset * w._itemsize)
    if has_bias:
        bias_ptr_bwd = <uintptr_t>b._storage._untyped._device_ptr + <uintptr_t>(b._c_offset * b._itemsize)
    cdef uintptr_t gi_raw = <uintptr_t>int(gi_ptr)
    cdef uintptr_t gw_raw = <uintptr_t>int(gw_ptr) if need_weight else 0
    cdef uintptr_t gb_raw = <uintptr_t>int(gb_ptr) if need_bias else 0
    cdef uintptr_t mean_raw = <uintptr_t>backward_data["mean_ptr"]
    cdef uintptr_t rstd_raw = <uintptr_t>backward_data["rstd_ptr"]
    cdef object output_mask = (True, need_weight, need_bias)
    cdef object ws_size
    cdef object executor = 0
    cdef object runtime = None
    try:
        ws_size, executor = _ffi_ref.layer_norm_backward_op(
            _layer_norm_backward_getws_ptr,
            _layer_norm_backward_exec_ptr,
            g._shape_tuple,
            g._stride_tuple,
            x._shape_tuple,
            x._stride_tuple,
            gi_shape,
            gi_stride,
            backward_data["stats_shape"],
            backward_data["stats_stride"],
            w._shape_tuple if has_weight else None,
            w._stride_tuple if has_weight else None,
            b._shape_tuple if has_bias else None,
            b._stride_tuple if has_bias else None,
            normalized_shape,
            output_mask,
            dtype_code,
            0,
            2,
            grad_ptr,
            input_ptr_bwd,
            mean_raw,
            rstd_raw,
            weight_ptr_bwd,
            bias_ptr_bwd,
            gi_raw,
            gw_raw,
            gb_raw,
            stream_raw,
        )
        runtime = _run_executor(<uintptr_t>int(_layer_norm_backward_exec_ptr), ws_size, <uintptr_t>int(executor), stream_raw, dev_idx, runtime, "aclnnLayerNormBackward")
        _defer_executor_handle(<uintptr_t>int(executor))
        executor = 0
    finally:
        if executor:
            _ffi_ref.destroy_executor(<uintptr_t>int(executor))

    cdef object grad_input = _make_npu_tensor_fast_large(gi_raw, max(gi_numel, 1), x_dtype, x_dev, gi_shape, gi_stride, isize, dev_idx, x._dtype_code)
    cdef object grad_weight = None
    cdef object grad_bias = None
    if need_weight:
        grad_weight = _make_npu_tensor_fast_large(gw_raw, max(w._c_numel, 1), x_dtype, x_dev, w._shape_tuple, w._stride_tuple, isize, dev_idx, x._dtype_code)
    if need_bias:
        grad_bias = _make_npu_tensor_fast_large(gb_raw, max(b._c_numel, 1), x_dtype, x_dev, b._shape_tuple, b._stride_tuple, isize, dev_idx, x._dtype_code)
    return grad_input, grad_weight, grad_bias


cpdef object fast_sdpa_flash_attention(object query, object key, object value, double scale_value, object attn_mask=None):
    """Fused NPU SDPA forward using ACLNN FlashAttentionScore for BNSD q/k/v."""
    _ensure_npu_imports()
    _ensure_ffi_sdpa_flash()

    if not (isinstance(query, TensorImpl) and isinstance(key, TensorImpl) and isinstance(value, TensorImpl)):
        raise ValueError("fast_sdpa_flash_attention expects base TensorImpl operands")
    if attn_mask is not None and not isinstance(attn_mask, TensorImpl):
        raise ValueError("fast_sdpa_flash_attention expects TensorImpl attention mask")
    cdef TensorImpl q = <TensorImpl>query
    cdef TensorImpl k = <TensorImpl>key
    cdef TensorImpl v = <TensorImpl>value
    cdef TensorImpl mask
    cdef bint has_mask = attn_mask is not None
    if has_mask:
        mask = <TensorImpl>attn_mask
    if q._device_type != 1 or k._device_type != 1 or v._device_type != 1:
        raise ValueError("fast_sdpa_flash_attention expects NPU tensors")
    if has_mask and mask._device_type != 1:
        raise ValueError("fast_sdpa_flash_attention expects an NPU attention mask")
    if q._dtype_code != k._dtype_code or q._dtype_code != v._dtype_code:
        raise ValueError("fast_sdpa_flash_attention requires matching dtypes")
    if has_mask and mask._dtype_code != 9:
        raise ValueError("fast_sdpa_flash_attention requires a bool attention mask")
    if q._device_index != k._device_index or q._device_index != v._device_index:
        raise ValueError("fast_sdpa_flash_attention requires tensors on the same device")
    if has_mask and mask._device_index != q._device_index:
        raise ValueError("fast_sdpa_flash_attention requires mask on the same device")
    if q._ndim != 4:
        raise ValueError("fast_sdpa_flash_attention expects rank-4 BNSD query")

    cdef int dev_idx = q._device_index
    cdef object q_dev = q._device_obj
    cdef object q_dtype = q._dtype_obj
    cdef int dtype_code = _tensor_dtype_to_acl_code(q)
    cdef int isize = q._itemsize
    cdef int64_t B = q._c_shape[0]
    cdef int64_t H = q._c_shape[1]
    cdef int64_t Sq = q._c_shape[2]
    cdef int64_t D = q._c_shape[3]
    cdef int64_t Sk = k._c_shape[2]
    if not _sdpa_flash_valid_layout(q, B, H, Sq, D):
        raise ValueError("unsupported query layout for FlashAttentionScore")
    if not _sdpa_flash_valid_layout(k, B, H, Sk, D):
        raise ValueError("unsupported key layout for FlashAttentionScore")
    if not _sdpa_flash_valid_layout(v, B, H, Sk, D):
        raise ValueError("unsupported value layout for FlashAttentionScore")
    if has_mask:
        if mask._ndim != 2 or mask._c_shape[0] != Sq or mask._c_shape[1] != Sk:
            raise ValueError("unsupported attention mask shape for FlashAttentionScore")
        if mask._c_stride[1] != 1:
            raise ValueError("unsupported attention mask layout for FlashAttentionScore")

    cdef object stream_obj = _get_stream_obj_fast(dev_idx)
    cdef uintptr_t stream_raw = _get_stream_raw_fast(dev_idx)
    cdef int64_t out_numel = q._c_numel
    cdef int64_t out_alloc = out_numel * isize
    cdef object out_ptr
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc_large_cached(out_alloc, stream_obj)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(out_alloc, stream=stream_obj)

    cdef int64_t out_shape_buf[MAX_NDIM]
    cdef int64_t out_stride_buf[MAX_NDIM]
    cdef int64_t aux_shape_buf[MAX_NDIM]
    cdef int64_t aux_stride_buf[MAX_NDIM]
    out_shape_buf[0] = B; out_shape_buf[1] = H; out_shape_buf[2] = Sq; out_shape_buf[3] = D
    if q._c_stride[0] == H * Sq * D and q._c_stride[1] == D and q._c_stride[2] == H * D and q._c_stride[3] == 1:
        # Preserve the dense BNSD view over a contiguous (B, S, H, D) buffer.
        # This lets MHA merge heads back to (B, S, E) with a view instead of
        # materializing an extra contiguous copy, while still allocating a compact
        # output storage (the layout's storage footprint equals numel).
        out_stride_buf[0] = q._c_stride[0]
        out_stride_buf[1] = q._c_stride[1]
        out_stride_buf[2] = q._c_stride[2]
        out_stride_buf[3] = q._c_stride[3]
    else:
        c_contiguous_stride(out_shape_buf, 4, out_stride_buf)
    aux_shape_buf[0] = B; aux_shape_buf[1] = H; aux_shape_buf[2] = Sq; aux_shape_buf[3] = 8
    c_contiguous_stride(aux_shape_buf, 4, aux_stride_buf)
    cdef tuple out_shape = _shape_tuple_from_buf(out_shape_buf, 4)
    cdef tuple out_stride = _shape_tuple_from_buf(out_stride_buf, 4)
    cdef tuple aux_shape = _shape_tuple_from_buf(aux_shape_buf, 4)
    cdef tuple aux_stride = _shape_tuple_from_buf(aux_stride_buf, 4)
    cdef int64_t aux_numel = B * H * Sq * 8
    cdef int64_t aux_alloc = aux_numel * 4
    cdef object softmax_max_ptr
    cdef object softmax_sum_ptr
    if dev_idx == 0:
        softmax_max_ptr = _fast_allocator_dev0.malloc_large_cached(aux_alloc, stream_obj)
        softmax_sum_ptr = _fast_allocator_dev0.malloc_large_cached(aux_alloc, stream_obj)
    else:
        softmax_max_ptr = _get_allocator_fn_ref(dev_idx).malloc(aux_alloc, stream=stream_obj)
        softmax_sum_ptr = _get_allocator_fn_ref(dev_idx).malloc(aux_alloc, stream=stream_obj)

    cdef uintptr_t q_ptr = <uintptr_t>q._storage._untyped._device_ptr + <uintptr_t>(q._c_offset * isize)
    cdef uintptr_t k_ptr = <uintptr_t>k._storage._untyped._device_ptr + <uintptr_t>(k._c_offset * isize)
    cdef uintptr_t v_ptr = <uintptr_t>v._storage._untyped._device_ptr + <uintptr_t>(v._c_offset * isize)
    cdef uintptr_t mask_ptr = 0
    if has_mask:
        mask_ptr = <uintptr_t>mask._storage._untyped._device_ptr + <uintptr_t>(mask._c_offset * mask._itemsize)
    cdef uintptr_t o_ptr = <uintptr_t>int(out_ptr)
    cdef uintptr_t sm_max_ptr = <uintptr_t>int(softmax_max_ptr)
    cdef uintptr_t sm_sum_ptr = <uintptr_t>int(softmax_sum_ptr)
    cdef uintptr_t flash_getws_raw = <uintptr_t>int(_sdpa_flash_getws_ptr)
    cdef uintptr_t flash_exec_raw = <uintptr_t>int(_sdpa_flash_exec_ptr)

    cdef void* q_t = NULL
    cdef void* k_t = NULL
    cdef void* v_t = NULL
    cdef void* mask_t = NULL
    cdef void* out_t = NULL
    cdef void* softmax_max_t = NULL
    cdef void* softmax_sum_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef bint pta_active = False
    cdef uint64_t pta_ws_size = 0
    cdef uintptr_t pta_executor = 0
    cdef int pta_lookup = 0
    cdef object workspace_ptr
    cdef object ret_obj
    cdef int ret_i
    cdef int32_t ret
    cdef object runtime = None
    cdef char layout[5]
    layout[0] = 66; layout[1] = 78; layout[2] = 83; layout[3] = 68; layout[4] = 0
    if (not has_mask) and _use_sdpa_flash_pta_cache and _pta_cache_end_fn is not None:
        pta_lookup = _ffi_pta_begin_sdpa_flash_cache_lookup_raw(
            q._shape_tuple, q._stride_tuple,
            k._shape_tuple, k._stride_tuple,
            v._shape_tuple, v._stride_tuple,
            out_shape, out_stride,
            aux_shape, aux_stride,
            dtype_code,
            q_ptr, k_ptr, v_ptr, o_ptr, sm_max_ptr, sm_sum_ptr,
            scale_value, 1.0, 2147483647, 2147483647, H, 0, 0,
            stream_raw,
            &pta_active,
            &pta_ws_size,
            &pta_executor)
        if pta_lookup and pta_executor != 0:
            try:
                runtime = _run_executor(
                    flash_exec_raw, pta_ws_size, pta_executor,
                    stream_raw, dev_idx, runtime, "FlashAttentionScore")
                return _make_npu_tensor_fast_large(o_ptr, out_numel, q_dtype, q_dev, out_shape, out_stride, isize, dev_idx, q._dtype_code), _make_npu_tensor_fast_large(sm_max_ptr, aux_numel, _get_float32_dtype(), q_dev, aux_shape, aux_stride, 4, dev_idx, 0), _make_npu_tensor_fast_large(sm_sum_ptr, aux_numel, _get_float32_dtype(), q_dev, aux_shape, aux_stride, 4, dev_idx, 0)
            finally:
                if pta_active:
                    _ffi_pta_end_cache_lookup()
                    pta_active = False

    with nogil:
        q_t = _ffi_create_tensor_raw(q._c_shape, q._c_stride, <uint64_t>4, dtype_code, 2, <void*>q_ptr)
        k_t = _ffi_create_tensor_raw(k._c_shape, k._c_stride, <uint64_t>4, dtype_code, 2, <void*>k_ptr)
        v_t = _ffi_create_tensor_raw(v._c_shape, v._c_stride, <uint64_t>4, dtype_code, 2, <void*>v_ptr)
        if has_mask:
            mask_t = _ffi_create_tensor_raw(mask._c_shape, mask._c_stride, <uint64_t>2, 12, 2, <void*>mask_ptr)
        out_t = _ffi_create_tensor_raw(out_shape_buf, out_stride_buf, <uint64_t>4, dtype_code, 2, <void*>o_ptr)
        softmax_max_t = _ffi_create_tensor_raw(aux_shape_buf, aux_stride_buf, <uint64_t>4, 0, 2, <void*>sm_max_ptr)
        softmax_sum_t = _ffi_create_tensor_raw(aux_shape_buf, aux_stride_buf, <uint64_t>4, 0, 2, <void*>sm_sum_ptr)
    if q_t == NULL or k_t == NULL or v_t == NULL or (has_mask and mask_t == NULL) or out_t == NULL or softmax_max_t == NULL or softmax_sum_t == NULL:
        if pta_active:
            _ffi_pta_end_cache_lookup()
            pta_active = False
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            ret = (<FlashAttentionScoreGetWorkspaceSize_t>flash_getws_raw)(
                q_t, k_t, v_t, NULL, NULL, NULL, mask_t, NULL,
                scale_value, 1.0, 2147483647, 2147483647, H, layout, 0, 0,
                softmax_max_t, softmax_sum_t, NULL, out_t, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"FlashAttentionScore GetWorkspaceSize failed: {ret}")
        runtime = _run_executor(flash_exec_raw, ws_size, <uintptr_t>executor, stream_raw, dev_idx, runtime, "FlashAttentionScore")
        _defer_executor_handle(<uintptr_t>executor)
        executor = NULL
    finally:
        if pta_active:
            _ffi_pta_end_cache_lookup()
        if executor != NULL:
            _ffi_ref.destroy_executor(<uintptr_t>executor)
        with nogil:
            if q_t != NULL: _ffi_destroy_tensor_raw(q_t)
            if k_t != NULL: _ffi_destroy_tensor_raw(k_t)
            if v_t != NULL: _ffi_destroy_tensor_raw(v_t)
            if mask_t != NULL: _ffi_destroy_tensor_raw(mask_t)
            if out_t != NULL: _ffi_destroy_tensor_raw(out_t)
            if softmax_max_t != NULL: _ffi_destroy_tensor_raw(softmax_max_t)
            if softmax_sum_t != NULL: _ffi_destroy_tensor_raw(softmax_sum_t)

    cdef object out = _make_npu_tensor_fast_large(o_ptr, out_numel, q_dtype, q_dev, out_shape, out_stride, isize, dev_idx, q._dtype_code)
    cdef object softmax_max = _make_npu_tensor_fast_large(sm_max_ptr, aux_numel, _get_float32_dtype(), q_dev, aux_shape, aux_stride, 4, dev_idx, 0)
    cdef object softmax_sum = _make_npu_tensor_fast_large(sm_sum_ptr, aux_numel, _get_float32_dtype(), q_dev, aux_shape, aux_stride, 4, dev_idx, 0)
    return out, softmax_max, softmax_sum


cpdef object fast_sdpa_flash_attention_backward(object grad_out, object query, object key, object value,
                                                object output, object softmax_max, object softmax_sum,
                                                double scale_value, object attn_mask=None):
    """Fused NPU SDPA backward using ACLNN FlashAttentionScoreGrad."""
    _ensure_npu_imports()
    _ensure_ffi_sdpa_flash()

    if not (
        isinstance(grad_out, TensorImpl)
        and isinstance(query, TensorImpl)
        and isinstance(key, TensorImpl)
        and isinstance(value, TensorImpl)
        and isinstance(output, TensorImpl)
        and isinstance(softmax_max, TensorImpl)
        and isinstance(softmax_sum, TensorImpl)
    ):
        raise ValueError("fast_sdpa_flash_attention_backward expects base TensorImpl operands")
    if attn_mask is not None and not isinstance(attn_mask, TensorImpl):
        raise ValueError("fast_sdpa_flash_attention_backward expects TensorImpl attention mask")
    cdef TensorImpl g = <TensorImpl>grad_out
    cdef TensorImpl q = <TensorImpl>query
    cdef TensorImpl k = <TensorImpl>key
    cdef TensorImpl v = <TensorImpl>value
    cdef TensorImpl o = <TensorImpl>output
    cdef TensorImpl sm_max = <TensorImpl>softmax_max
    cdef TensorImpl sm_sum = <TensorImpl>softmax_sum
    cdef TensorImpl mask
    cdef bint has_mask = attn_mask is not None
    if has_mask:
        mask = <TensorImpl>attn_mask

    if q._device_type != 1 or k._device_type != 1 or v._device_type != 1 or g._device_type != 1:
        raise ValueError("fast_sdpa_flash_attention_backward expects NPU tensors")
    if has_mask and mask._device_type != 1:
        raise ValueError("fast_sdpa_flash_attention_backward expects an NPU attention mask")
    if q._dtype_code != k._dtype_code or q._dtype_code != v._dtype_code or q._dtype_code != g._dtype_code:
        raise ValueError("fast_sdpa_flash_attention_backward requires matching q/k/v/grad dtypes")
    if has_mask and mask._dtype_code != 9:
        raise ValueError("fast_sdpa_flash_attention_backward requires a bool attention mask")
    if q._device_index != k._device_index or q._device_index != v._device_index or q._device_index != g._device_index:
        raise ValueError("fast_sdpa_flash_attention_backward requires tensors on the same device")
    if has_mask and mask._device_index != q._device_index:
        raise ValueError("fast_sdpa_flash_attention_backward requires mask on the same device")
    cdef int dev_idx = q._device_index
    cdef object q_dev = q._device_obj
    cdef object q_dtype = q._dtype_obj
    cdef int dtype_code = _tensor_dtype_to_acl_code(q)
    cdef int isize = q._itemsize
    cdef int64_t B = q._c_shape[0]
    cdef int64_t H = q._c_shape[1]
    cdef int64_t Sq = q._c_shape[2]
    cdef int64_t D = q._c_shape[3]
    cdef int64_t Sk = k._c_shape[2]
    if not _sdpa_flash_valid_layout(q, B, H, Sq, D):
        raise ValueError("unsupported query layout for FlashAttentionScoreGrad")
    if not _sdpa_flash_valid_layout(k, B, H, Sk, D):
        raise ValueError("unsupported key layout for FlashAttentionScoreGrad")
    if not _sdpa_flash_valid_layout(v, B, H, Sk, D):
        raise ValueError("unsupported value layout for FlashAttentionScoreGrad")
    if not _sdpa_flash_valid_grad_layout(g, B, H, Sq, D):
        raise ValueError("unsupported grad layout for FlashAttentionScoreGrad")
    if has_mask:
        if mask._ndim != 2 or mask._c_shape[0] != Sq or mask._c_shape[1] != Sk:
            raise ValueError("unsupported attention mask shape for FlashAttentionScoreGrad")
        if mask._c_stride[1] != 1:
            raise ValueError("unsupported attention mask layout for FlashAttentionScoreGrad")

    cdef object stream_obj = _get_stream_obj_fast(dev_idx)
    cdef uintptr_t stream_raw = _get_stream_raw_fast(dev_idx)
    cdef int64_t q_numel = q._c_numel
    cdef int64_t kv_numel = k._c_numel
    cdef int64_t q_alloc = q_numel * isize
    cdef int64_t kv_alloc = kv_numel * isize
    cdef object dq_ptr
    cdef object dk_ptr
    cdef object dv_ptr
    if dev_idx == 0:
        _ensure_allocator_dev0()
        dq_ptr = _fast_allocator_dev0.malloc_large_cached(q_alloc, stream_obj)
        dk_ptr = _fast_allocator_dev0.malloc_large_cached(kv_alloc, stream_obj)
        dv_ptr = _fast_allocator_dev0.malloc_large_cached(kv_alloc, stream_obj)
    else:
        dq_ptr = _get_allocator_fn_ref(dev_idx).malloc(q_alloc, stream=stream_obj)
        dk_ptr = _get_allocator_fn_ref(dev_idx).malloc(kv_alloc, stream=stream_obj)
        dv_ptr = _get_allocator_fn_ref(dev_idx).malloc(kv_alloc, stream=stream_obj)

    cdef int64_t q_shape_buf[MAX_NDIM]
    cdef int64_t q_stride_buf[MAX_NDIM]
    cdef int64_t kv_shape_buf[MAX_NDIM]
    cdef int64_t kv_stride_buf[MAX_NDIM]
    cdef int64_t softmax_empty_shape_buf[1]
    cdef int64_t softmax_empty_stride_buf[1]
    q_shape_buf[0] = B; q_shape_buf[1] = H; q_shape_buf[2] = Sq; q_shape_buf[3] = D
    c_contiguous_stride(q_shape_buf, 4, q_stride_buf)
    kv_shape_buf[0] = B; kv_shape_buf[1] = H; kv_shape_buf[2] = Sk; kv_shape_buf[3] = D
    c_contiguous_stride(kv_shape_buf, 4, kv_stride_buf)
    softmax_empty_shape_buf[0] = 0
    softmax_empty_stride_buf[0] = 1
    cdef tuple q_shape = _shape_tuple_from_buf(q_shape_buf, 4)
    cdef tuple q_stride = _shape_tuple_from_buf(q_stride_buf, 4)
    cdef tuple kv_shape = _shape_tuple_from_buf(kv_shape_buf, 4)
    cdef tuple kv_stride = _shape_tuple_from_buf(kv_stride_buf, 4)

    cdef uintptr_t g_ptr = <uintptr_t>g._storage._untyped._device_ptr + <uintptr_t>(g._c_offset * isize)
    cdef uintptr_t q_ptr = <uintptr_t>q._storage._untyped._device_ptr + <uintptr_t>(q._c_offset * isize)
    cdef uintptr_t k_ptr = <uintptr_t>k._storage._untyped._device_ptr + <uintptr_t>(k._c_offset * isize)
    cdef uintptr_t v_ptr = <uintptr_t>v._storage._untyped._device_ptr + <uintptr_t>(v._c_offset * isize)
    cdef uintptr_t o_ptr = <uintptr_t>o._storage._untyped._device_ptr + <uintptr_t>(o._c_offset * isize)
    cdef uintptr_t sm_max_ptr = <uintptr_t>sm_max._storage._untyped._device_ptr + <uintptr_t>(sm_max._c_offset * sm_max._itemsize)
    cdef uintptr_t sm_sum_ptr = <uintptr_t>sm_sum._storage._untyped._device_ptr + <uintptr_t>(sm_sum._c_offset * sm_sum._itemsize)
    cdef uintptr_t mask_ptr = 0
    if has_mask:
        mask_ptr = <uintptr_t>mask._storage._untyped._device_ptr + <uintptr_t>(mask._c_offset * mask._itemsize)
    cdef uintptr_t dq_raw = <uintptr_t>int(dq_ptr)
    cdef uintptr_t dk_raw = <uintptr_t>int(dk_ptr)
    cdef uintptr_t dv_raw = <uintptr_t>int(dv_ptr)
    cdef uintptr_t flash_grad_getws_raw
    cdef uintptr_t flash_grad_exec_raw
    cdef bint use_grad_v2 = _use_sdpa_flash_grad_v2 and _sdpa_flash_grad_v2_getws_ptr is not None
    cdef int64_t pse_type = 1
    if use_grad_v2:
        flash_grad_getws_raw = <uintptr_t>int(_sdpa_flash_grad_v2_getws_ptr)
        flash_grad_exec_raw = <uintptr_t>int(_sdpa_flash_grad_v2_exec_ptr)
    else:
        flash_grad_getws_raw = <uintptr_t>int(_sdpa_flash_grad_getws_ptr)
        flash_grad_exec_raw = <uintptr_t>int(_sdpa_flash_grad_exec_ptr)

    cdef void* q_t = NULL
    cdef void* k_t = NULL
    cdef void* v_t = NULL
    cdef void* g_t = NULL
    cdef void* o_t = NULL
    cdef void* softmax_max_t = NULL
    cdef void* softmax_sum_t = NULL
    cdef void* softmax_in_t = NULL
    cdef void* mask_t = NULL
    cdef void* dq_t = NULL
    cdef void* dk_t = NULL
    cdef void* dv_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef bint pta_active = False
    cdef uint64_t pta_ws_size = 0
    cdef uintptr_t pta_executor = 0
    cdef int pta_lookup = 0
    cdef object workspace_ptr
    cdef object ret_obj
    cdef int ret_i
    cdef int32_t ret
    cdef object runtime = None
    cdef char layout[5]
    layout[0] = 66; layout[1] = 78; layout[2] = 83; layout[3] = 68; layout[4] = 0
    if (not has_mask) and _use_sdpa_flash_grad_pta_cache and _pta_cache_end_fn is not None:
        if use_grad_v2:
            pta_lookup = _ffi_pta_begin_sdpa_flash_grad_v2_cache_lookup_raw(
                q._shape_tuple, q._stride_tuple,
                k._shape_tuple, k._stride_tuple,
                v._shape_tuple, v._stride_tuple,
                g._shape_tuple, g._stride_tuple,
                o._shape_tuple, o._stride_tuple,
                sm_max._shape_tuple, sm_max._stride_tuple,
                q_shape, q_stride,
                kv_shape, kv_stride,
                kv_shape, kv_stride,
                dtype_code,
                q_ptr, k_ptr, v_ptr, g_ptr, o_ptr, sm_max_ptr, sm_sum_ptr,
                dq_raw, dk_raw, dv_raw,
                scale_value, 1.0, 2147483647, 2147483647, H, 0, 0, pse_type,
                stream_raw,
                &pta_active,
                &pta_ws_size,
                &pta_executor)
        else:
            pta_lookup = _ffi_pta_begin_sdpa_flash_grad_cache_lookup_raw(
                q._shape_tuple, q._stride_tuple,
                k._shape_tuple, k._stride_tuple,
                v._shape_tuple, v._stride_tuple,
                g._shape_tuple, g._stride_tuple,
                o._shape_tuple, o._stride_tuple,
                sm_max._shape_tuple, sm_max._stride_tuple,
                q_shape, q_stride,
                kv_shape, kv_stride,
                kv_shape, kv_stride,
                dtype_code,
                q_ptr, k_ptr, v_ptr, g_ptr, o_ptr, sm_max_ptr, sm_sum_ptr,
                dq_raw, dk_raw, dv_raw,
                scale_value, 1.0, 2147483647, 2147483647, H, 0, 0,
                stream_raw,
                &pta_active,
                &pta_ws_size,
                &pta_executor)
        if pta_lookup and pta_executor != 0:
            try:
                runtime = _run_executor(
                    flash_grad_exec_raw, pta_ws_size, pta_executor,
                    stream_raw, dev_idx, runtime, "FlashAttentionScoreGrad")
                return _make_npu_tensor_fast_large(dq_raw, q_numel, q_dtype, q_dev, q_shape, q_stride, isize, dev_idx, q._dtype_code), _make_npu_tensor_fast_large(dk_raw, kv_numel, q_dtype, q_dev, kv_shape, kv_stride, isize, dev_idx, q._dtype_code), _make_npu_tensor_fast_large(dv_raw, kv_numel, q_dtype, q_dev, kv_shape, kv_stride, isize, dev_idx, q._dtype_code)
            finally:
                if pta_active:
                    _ffi_pta_end_cache_lookup()
                    pta_active = False

    with nogil:
        q_t = _ffi_create_tensor_raw(q._c_shape, q._c_stride, <uint64_t>4, dtype_code, 2, <void*>q_ptr)
        k_t = _ffi_create_tensor_raw(k._c_shape, k._c_stride, <uint64_t>4, dtype_code, 2, <void*>k_ptr)
        v_t = _ffi_create_tensor_raw(v._c_shape, v._c_stride, <uint64_t>4, dtype_code, 2, <void*>v_ptr)
        g_t = _ffi_create_tensor_raw(g._c_shape, g._c_stride, <uint64_t>4, dtype_code, 2, <void*>g_ptr)
        o_t = _ffi_create_tensor_raw(o._c_shape, o._c_stride, <uint64_t>4, dtype_code, 2, <void*>o_ptr)
        softmax_max_t = _ffi_create_tensor_raw(sm_max._c_shape, sm_max._c_stride, <uint64_t>4, 0, 2, <void*>sm_max_ptr)
        softmax_sum_t = _ffi_create_tensor_raw(sm_sum._c_shape, sm_sum._c_stride, <uint64_t>4, 0, 2, <void*>sm_sum_ptr)
        if has_mask:
            mask_t = _ffi_create_tensor_raw(mask._c_shape, mask._c_stride, <uint64_t>2, 12, 2, <void*>mask_ptr)
        if use_grad_v2:
            softmax_in_t = _ffi_create_tensor_raw(softmax_empty_shape_buf, softmax_empty_stride_buf, <uint64_t>1, dtype_code, 2, <void*>q_ptr)
        dq_t = _ffi_create_tensor_raw(q_shape_buf, q_stride_buf, <uint64_t>4, dtype_code, 2, <void*>dq_raw)
        dk_t = _ffi_create_tensor_raw(kv_shape_buf, kv_stride_buf, <uint64_t>4, dtype_code, 2, <void*>dk_raw)
        dv_t = _ffi_create_tensor_raw(kv_shape_buf, kv_stride_buf, <uint64_t>4, dtype_code, 2, <void*>dv_raw)
    if q_t == NULL or k_t == NULL or v_t == NULL or g_t == NULL or o_t == NULL or softmax_max_t == NULL or softmax_sum_t == NULL or (has_mask and mask_t == NULL) or dq_t == NULL or dk_t == NULL or dv_t == NULL or (use_grad_v2 and softmax_in_t == NULL):
        if pta_active:
            _ffi_pta_end_cache_lookup()
            pta_active = False
        raise RuntimeError("aclCreateTensor returned null")
    try:
        with nogil:
            if use_grad_v2:
                ret = (<FlashAttentionScoreGradV2GetWorkspaceSize_t>flash_grad_getws_raw)(
                    q_t, k_t, v_t, g_t, NULL, NULL, NULL, mask_t,
                    softmax_max_t, softmax_sum_t, softmax_in_t, o_t, NULL, NULL, NULL,
                    scale_value, 1.0, 2147483647, 2147483647, H, layout, 0, 0, pse_type,
                    dq_t, dk_t, dv_t, NULL, &ws_size, &executor)
            else:
                ret = (<FlashAttentionScoreGradGetWorkspaceSize_t>flash_grad_getws_raw)(
                    q_t, k_t, v_t, g_t, NULL, NULL, NULL, mask_t,
                    softmax_max_t, softmax_sum_t, NULL, o_t, NULL,
                    scale_value, 1.0, 2147483647, 2147483647, H, layout, 0, 0,
                    dq_t, dk_t, dv_t, NULL, &ws_size, &executor)
        if ret != 0:
            raise RuntimeError(f"FlashAttentionScoreGrad GetWorkspaceSize failed: {ret}")
        runtime = _run_executor(flash_grad_exec_raw, ws_size, <uintptr_t>executor, stream_raw, dev_idx, runtime, "FlashAttentionScoreGrad")
        _defer_executor_handle(<uintptr_t>executor)
        executor = NULL
    finally:
        if pta_active:
            _ffi_pta_end_cache_lookup()
        if executor != NULL:
            _ffi_ref.destroy_executor(<uintptr_t>executor)
        with nogil:
            if q_t != NULL: _ffi_destroy_tensor_raw(q_t)
            if k_t != NULL: _ffi_destroy_tensor_raw(k_t)
            if v_t != NULL: _ffi_destroy_tensor_raw(v_t)
            if g_t != NULL: _ffi_destroy_tensor_raw(g_t)
            if o_t != NULL: _ffi_destroy_tensor_raw(o_t)
            if softmax_max_t != NULL: _ffi_destroy_tensor_raw(softmax_max_t)
            if softmax_sum_t != NULL: _ffi_destroy_tensor_raw(softmax_sum_t)
            if softmax_in_t != NULL: _ffi_destroy_tensor_raw(softmax_in_t)
            if mask_t != NULL: _ffi_destroy_tensor_raw(mask_t)
            if dq_t != NULL: _ffi_destroy_tensor_raw(dq_t)
            if dk_t != NULL: _ffi_destroy_tensor_raw(dk_t)
            if dv_t != NULL: _ffi_destroy_tensor_raw(dv_t)

    cdef object dq = _make_npu_tensor_fast_large(dq_raw, q_numel, q_dtype, q_dev, q_shape, q_stride, isize, dev_idx, q._dtype_code)
    cdef object dk = _make_npu_tensor_fast_large(dk_raw, kv_numel, q_dtype, q_dev, kv_shape, kv_stride, isize, dev_idx, q._dtype_code)
    cdef object dv = _make_npu_tensor_fast_large(dv_raw, kv_numel, q_dtype, q_dev, kv_shape, kv_stride, isize, dev_idx, q._dtype_code)
    return dq, dk, dv


cpdef object fast_packed_qkv_projection_forward(object packed, int64_t embed,
                                                int64_t heads, bint batch_first):
    """Split packed self-attention projection into compact BNSD Q/K/V tensors."""
    _ensure_npu_imports()
    _ensure_ffi_binary()

    if not isinstance(packed, TensorImpl):
        return None
    cdef TensorImpl p = <TensorImpl>packed
    if p._device_type != 1:
        return None
    if p._ndim != 3:
        raise ValueError("fast_packed_qkv_projection_forward expects rank-3 packed input")
    if heads <= 0 or embed <= 0 or embed % heads != 0:
        raise ValueError("invalid packed QKV embed/head metadata")
    if p._c_shape[2] != 3 * embed:
        raise ValueError("packed QKV last dimension must equal 3 * embed_dim")

    cdef int64_t B
    cdef int64_t S
    cdef int64_t D = embed // heads
    cdef int64_t raw_numel
    cdef int64_t alloc_size
    cdef int dev_idx = p._device_index
    cdef int isize = p._itemsize
    cdef int dtype_code = _tensor_dtype_to_acl_code(p)
    cdef object p_dev = p._device_obj
    cdef object p_dtype = p._dtype_obj
    cdef object raw_shape
    cdef object raw_stride
    cdef object bnsd_shape
    cdef object bnsd_stride
    cdef object stream_obj = _get_stream_obj_fast(dev_idx)
    cdef uintptr_t stream_raw = _get_stream_raw_fast(dev_idx)
    cdef object q_ptr
    cdef object k_ptr
    cdef object v_ptr
    cdef uintptr_t q_raw
    cdef uintptr_t k_raw
    cdef uintptr_t v_raw
    cdef uintptr_t p_raw
    cdef object split_getws_ptr
    cdef object split_exec_ptr
    cdef object ws_size
    cdef object executor = 0
    cdef object workspace_ptr
    cdef object ret
    cdef object runtime = None
    cdef object split_sizes
    cdef object out_ptrs
    cdef object out_shapes
    cdef object out_strides
    cdef object pta_state
    cdef bint pta_active = False
    cdef uint64_t ws_size_raw = 0
    cdef uintptr_t executor_raw = 0
    cdef int ret_i

    if batch_first:
        B = p._c_shape[0]
        S = p._c_shape[1]
        raw_shape = (B, S, embed)
        raw_stride = (S * embed, embed, 1)
        bnsd_shape = (B, heads, S, D)
        bnsd_stride = (S * embed, D, embed, 1)
    else:
        S = p._c_shape[0]
        B = p._c_shape[1]
        raw_shape = (S, B, embed)
        raw_stride = (B * embed, embed, 1)
        bnsd_shape = (B, heads, S, D)
        bnsd_stride = (embed, D, B * embed, 1)

    raw_numel = B * S * embed
    alloc_size = raw_numel * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        q_ptr = _fast_allocator_dev0.malloc_large_cached(alloc_size, stream_obj)
        k_ptr = _fast_allocator_dev0.malloc_large_cached(alloc_size, stream_obj)
        v_ptr = _fast_allocator_dev0.malloc_large_cached(alloc_size, stream_obj)
    else:
        q_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size, stream=stream_obj)
        k_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size, stream=stream_obj)
        v_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size, stream=stream_obj)

    q_raw = <uintptr_t>int(q_ptr)
    k_raw = <uintptr_t>int(k_ptr)
    v_raw = <uintptr_t>int(v_ptr)
    p_raw = <uintptr_t>p._storage._untyped._device_ptr + <uintptr_t>(p._c_offset * isize)
    split_getws_ptr, split_exec_ptr = _ffi_ref.resolve_op("SplitWithSize")
    split_sizes = (embed, embed, embed)
    out_ptrs = (q_raw, k_raw, v_raw)
    out_shapes = (raw_shape, raw_shape, raw_shape)
    out_strides = (raw_stride, raw_stride, raw_stride)
    try:
        pta_state = _ffi_pta_begin_split_with_size_cache_lookup(
            p._shape_tuple, p._stride_tuple,
            split_sizes,
            out_ptrs,
            out_shapes,
            out_strides,
            2,
            dtype_code,
            p_raw,
            stream_raw,
        )
        if pta_state is not None:
            pta_active = bool(pta_state[0])
            ws_size_raw = <uint64_t>int(pta_state[1])
            executor_raw = <uintptr_t>int(pta_state[2])
            if executor_raw != 0:
                runtime = _run_executor(
                    <uintptr_t>int(split_exec_ptr), ws_size_raw, executor_raw,
                    stream_raw, dev_idx, runtime, "aclnnSplitWithSize")
                return (
                    _make_npu_tensor_fast_large(q_raw, raw_numel, p_dtype, p_dev, bnsd_shape, bnsd_stride, isize, dev_idx, p._dtype_code),
                    _make_npu_tensor_fast_large(k_raw, raw_numel, p_dtype, p_dev, bnsd_shape, bnsd_stride, isize, dev_idx, p._dtype_code),
                    _make_npu_tensor_fast_large(v_raw, raw_numel, p_dtype, p_dev, bnsd_shape, bnsd_stride, isize, dev_idx, p._dtype_code),
                )

        ws_size, executor = _ffi_split_with_size_op(
            split_getws_ptr, split_exec_ptr,
            p._shape_tuple, p._stride_tuple,
            split_sizes,
            out_ptrs,
            out_shapes,
            out_strides,
            2,
            dtype_code, 2,
            p_raw,
            stream_raw)
        if ws_size:
            workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            try:
                ret_i = _ffi_execute(split_exec_ptr, <uintptr_t>int(workspace_ptr), ws_size, executor, stream_raw)
                if ret_i != 0:
                    raise RuntimeError(f"aclnnSplitWithSize execute failed: {ret_i}")
            finally:
                if runtime is None:
                    runtime = _get_runtime_fast(dev_idx)
                runtime.defer_raw_free(workspace_ptr)
    except Exception:
        _get_allocator_fn_ref(dev_idx).free(<int64_t>q_raw, stream=stream_obj)
        _get_allocator_fn_ref(dev_idx).free(<int64_t>k_raw, stream=stream_obj)
        _get_allocator_fn_ref(dev_idx).free(<int64_t>v_raw, stream=stream_obj)
        raise
    finally:
        if pta_active:
            _ffi_pta_end_cache_lookup()
        if executor:
            _defer_executor_fn(executor)

    return (
        _make_npu_tensor_fast_large(q_raw, raw_numel, p_dtype, p_dev, bnsd_shape, bnsd_stride, isize, dev_idx, p._dtype_code),
        _make_npu_tensor_fast_large(k_raw, raw_numel, p_dtype, p_dev, bnsd_shape, bnsd_stride, isize, dev_idx, p._dtype_code),
        _make_npu_tensor_fast_large(v_raw, raw_numel, p_dtype, p_dev, bnsd_shape, bnsd_stride, isize, dev_idx, p._dtype_code),
    )


cpdef object fast_packed_qkv_projection_backward(object grad_q, object grad_k, object grad_v,
                                                 object input_shape, int64_t embed,
                                                 int64_t heads, bint batch_first):
    """Pack native BNSD SDPA grads back into the packed QKV projection gradient."""
    _ensure_npu_imports()
    _ensure_ffi_binary()

    if not (isinstance(grad_q, TensorImpl) and isinstance(grad_k, TensorImpl) and isinstance(grad_v, TensorImpl)):
        raise ValueError("fast_packed_qkv_projection_backward expects base TensorImpl grads")
    cdef TensorImpl q = <TensorImpl>grad_q
    cdef TensorImpl k = <TensorImpl>grad_k
    cdef TensorImpl v = <TensorImpl>grad_v
    cdef int dev_idx = q._device_index
    cdef object q_dev = q._device_obj
    cdef object q_dtype = q._dtype_obj
    cdef int dtype_code = _tensor_dtype_to_acl_code(q)
    cdef int isize = q._itemsize
    cdef int64_t B
    cdef int64_t H
    cdef int64_t S
    cdef int64_t D
    cdef int64_t packed_width
    cdef int64_t out_numel
    cdef int64_t alloc_size
    cdef tuple out_shape
    cdef tuple out_stride
    cdef tuple source_shape
    cdef tuple source_stride
    cdef tuple stack_shape
    cdef tuple stack_stride
    cdef object stream_obj
    cdef uintptr_t stream_raw
    cdef object out_ptr
    cdef uintptr_t out_raw
    cdef uintptr_t q_ptr
    cdef uintptr_t k_ptr
    cdef uintptr_t v_ptr
    cdef object stack_getws_ptr
    cdef object stack_exec_ptr
    cdef object ws_size
    cdef object executor = 0
    cdef object workspace_ptr
    cdef object ret
    cdef int ret_i
    cdef object runtime = None
    cdef object tensor_ptrs
    cdef object source_shapes
    cdef object source_strides
    cdef object pta_state
    cdef bint pta_active = False
    cdef uint64_t ws_size_raw = 0
    cdef uintptr_t executor_raw = 0

    if q._device_type != 1 or k._device_type != 1 or v._device_type != 1:
        raise ValueError("fast_packed_qkv_projection_backward expects NPU tensors")
    if q._device_index != k._device_index or q._device_index != v._device_index:
        raise ValueError("fast_packed_qkv_projection_backward requires tensors on the same device")
    if q._dtype_code != k._dtype_code or q._dtype_code != v._dtype_code:
        raise ValueError("fast_packed_qkv_projection_backward requires matching dtypes")
    if q._ndim != 4 or k._ndim != 4 or v._ndim != 4:
        raise ValueError("fast_packed_qkv_projection_backward expects rank-4 BNSD grads")

    B = q._c_shape[0]
    H = q._c_shape[1]
    S = q._c_shape[2]
    D = q._c_shape[3]
    if k._c_shape[0] != B or k._c_shape[1] != H or k._c_shape[2] != S or k._c_shape[3] != D:
        raise ValueError("fast_packed_qkv_projection_backward requires matching q/k shapes")
    if v._c_shape[0] != B or v._c_shape[1] != H or v._c_shape[2] != S or v._c_shape[3] != D:
        raise ValueError("fast_packed_qkv_projection_backward requires matching q/v shapes")
    if heads != H or embed != H * D:
        raise ValueError("fast_packed_qkv_projection_backward metadata does not match grad shape")
    if not isinstance(input_shape, tuple) or len(input_shape) != 3:
        raise ValueError("fast_packed_qkv_projection_backward expects packed rank-3 input shape")

    packed_width = 3 * embed
    if batch_first:
        if input_shape[0] != B or input_shape[1] != S or input_shape[2] != packed_width:
            raise ValueError("fast_packed_qkv_projection_backward batch-first shape mismatch")
        source_shape = (B, S, H, D)
        source_stride = (q._c_stride[0], q._c_stride[2], q._c_stride[1], q._c_stride[3])
        stack_shape = (B, S, 3, H, D)
        stack_stride = (S * 3 * H * D, 3 * H * D, H * D, D, 1)
        out_shape = (B, S, packed_width)
        out_stride = (S * packed_width, packed_width, 1)
    else:
        if input_shape[0] != S or input_shape[1] != B or input_shape[2] != packed_width:
            raise ValueError("fast_packed_qkv_projection_backward sequence-first shape mismatch")
        source_shape = (S, B, H, D)
        source_stride = (q._c_stride[2], q._c_stride[0], q._c_stride[1], q._c_stride[3])
        stack_shape = (S, B, 3, H, D)
        stack_stride = (B * 3 * H * D, 3 * H * D, H * D, D, 1)
        out_shape = (S, B, packed_width)
        out_stride = (B * packed_width, packed_width, 1)

    out_numel = B * S * packed_width
    alloc_size = out_numel * isize
    stream_obj = _get_stream_obj_fast(dev_idx)
    stream_raw = _get_stream_raw_fast(dev_idx)
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc_large_cached(alloc_size, stream_obj)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size, stream=stream_obj)
    out_raw = <uintptr_t>int(out_ptr)
    q_ptr = <uintptr_t>q._storage._untyped._device_ptr + <uintptr_t>(q._c_offset * isize)
    k_ptr = <uintptr_t>k._storage._untyped._device_ptr + <uintptr_t>(k._c_offset * isize)
    v_ptr = <uintptr_t>v._storage._untyped._device_ptr + <uintptr_t>(v._c_offset * isize)
    stack_getws_ptr, stack_exec_ptr = _ffi_ref.resolve_op("Stack")
    tensor_ptrs = (q_ptr, k_ptr, v_ptr)
    source_shapes = (source_shape, source_shape, source_shape)
    source_strides = (source_stride, source_stride, source_stride)
    try:
        pta_state = _ffi_pta_begin_stack_cache_lookup(
            tensor_ptrs,
            source_shapes,
            source_strides,
            2,
            stack_shape, stack_stride,
            dtype_code,
            out_raw,
            stream_raw,
        )
        if pta_state is not None:
            pta_active = bool(pta_state[0])
            ws_size_raw = <uint64_t>int(pta_state[1])
            executor_raw = <uintptr_t>int(pta_state[2])
            if executor_raw != 0:
                runtime = _run_executor(
                    <uintptr_t>int(stack_exec_ptr), ws_size_raw, executor_raw,
                    stream_raw, dev_idx, runtime, "aclnnStack")
                return _make_npu_tensor_fast_large(
                    out_raw, out_numel, q_dtype, q_dev, out_shape, out_stride,
                    isize, dev_idx, q._dtype_code)

        ws_size, executor = _ffi_ref.tensor_list_axis_op(
            stack_getws_ptr, stack_exec_ptr,
            tensor_ptrs,
            source_shapes,
            source_strides,
            (q_dtype, q_dtype, q_dtype),
            2,
            stack_shape, stack_stride,
            dtype_code, 2,
            out_raw,
            stream_raw)
        if ws_size:
            workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            try:
                ret_i = _ffi_execute(stack_exec_ptr, <uintptr_t>int(workspace_ptr), ws_size, executor, stream_raw)
                if ret_i != 0:
                    raise RuntimeError(f"aclnnStack execute failed: {ret_i}")
            finally:
                if runtime is None:
                    runtime = _get_runtime_fast(dev_idx)
                runtime.defer_raw_free(workspace_ptr)
    except Exception:
        _get_allocator_fn_ref(dev_idx).free(<int64_t>out_raw, stream=stream_obj)
        raise
    finally:
        if pta_active:
            _ffi_pta_end_cache_lookup()
        if executor:
            _defer_executor_fn(executor)
    return _make_npu_tensor_fast_large(out_raw, out_numel, q_dtype, q_dev, out_shape, out_stride, isize, dev_idx, q._dtype_code)



cpdef object fast_sdpa_flash_attention_backward_packed_qkv(object grad_out, object query, object key, object value,
                                                           object output, object softmax_max, object softmax_sum,
                                                           double scale_value, object input_shape,
                                                           int64_t embed, int64_t heads, bint batch_first):
    """Fused NPU SDPA backward writing dq/dk/dv directly into one packed-QKV grad buffer."""
    cdef TensorImpl g
    cdef TensorImpl q
    cdef TensorImpl k
    cdef TensorImpl v
    cdef TensorImpl o
    cdef TensorImpl sm_max
    cdef TensorImpl sm_sum
    cdef int dev_idx
    cdef object q_dev
    cdef object q_dtype
    cdef int dtype_code
    cdef int isize
    cdef int64_t B
    cdef int64_t H
    cdef int64_t Sq
    cdef int64_t D
    cdef int64_t Sk
    cdef int64_t packed_width
    cdef int64_t packed_numel
    cdef int64_t packed_alloc
    cdef object packed_shape
    cdef object packed_stride
    cdef object grad_shape
    cdef object grad_stride
    cdef object stream_obj
    cdef uintptr_t stream_raw
    cdef object packed_ptr
    cdef uintptr_t packed_raw
    cdef uintptr_t dq_raw
    cdef uintptr_t dk_raw
    cdef uintptr_t dv_raw
    cdef uintptr_t g_ptr
    cdef uintptr_t q_ptr
    cdef uintptr_t k_ptr
    cdef uintptr_t v_ptr
    cdef uintptr_t o_ptr
    cdef uintptr_t sm_max_ptr
    cdef uintptr_t sm_sum_ptr
    cdef uintptr_t flash_grad_getws_raw
    cdef uintptr_t flash_grad_exec_raw
    cdef bint use_grad_v2
    cdef int64_t pse_type = 1
    cdef int64_t q_shape_buf[MAX_NDIM]
    cdef int64_t q_stride_buf[MAX_NDIM]
    cdef int64_t packed_storage_shape_buf[3]
    cdef int64_t softmax_empty_shape_buf[1]
    cdef int64_t softmax_empty_stride_buf[1]
    cdef void* q_t = NULL
    cdef void* k_t = NULL
    cdef void* v_t = NULL
    cdef void* g_t = NULL
    cdef void* o_t = NULL
    cdef void* softmax_max_t = NULL
    cdef void* softmax_sum_t = NULL
    cdef void* softmax_in_t = NULL
    cdef void* dq_t = NULL
    cdef void* dk_t = NULL
    cdef void* dv_t = NULL
    cdef uint64_t ws_size = 0
    cdef void* executor = NULL
    cdef bint pta_active = False
    cdef uint64_t pta_ws_size = 0
    cdef uintptr_t pta_executor = 0
    cdef int pta_lookup = 0
    cdef int32_t ret
    cdef object runtime = None
    cdef char layout[5]
    cdef object packed_grad = None
    cdef object packed_storage = None
    cdef object grad_q = None
    cdef object grad_k = None
    cdef object grad_v = None
    cdef bint packed_wrapped = False

    _ensure_npu_imports()
    _ensure_ffi_sdpa_flash()

    if not (
        isinstance(grad_out, TensorImpl)
        and isinstance(query, TensorImpl)
        and isinstance(key, TensorImpl)
        and isinstance(value, TensorImpl)
        and isinstance(output, TensorImpl)
        and isinstance(softmax_max, TensorImpl)
        and isinstance(softmax_sum, TensorImpl)
    ):
        raise ValueError("fast_sdpa_flash_attention_backward_packed_qkv expects base TensorImpl operands")
    if not isinstance(input_shape, tuple) or len(input_shape) != 3:
        raise ValueError("fast_sdpa_flash_attention_backward_packed_qkv expects packed rank-3 input shape")

    g = <TensorImpl>grad_out
    q = <TensorImpl>query
    k = <TensorImpl>key
    v = <TensorImpl>value
    o = <TensorImpl>output
    sm_max = <TensorImpl>softmax_max
    sm_sum = <TensorImpl>softmax_sum

    if q._device_type != 1 or k._device_type != 1 or v._device_type != 1 or g._device_type != 1:
        raise ValueError("fast_sdpa_flash_attention_backward_packed_qkv expects NPU tensors")
    if q._dtype_code != k._dtype_code or q._dtype_code != v._dtype_code or q._dtype_code != g._dtype_code:
        raise ValueError("fast_sdpa_flash_attention_backward_packed_qkv requires matching q/k/v/grad dtypes")
    if q._device_index != k._device_index or q._device_index != v._device_index or q._device_index != g._device_index:
        raise ValueError("fast_sdpa_flash_attention_backward_packed_qkv requires tensors on the same device")

    dev_idx = q._device_index
    q_dev = q._device_obj
    q_dtype = q._dtype_obj
    dtype_code = _tensor_dtype_to_acl_code(q)
    isize = q._itemsize
    B = q._c_shape[0]
    H = q._c_shape[1]
    Sq = q._c_shape[2]
    D = q._c_shape[3]
    Sk = k._c_shape[2]
    if heads != H or embed != H * D:
        raise ValueError("packed QKV metadata does not match SDPA grad shape")
    packed_width = 3 * embed
    if batch_first:
        if input_shape[0] != B or input_shape[1] != Sq or input_shape[2] != packed_width:
            raise ValueError("packed QKV batch-first shape mismatch")
        packed_shape = input_shape
        packed_stride = (Sq * packed_width, packed_width, 1)
        grad_stride = (Sq * packed_width, D, packed_width, 1)
    else:
        if input_shape[0] != Sq or input_shape[1] != B or input_shape[2] != packed_width:
            raise ValueError("packed QKV sequence-first shape mismatch")
        packed_shape = input_shape
        packed_stride = (B * packed_width, packed_width, 1)
        grad_stride = (packed_width, D, B * packed_width, 1)
    grad_shape = (B, H, Sq, D)
    packed_numel = B * Sq * packed_width
    packed_alloc = packed_numel * isize

    if not _sdpa_flash_valid_layout(q, B, H, Sq, D):
        raise ValueError("unsupported query layout for packed FlashAttentionScoreGrad")
    if not _sdpa_flash_valid_layout(k, B, H, Sk, D):
        raise ValueError("unsupported key layout for packed FlashAttentionScoreGrad")
    if not _sdpa_flash_valid_layout(v, B, H, Sk, D):
        raise ValueError("unsupported value layout for packed FlashAttentionScoreGrad")
    if Sk != Sq:
        raise ValueError("packed QKV FlashAttentionScoreGrad requires self-attention sequence lengths")
    if not _sdpa_flash_valid_grad_layout(g, B, H, Sq, D):
        raise ValueError("unsupported grad layout for packed FlashAttentionScoreGrad")

    stream_obj = _get_stream_obj_fast(dev_idx)
    stream_raw = _get_stream_raw_fast(dev_idx)
    if dev_idx == 0:
        _ensure_allocator_dev0()
        packed_ptr = _fast_allocator_dev0.malloc_large_cached(packed_alloc, stream_obj)
    else:
        packed_ptr = _get_allocator_fn_ref(dev_idx).malloc(packed_alloc, stream=stream_obj)
    packed_raw = <uintptr_t>int(packed_ptr)
    dq_raw = packed_raw
    dk_raw = packed_raw + <uintptr_t>(embed * isize)
    dv_raw = packed_raw + <uintptr_t>(2 * embed * isize)

    q_shape_buf[0] = B
    q_shape_buf[1] = H
    q_shape_buf[2] = Sq
    q_shape_buf[3] = D
    q_stride_buf[0] = <int64_t>grad_stride[0]
    q_stride_buf[1] = <int64_t>grad_stride[1]
    q_stride_buf[2] = <int64_t>grad_stride[2]
    q_stride_buf[3] = <int64_t>grad_stride[3]
    packed_storage_shape_buf[0] = <int64_t>packed_shape[0]
    packed_storage_shape_buf[1] = <int64_t>packed_shape[1]
    packed_storage_shape_buf[2] = <int64_t>packed_shape[2]
    softmax_empty_shape_buf[0] = 0
    softmax_empty_stride_buf[0] = 1

    g_ptr = <uintptr_t>g._storage._untyped._device_ptr + <uintptr_t>(g._c_offset * isize)
    q_ptr = <uintptr_t>q._storage._untyped._device_ptr + <uintptr_t>(q._c_offset * isize)
    k_ptr = <uintptr_t>k._storage._untyped._device_ptr + <uintptr_t>(k._c_offset * isize)
    v_ptr = <uintptr_t>v._storage._untyped._device_ptr + <uintptr_t>(v._c_offset * isize)
    o_ptr = <uintptr_t>o._storage._untyped._device_ptr + <uintptr_t>(o._c_offset * isize)
    sm_max_ptr = <uintptr_t>sm_max._storage._untyped._device_ptr + <uintptr_t>(sm_max._c_offset * sm_max._itemsize)
    sm_sum_ptr = <uintptr_t>sm_sum._storage._untyped._device_ptr + <uintptr_t>(sm_sum._c_offset * sm_sum._itemsize)

    use_grad_v2 = _use_sdpa_flash_grad_v2 and _sdpa_flash_grad_v2_getws_ptr is not None
    if use_grad_v2:
        flash_grad_getws_raw = <uintptr_t>int(_sdpa_flash_grad_v2_getws_ptr)
        flash_grad_exec_raw = <uintptr_t>int(_sdpa_flash_grad_v2_exec_ptr)
    else:
        flash_grad_getws_raw = <uintptr_t>int(_sdpa_flash_grad_getws_ptr)
        flash_grad_exec_raw = <uintptr_t>int(_sdpa_flash_grad_exec_ptr)
    layout[0] = 66
    layout[1] = 78
    layout[2] = 83
    layout[3] = 68
    layout[4] = 0

    try:
        if _use_sdpa_flash_grad_pta_cache and _pta_cache_end_fn is not None:
            if use_grad_v2:
                pta_lookup = _ffi_pta_begin_sdpa_flash_grad_storage_cache_lookup_raw(
                    b"aclnnFlashAttentionScoreGradV2",
                    q._shape_tuple, q._stride_tuple,
                    k._shape_tuple, k._stride_tuple,
                    v._shape_tuple, v._stride_tuple,
                    g._shape_tuple, g._stride_tuple,
                    o._shape_tuple, o._stride_tuple,
                    sm_max._shape_tuple, sm_max._stride_tuple,
                    grad_shape, grad_stride,
                    grad_shape, grad_stride,
                    grad_shape, grad_stride,
                    packed_shape, packed_shape, packed_shape,
                    0, embed, 2 * embed,
                    dtype_code,
                    q_ptr, k_ptr, v_ptr, g_ptr, o_ptr, sm_max_ptr, sm_sum_ptr,
                    packed_raw, packed_raw, packed_raw,
                    scale_value, 1.0, 2147483647, 2147483647, H, 0, 0, pse_type,
                    stream_raw,
                    &pta_active,
                    &pta_ws_size,
                    &pta_executor)
            else:
                pta_lookup = _ffi_pta_begin_sdpa_flash_grad_storage_cache_lookup_raw(
                    b"aclnnFlashAttentionScoreGrad",
                    q._shape_tuple, q._stride_tuple,
                    k._shape_tuple, k._stride_tuple,
                    v._shape_tuple, v._stride_tuple,
                    g._shape_tuple, g._stride_tuple,
                    o._shape_tuple, o._stride_tuple,
                    sm_max._shape_tuple, sm_max._stride_tuple,
                    grad_shape, grad_stride,
                    grad_shape, grad_stride,
                    grad_shape, grad_stride,
                    packed_shape, packed_shape, packed_shape,
                    0, embed, 2 * embed,
                    dtype_code,
                    q_ptr, k_ptr, v_ptr, g_ptr, o_ptr, sm_max_ptr, sm_sum_ptr,
                    packed_raw, packed_raw, packed_raw,
                    scale_value, 1.0, 2147483647, 2147483647, H, 0, 0, 0,
                    stream_raw,
                    &pta_active,
                    &pta_ws_size,
                    &pta_executor)
            if pta_lookup and pta_executor != 0:
                try:
                    runtime = _run_executor(
                        flash_grad_exec_raw, pta_ws_size, pta_executor,
                        stream_raw, dev_idx, runtime, "FlashAttentionScoreGrad")
                    packed_grad = _make_npu_tensor_fast_large(
                        packed_raw, packed_numel, q_dtype, q_dev, packed_shape, packed_stride,
                        isize, dev_idx, q._dtype_code)
                    packed_wrapped = True
                    packed_storage = (<TensorImpl>packed_grad)._storage
                    grad_q = cy_make_tensor_from_storage_trusted(packed_storage, grad_shape, grad_stride, 0, q_dev, 1, dev_idx, q_dtype, q._dtype_code, isize)
                    grad_k = cy_make_tensor_from_storage_trusted(packed_storage, grad_shape, grad_stride, embed, q_dev, 1, dev_idx, q_dtype, q._dtype_code, isize)
                    grad_v = cy_make_tensor_from_storage_trusted(packed_storage, grad_shape, grad_stride, 2 * embed, q_dev, 1, dev_idx, q_dtype, q._dtype_code, isize)
                    return grad_q, grad_k, grad_v, packed_grad
                finally:
                    if pta_active:
                        _ffi_pta_end_cache_lookup()
                        pta_active = False

        with nogil:
            q_t = _ffi_create_tensor_raw(q._c_shape, q._c_stride, <uint64_t>4, dtype_code, 2, <void*>q_ptr)
            k_t = _ffi_create_tensor_raw(k._c_shape, k._c_stride, <uint64_t>4, dtype_code, 2, <void*>k_ptr)
            v_t = _ffi_create_tensor_raw(v._c_shape, v._c_stride, <uint64_t>4, dtype_code, 2, <void*>v_ptr)
            g_t = _ffi_create_tensor_raw(g._c_shape, g._c_stride, <uint64_t>4, dtype_code, 2, <void*>g_ptr)
            o_t = _ffi_create_tensor_raw(o._c_shape, o._c_stride, <uint64_t>4, dtype_code, 2, <void*>o_ptr)
            softmax_max_t = _ffi_create_tensor_raw(sm_max._c_shape, sm_max._c_stride, <uint64_t>4, 0, 2, <void*>sm_max_ptr)
            softmax_sum_t = _ffi_create_tensor_raw(sm_sum._c_shape, sm_sum._c_stride, <uint64_t>4, 0, 2, <void*>sm_sum_ptr)
            if use_grad_v2:
                softmax_in_t = _ffi_create_tensor_raw(softmax_empty_shape_buf, softmax_empty_stride_buf, <uint64_t>1, dtype_code, 2, <void*>q_ptr)
            dq_t = _ffi_create_tensor_raw_with_storage(q_shape_buf, q_stride_buf, <uint64_t>4, packed_storage_shape_buf, <uint64_t>3, 0, dtype_code, 2, <void*>packed_raw)
            dk_t = _ffi_create_tensor_raw_with_storage(q_shape_buf, q_stride_buf, <uint64_t>4, packed_storage_shape_buf, <uint64_t>3, embed, dtype_code, 2, <void*>packed_raw)
            dv_t = _ffi_create_tensor_raw_with_storage(q_shape_buf, q_stride_buf, <uint64_t>4, packed_storage_shape_buf, <uint64_t>3, 2 * embed, dtype_code, 2, <void*>packed_raw)
        if q_t == NULL or k_t == NULL or v_t == NULL or g_t == NULL or o_t == NULL or softmax_max_t == NULL or softmax_sum_t == NULL or dq_t == NULL or dk_t == NULL or dv_t == NULL or (use_grad_v2 and softmax_in_t == NULL):
            if pta_active:
                _ffi_pta_end_cache_lookup()
                pta_active = False
            raise RuntimeError("aclCreateTensor returned null")
        try:
            with nogil:
                if use_grad_v2:
                    ret = (<FlashAttentionScoreGradV2GetWorkspaceSize_t>flash_grad_getws_raw)(
                        q_t, k_t, v_t, g_t, NULL, NULL, NULL, NULL,
                        softmax_max_t, softmax_sum_t, softmax_in_t, o_t, NULL, NULL, NULL,
                        scale_value, 1.0, 2147483647, 2147483647, H, layout, 0, 0, pse_type,
                        dq_t, dk_t, dv_t, NULL, &ws_size, &executor)
                else:
                    ret = (<FlashAttentionScoreGradGetWorkspaceSize_t>flash_grad_getws_raw)(
                        q_t, k_t, v_t, g_t, NULL, NULL, NULL, NULL,
                        softmax_max_t, softmax_sum_t, NULL, o_t, NULL,
                        scale_value, 1.0, 2147483647, 2147483647, H, layout, 0, 0,
                        dq_t, dk_t, dv_t, NULL, &ws_size, &executor)
            if ret != 0:
                raise RuntimeError(f"FlashAttentionScoreGrad GetWorkspaceSize failed: {ret}")
            runtime = _run_executor(flash_grad_exec_raw, ws_size, <uintptr_t>executor, stream_raw, dev_idx, runtime, "FlashAttentionScoreGrad")
            _defer_executor_handle(<uintptr_t>executor)
            executor = NULL
        finally:
            if pta_active:
                _ffi_pta_end_cache_lookup()
            if executor != NULL:
                _ffi_ref.destroy_executor(<uintptr_t>executor)
            with nogil:
                if q_t != NULL: _ffi_destroy_tensor_raw(q_t)
                if k_t != NULL: _ffi_destroy_tensor_raw(k_t)
                if v_t != NULL: _ffi_destroy_tensor_raw(v_t)
                if g_t != NULL: _ffi_destroy_tensor_raw(g_t)
                if o_t != NULL: _ffi_destroy_tensor_raw(o_t)
                if softmax_max_t != NULL: _ffi_destroy_tensor_raw(softmax_max_t)
                if softmax_sum_t != NULL: _ffi_destroy_tensor_raw(softmax_sum_t)
                if softmax_in_t != NULL: _ffi_destroy_tensor_raw(softmax_in_t)
                if dq_t != NULL: _ffi_destroy_tensor_raw(dq_t)
                if dk_t != NULL: _ffi_destroy_tensor_raw(dk_t)
                if dv_t != NULL: _ffi_destroy_tensor_raw(dv_t)

        packed_grad = _make_npu_tensor_fast_large(
            packed_raw, packed_numel, q_dtype, q_dev, packed_shape, packed_stride,
            isize, dev_idx, q._dtype_code)
        packed_wrapped = True
        packed_storage = (<TensorImpl>packed_grad)._storage
        grad_q = cy_make_tensor_from_storage_trusted(packed_storage, grad_shape, grad_stride, 0, q_dev, 1, dev_idx, q_dtype, q._dtype_code, isize)
        grad_k = cy_make_tensor_from_storage_trusted(packed_storage, grad_shape, grad_stride, embed, q_dev, 1, dev_idx, q_dtype, q._dtype_code, isize)
        grad_v = cy_make_tensor_from_storage_trusted(packed_storage, grad_shape, grad_stride, 2 * embed, q_dev, 1, dev_idx, q_dtype, q._dtype_code, isize)
        return grad_q, grad_k, grad_v, packed_grad
    except Exception:
        if not packed_wrapped:
            _get_allocator_fn_ref(dev_idx).free(<int64_t>packed_raw, stream=stream_obj)
        raise


cpdef object fast_matmul_exact(TensorImpl a, TensorImpl b):
    """2D Matmul for already-validated exact base NPU tensors."""
    _ensure_npu_imports()
    _ensure_ffi_binary()

    cdef int dev_idx = a._device_index
    cdef object a_dev = a._device_obj
    cdef object a_dtype = a._dtype_obj
    cdef object runtime = None
    cdef object stream_obj = _get_stream_obj_fast(dev_idx)
    cdef int64_t m
    cdef int64_t k
    cdef int64_t k_b
    cdef int64_t n
    cdef int64_t numel
    cdef int64_t alloc_size
    cdef int isize
    cdef int dtype_code
    cdef object out_shape
    cdef object out_stride
    cdef object out_ptr
    cdef uintptr_t a_ptr
    cdef uintptr_t b_ptr
    cdef uintptr_t o_ptr
    cdef uintptr_t stream_raw
    cdef bint pta_active = False
    cdef uint64_t ws_size_raw = 0
    cdef uintptr_t executor_raw = 0
    cdef int pta_lookup
    cdef int ret_i
    cdef object ws_size
    cdef object executor = 0
    cdef object workspace_ptr
    cdef object ret

    if a._dtype_code != b._dtype_code:
        raise ValueError("NPU matmul requires matching dtypes")
    if a._device_index != b._device_index:
        raise ValueError("NPU matmul requires tensors on the same device")
    if a._ndim != 2 or b._ndim != 2:
        raise ValueError("fast_matmul_exact currently supports 2D tensors")

    m = a._c_shape[0]
    k = a._c_shape[1]
    k_b = b._c_shape[0]
    n = b._c_shape[1]
    if k != k_b:
        raise ValueError(f"matmul shape mismatch: ({m}, {k}) @ ({k_b}, {n})")

    isize = a._itemsize
    numel = m * n
    alloc_size = numel * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc_large_cached(alloc_size, stream_obj)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size, stream=stream_obj)

    dtype_code = _tensor_dtype_to_acl_code(a)
    out_shape = (m, n)
    out_stride = (n, 1)
    a_ptr = <uintptr_t>a._storage._untyped._device_ptr + <uintptr_t>(a._c_offset * isize)
    b_ptr = <uintptr_t>b._storage._untyped._device_ptr + <uintptr_t>(b._c_offset * isize)
    o_ptr = <uintptr_t>int(out_ptr)
    stream_raw = _get_stream_raw_fast(dev_idx)

    if _use_matmul_pta_cache and _pta_binary_begin_fn is not None:
        pta_lookup = _ffi_pta_begin_binary_cache_lookup_raw(
            b"aclnnMatmul",
            a._shape_tuple, a._stride_tuple,
            b._shape_tuple, b._stride_tuple,
            out_shape, out_stride,
            dtype_code,
            a_ptr, b_ptr, o_ptr,
            stream_raw,
            &pta_active,
            &ws_size_raw,
            &executor_raw)
        if pta_lookup and executor_raw != 0:
            try:
                runtime = _run_executor(
                    <uintptr_t>int(_matmul_exec_ptr), ws_size_raw, executor_raw,
                    stream_raw, dev_idx, runtime, "aclnnMatmul")
                return _make_npu_tensor_fast_large(out_ptr, numel, a_dtype, a_dev, out_shape, out_stride, isize, dev_idx, a._dtype_code)
            finally:
                if pta_active:
                    _ffi_pta_end_cache_lookup()
                    pta_active = False

    try:
        ws_size, executor = _ffi_binary_two_inputs_with_int8_op(
            _matmul_getws_ptr, _matmul_exec_ptr,
            a._shape_tuple, a._stride_tuple,
            b._shape_tuple, b._stride_tuple,
            out_shape, out_stride,
            1,
            dtype_code, dtype_code, dtype_code, 2,
            a_ptr, b_ptr, o_ptr,
            stream_raw)
        if ws_size:
            workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            try:
                ret = _ffi_execute(
                    _matmul_exec_ptr, int(workspace_ptr), ws_size,
                    executor, stream_raw)
                if ret != 0:
                    raise RuntimeError(f"aclnnMatmul execute failed: {ret}")
            finally:
                if runtime is None:
                    runtime = _get_runtime_fast(dev_idx)
                runtime.defer_raw_free(workspace_ptr)
    finally:
        if executor:
            _defer_executor_handle(<uintptr_t>int(executor))
        if pta_active:
            _ffi_pta_end_cache_lookup()

    return _make_npu_tensor_fast_large(out_ptr, numel, a_dtype, a_dev, out_shape, out_stride, isize, dev_idx, a._dtype_code)


cpdef object fast_matmul(object a, object b):
    """Optimized 2D NPU matmul that bypasses the Python aclnn.matmul wrapper."""
    if isinstance(a, TensorImpl) and isinstance(b, TensorImpl):
        return fast_matmul_exact(<TensorImpl>a, <TensorImpl>b)
    raise ValueError("fast_matmul expects base TensorImpl operands")


cpdef object fast_sum(object a, object dim=None, bint keepdim=False):
    """Optimized NPU ReduceSum that bypasses the Python aclnn.reduce_sum wrapper."""
    _ensure_npu_imports()
    _ensure_ffi_reduce_sum()

    if not isinstance(a, TensorImpl):
        raise ValueError("fast_sum expects a base TensorImpl operand")
    cdef TensorImpl t = <TensorImpl>a
    if t._device_type != 1:
        raise ValueError("NPU sum expects an NPU tensor")

    cdef int dev_idx = t._device_index
    cdef object a_dev = t._device_obj
    cdef object a_dtype = t._dtype_obj
    cdef object stream_obj = _get_stream_obj_fast(dev_idx)
    cdef object shape = t._shape_tuple
    cdef object stride = t._stride_tuple
    cdef int ndim = t._ndim
    cdef object dims
    cdef list out_shape_list
    cdef object out_shape
    cdef object out_stride
    cdef int d
    cdef int d_norm
    cdef int64_t n = 1
    cdef int isize = t._itemsize
    cdef int dtype_code = _tensor_dtype_to_acl_code(t)
    cdef int64_t alloc_size
    cdef object out_ptr
    cdef uintptr_t a_ptr
    cdef uintptr_t o_ptr
    cdef uintptr_t stream_raw
    cdef object ws_size
    cdef object executor = 0
    cdef object workspace_ptr
    cdef object ret
    cdef object runtime = None
    cdef bint pta_active = False
    cdef uint64_t ws_size_raw = 0
    cdef uintptr_t executor_raw = 0
    cdef int pta_lookup
    cdef int ret_i

    if dim is None or (isinstance(dim, (list, tuple)) and len(dim) == 0):
        dims = tuple(range(ndim))
    elif isinstance(dim, int):
        d = <int>dim
        if d < -ndim or d >= ndim:
            raise IndexError(f"Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], but got {d})")
        dims = ((d + ndim) if d < 0 else d,)
    elif isinstance(dim, (list, tuple)):
        norm_dims = []
        seen_dims = set()
        for item in dim:
            d = <int>item
            if d < -ndim or d >= ndim:
                raise IndexError(f"Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], but got {d})")
            d_norm = (d + ndim) if d < 0 else d
            if d_norm in seen_dims:
                raise RuntimeError(f"dim {d_norm} appears multiple times in the list of dims")
            seen_dims.add(d_norm)
            norm_dims.append(d_norm)
        dims = tuple(norm_dims)
    else:
        raise TypeError("sum dim must be int, tuple/list, or None")

    out_shape_list = [shape[i] for i in range(ndim)]
    for d_norm in sorted(dims):
        out_shape_list[d_norm] = 1
    if not keepdim:
        out_shape_list = [s for i, s in enumerate(out_shape_list) if i not in dims]
    out_shape = tuple(out_shape_list)
    out_stride = _contiguous_stride_tuple(out_shape)
    for size in out_shape:
        n *= <int64_t>size
    if len(out_shape) == 0:
        n = 1

    alloc_size = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc_large_cached(alloc_size, stream_obj)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size, stream=stream_obj)

    a_ptr = <uintptr_t>t._storage._untyped._device_ptr + <uintptr_t>(t._c_offset * isize)
    o_ptr = <uintptr_t>int(out_ptr)
    stream_raw = _get_stream_raw_fast(dev_idx)

    if _use_reduce_sum_pta_cache and _pta_cache_begin_fn is not None:
        pta_lookup = _ffi_pta_begin_reduce_sum_cache_lookup_raw(
            shape, stride,
            out_shape, out_stride,
            dims, keepdim,
            dtype_code,
            a_ptr, o_ptr,
            stream_raw,
            &pta_active,
            &ws_size_raw,
            &executor_raw)
        if pta_lookup and executor_raw != 0:
            try:
                runtime = _run_executor(
                    <uintptr_t>int(_reduce_sum_exec_ptr), ws_size_raw, executor_raw,
                    stream_raw, dev_idx, runtime, "aclnnReduceSum")
                return _make_npu_tensor_fast_large(out_ptr, n, a_dtype, a_dev, out_shape, out_stride, isize, dev_idx, t._dtype_code)
            finally:
                if pta_active:
                    _ffi_pta_end_cache_lookup()
                    pta_active = False

    try:
        ws_size, executor = _ffi_reduce_sum_op(
            _reduce_sum_getws_ptr, _reduce_sum_exec_ptr,
            shape, stride,
            out_shape, out_stride,
            dims, keepdim,
            dtype_code, 2,
            a_ptr, o_ptr,
            stream_raw)
        if ws_size:
            workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            try:
                ret = _ffi_execute(
                    _reduce_sum_exec_ptr, int(workspace_ptr), ws_size,
                    executor, stream_raw)
                if ret != 0:
                    raise RuntimeError(f"aclnnReduceSum execute failed: {ret}")
            finally:
                if runtime is None:
                    runtime = _get_runtime_fast(dev_idx)
                runtime.defer_raw_free(workspace_ptr)

        _defer_executor_handle(<uintptr_t>int(executor))
    finally:
        if pta_active:
            _ffi_pta_end_cache_lookup()

    return _make_npu_tensor_fast_large(out_ptr, n, a_dtype, a_dev, out_shape, out_stride, isize, dev_idx, t._dtype_code)


cpdef object fast_mm_mat1_backward(object grad, object mat2, object alpha=1):
    """Optimized addmm/mm mat1 gradient: grad @ mat2.T without transpose dispatch."""
    if alpha != 1:
        raise ValueError("fast_mm_mat1_backward only handles alpha=1")
    if not (isinstance(grad, TensorImpl) and isinstance(mat2, TensorImpl)):
        raise ValueError("fast_mm_mat1_backward expects base TensorImpl operands")
    cdef TensorImpl g = <TensorImpl>grad
    cdef TensorImpl b = <TensorImpl>mat2
    _ensure_npu_imports()
    _ensure_ffi_binary()
    if g._device_type != 1 or b._device_type != 1:
        raise ValueError("NPU mm_mat1_backward expects NPU tensors")
    if g._device_index != b._device_index:
        raise ValueError("NPU mm_mat1_backward requires tensors on the same device")
    if g._dtype_code != b._dtype_code:
        raise ValueError("NPU mm_mat1_backward requires matching dtypes")
    if g._ndim != 2 or b._ndim != 2:
        raise ValueError("fast_mm_mat1_backward expects 2D operands")

    cdef int64_t m = g._c_shape[0]
    cdef int64_t n = g._c_shape[1]
    cdef int64_t k = b._c_shape[0]
    cdef int64_t n_b = b._c_shape[1]
    if n != n_b:
        raise ValueError(f"mm_mat1_backward shape mismatch: grad ({m}, {n}) and mat2 ({k}, {n_b})")

    cdef int dev_idx = g._device_index
    cdef object a_dtype = g._dtype_obj
    cdef object a_dev = g._device_obj
    cdef object stream_obj = _get_stream_obj_fast(dev_idx)
    cdef int isize = g._itemsize
    cdef int64_t numel = m * k
    cdef int64_t alloc_size = numel * isize
    cdef object out_ptr
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc_large_cached(alloc_size, stream_obj)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size, stream=stream_obj)

    cdef object out_shape = (m, k)
    cdef object out_stride = (k, 1)
    cdef object b_t_shape = (n_b, k)
    cdef object b_t_stride = (b._c_stride[1], b._c_stride[0])
    cdef int dtype_code = _tensor_dtype_to_acl_code(g)
    cdef uintptr_t g_ptr = <uintptr_t>g._storage._untyped._device_ptr + <uintptr_t>(g._c_offset * isize)
    cdef uintptr_t b_ptr = <uintptr_t>b._storage._untyped._device_ptr + <uintptr_t>(b._c_offset * b._itemsize)
    cdef uintptr_t o_ptr = <uintptr_t>int(out_ptr)
    cdef uintptr_t stream_raw = _get_stream_raw_fast(dev_idx)
    cdef bint pta_active = False
    cdef uint64_t ws_size_raw = 0
    cdef uintptr_t executor_raw = 0
    cdef int pta_lookup
    cdef int ret_i
    cdef object ws_size
    cdef object executor = 0
    cdef object workspace_ptr
    cdef object ret
    cdef object runtime = None

    if _use_matmul_pta_cache and _pta_binary_begin_fn is not None:
        pta_lookup = _ffi_pta_begin_binary_cache_lookup_raw(
            b"aclnnMatmul",
            g._shape_tuple, g._stride_tuple,
            b_t_shape, b_t_stride,
            out_shape, out_stride,
            dtype_code,
            g_ptr, b_ptr, o_ptr,
            stream_raw,
            &pta_active,
            &ws_size_raw,
            &executor_raw)
        if pta_lookup and executor_raw != 0:
            try:
                runtime = _run_executor(
                    <uintptr_t>int(_matmul_exec_ptr), ws_size_raw, executor_raw,
                    stream_raw, dev_idx, runtime, "aclnnMatmul")
                return _make_npu_tensor_fast_large(out_ptr, numel, a_dtype, a_dev, out_shape, out_stride, isize, dev_idx, g._dtype_code)
            finally:
                if pta_active:
                    _ffi_pta_end_cache_lookup()
                    pta_active = False

    try:
        ws_size, executor = _ffi_binary_two_inputs_with_int8_op(
            _matmul_getws_ptr, _matmul_exec_ptr,
            g._shape_tuple, g._stride_tuple,
            b_t_shape, b_t_stride,
            out_shape, out_stride,
            1,
            dtype_code, dtype_code, dtype_code, 2,
            g_ptr, b_ptr, o_ptr,
            stream_raw)
        if ws_size:
            workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            try:
                ret = _ffi_execute(_matmul_exec_ptr, int(workspace_ptr), ws_size, executor, stream_raw)
                if ret != 0:
                    raise RuntimeError(f"aclnnMatmul execute failed: {ret}")
            finally:
                if runtime is None:
                    runtime = _get_runtime_fast(dev_idx)
                runtime.defer_raw_free(workspace_ptr)
    finally:
        if executor:
            _defer_executor_handle(<uintptr_t>int(executor))
        if pta_active:
            _ffi_pta_end_cache_lookup()
    return _make_npu_tensor_fast_large(out_ptr, numel, a_dtype, a_dev, out_shape, out_stride, isize, dev_idx, g._dtype_code)


cpdef object fast_mm_mat2_backward(object grad, object mat1, object alpha=1):
    """Optimized addmm/mm mat2 gradient: mat1.T @ grad without transpose dispatch."""
    if alpha != 1:
        raise ValueError("fast_mm_mat2_backward only handles alpha=1")
    if not (isinstance(grad, TensorImpl) and isinstance(mat1, TensorImpl)):
        raise ValueError("fast_mm_mat2_backward expects base TensorImpl operands")
    cdef TensorImpl g = <TensorImpl>grad
    cdef TensorImpl a = <TensorImpl>mat1
    _ensure_npu_imports()
    _ensure_ffi_binary()
    if g._device_type != 1 or a._device_type != 1:
        raise ValueError("NPU mm_mat2_backward expects NPU tensors")
    if g._device_index != a._device_index:
        raise ValueError("NPU mm_mat2_backward requires tensors on the same device")
    if g._dtype_code != a._dtype_code:
        raise ValueError("NPU mm_mat2_backward requires matching dtypes")
    if g._ndim != 2 or a._ndim != 2:
        raise ValueError("fast_mm_mat2_backward expects 2D operands")

    cdef int64_t m = g._c_shape[0]
    cdef int64_t n = g._c_shape[1]
    cdef int64_t m_a = a._c_shape[0]
    cdef int64_t k = a._c_shape[1]
    if m != m_a:
        raise ValueError(f"mm_mat2_backward shape mismatch: grad ({m}, {n}) and mat1 ({m_a}, {k})")

    cdef int dev_idx = g._device_index
    cdef object a_dtype = g._dtype_obj
    cdef object a_dev = g._device_obj
    cdef object stream_obj = _get_stream_obj_fast(dev_idx)
    cdef int isize = g._itemsize
    cdef int64_t numel = k * n
    cdef int64_t alloc_size = numel * isize
    cdef object out_ptr
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc_large_cached(alloc_size, stream_obj)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size, stream=stream_obj)

    cdef object a_t_shape = (k, m_a)
    cdef object a_t_stride = (a._c_stride[1], a._c_stride[0])
    cdef object out_shape = (k, n)
    cdef object out_stride = (n, 1)
    cdef int dtype_code = _tensor_dtype_to_acl_code(g)
    cdef uintptr_t a_ptr = <uintptr_t>a._storage._untyped._device_ptr + <uintptr_t>(a._c_offset * a._itemsize)
    cdef uintptr_t g_ptr = <uintptr_t>g._storage._untyped._device_ptr + <uintptr_t>(g._c_offset * isize)
    cdef uintptr_t o_ptr = <uintptr_t>int(out_ptr)
    cdef uintptr_t stream_raw = _get_stream_raw_fast(dev_idx)
    cdef bint pta_active = False
    cdef uint64_t ws_size_raw = 0
    cdef uintptr_t executor_raw = 0
    cdef int pta_lookup
    cdef int ret_i
    cdef object ws_size
    cdef object executor = 0
    cdef object workspace_ptr
    cdef object ret
    cdef object runtime = None

    if _use_matmul_pta_cache and _pta_binary_begin_fn is not None:
        pta_lookup = _ffi_pta_begin_binary_cache_lookup_raw(
            b"aclnnMatmul",
            a_t_shape, a_t_stride,
            g._shape_tuple, g._stride_tuple,
            out_shape, out_stride,
            dtype_code,
            a_ptr, g_ptr, o_ptr,
            stream_raw,
            &pta_active,
            &ws_size_raw,
            &executor_raw)
        if pta_lookup and executor_raw != 0:
            try:
                runtime = _run_executor(
                    <uintptr_t>int(_matmul_exec_ptr), ws_size_raw, executor_raw,
                    stream_raw, dev_idx, runtime, "aclnnMatmul")
                return _make_npu_tensor_fast_large(out_ptr, numel, a_dtype, a_dev, out_shape, out_stride, isize, dev_idx, g._dtype_code)
            finally:
                if pta_active:
                    _ffi_pta_end_cache_lookup()
                    pta_active = False

    try:
        ws_size, executor = _ffi_binary_two_inputs_with_int8_op(
            _matmul_getws_ptr, _matmul_exec_ptr,
            a_t_shape, a_t_stride,
            g._shape_tuple, g._stride_tuple,
            out_shape, out_stride,
            1,
            dtype_code, dtype_code, dtype_code, 2,
            a_ptr, g_ptr, o_ptr,
            stream_raw)
        if ws_size:
            workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            try:
                ret = _ffi_execute(_matmul_exec_ptr, int(workspace_ptr), ws_size, executor, stream_raw)
                if ret != 0:
                    raise RuntimeError(f"aclnnMatmul execute failed: {ret}")
            finally:
                if runtime is None:
                    runtime = _get_runtime_fast(dev_idx)
                runtime.defer_raw_free(workspace_ptr)
    finally:
        if executor:
            _defer_executor_handle(<uintptr_t>int(executor))
        if pta_active:
            _ffi_pta_end_cache_lookup()
    return _make_npu_tensor_fast_large(out_ptr, numel, a_dtype, a_dev, out_shape, out_stride, isize, dev_idx, g._dtype_code)


cpdef object fast_addmm(object bias, object mat1, object mat2, object beta=1, object alpha=1):
    """Optimized 2D NPU addmm that bypasses the Python aclnn.addmm wrapper."""
    _ensure_npu_imports()
    _ensure_ffi_addmm()

    if not (isinstance(bias, TensorImpl) and isinstance(mat1, TensorImpl) and isinstance(mat2, TensorImpl)):
        raise ValueError("fast_addmm expects base TensorImpl operands")
    cdef TensorImpl self_t = <TensorImpl>bias
    cdef TensorImpl a = <TensorImpl>mat1
    cdef TensorImpl b = <TensorImpl>mat2
    cdef int dev_idx = a._device_index
    cdef object a_dtype = a._dtype_obj
    cdef object a_dev = a._device_obj
    cdef object runtime = None
    cdef object stream_obj = _get_stream_obj_fast(dev_idx)
    cdef int64_t m
    cdef int64_t k
    cdef int64_t k_b
    cdef int64_t n
    cdef int64_t numel
    cdef int64_t alloc_size
    cdef int isize
    cdef int dtype_code
    cdef object out_shape
    cdef object out_stride
    cdef object out_ptr
    cdef uintptr_t self_ptr
    cdef uintptr_t a_ptr
    cdef uintptr_t b_ptr
    cdef uintptr_t o_ptr
    cdef uintptr_t beta_scalar = 0
    cdef uintptr_t alpha_scalar = 0
    cdef bint beta_scalar_owned = False
    cdef bint alpha_scalar_owned = False
    cdef uintptr_t stream_raw
    cdef bint pta_active = False
    cdef uint64_t ws_size_raw = 0
    cdef uintptr_t executor_raw = 0
    cdef int pta_lookup
    cdef int ret_i
    cdef object beta_bytes_pair
    cdef object alpha_bytes_pair
    cdef object ws_size
    cdef object executor = 0
    cdef object workspace_ptr
    cdef object ret

    if a._device_type != 1 or b._device_type != 1 or self_t._device_type != 1:
        raise ValueError("NPU addmm expects NPU tensors")
    if a._device_index != b._device_index or a._device_index != self_t._device_index:
        raise ValueError("NPU addmm requires tensors on the same device")
    if a._dtype_code != b._dtype_code or a._dtype_code != self_t._dtype_code:
        raise ValueError("NPU addmm requires matching dtypes")
    if a._ndim != 2 or b._ndim != 2:
        raise ValueError("fast_addmm currently supports 2D mat operands")

    m = a._c_shape[0]
    k = a._c_shape[1]
    k_b = b._c_shape[0]
    n = b._c_shape[1]
    if k != k_b:
        raise ValueError(f"addmm shape mismatch: ({m}, {k}) @ ({k_b}, {n})")
    if not (self_t._ndim == 1 and self_t._c_shape[0] == n):
        raise ValueError("fast_addmm currently supports 1D bias matching output columns")

    isize = a._itemsize
    numel = m * n
    alloc_size = numel * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc_large_cached(alloc_size, stream_obj)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size, stream=stream_obj)

    dtype_code = _tensor_dtype_to_acl_code(a)
    out_shape = (m, n)
    out_stride = (n, 1)
    self_ptr = <uintptr_t>self_t._storage._untyped._device_ptr + <uintptr_t>(self_t._c_offset * isize)
    a_ptr = <uintptr_t>a._storage._untyped._device_ptr + <uintptr_t>(a._c_offset * isize)
    b_ptr = <uintptr_t>b._storage._untyped._device_ptr + <uintptr_t>(b._c_offset * isize)
    o_ptr = <uintptr_t>int(out_ptr)
    stream_raw = _get_stream_raw_fast(dev_idx)

    if beta == 1:
        beta_scalar = _get_alpha_one(dtype_code)
        beta_bytes_pair = _get_alpha_one_bytes(dtype_code)
    else:
        beta_bytes_pair = (_scalar_bytes_fn(beta, a_dtype), dtype_code)
        beta_scalar = _create_scalar_fn(beta_bytes_pair[0], beta_bytes_pair[1])
        beta_scalar_owned = True
    if alpha == 1:
        alpha_scalar = _get_alpha_one(dtype_code)
        alpha_bytes_pair = _get_alpha_one_bytes(dtype_code)
    else:
        alpha_bytes_pair = (_scalar_bytes_fn(alpha, a_dtype), dtype_code)
        alpha_scalar = _create_scalar_fn(alpha_bytes_pair[0], alpha_bytes_pair[1])
        alpha_scalar_owned = True

    if _use_addmm_pta_cache and _pta_binary_begin_fn is not None:
        pta_lookup = _ffi_pta_begin_addmm_cache_lookup_raw(
            self_t._shape_tuple, self_t._stride_tuple,
            a._shape_tuple, a._stride_tuple,
            b._shape_tuple, b._stride_tuple,
            out_shape, out_stride,
            dtype_code, dtype_code, dtype_code, dtype_code,
            self_ptr, a_ptr, b_ptr, o_ptr,
            beta_bytes_pair[0], beta_bytes_pair[1],
            alpha_bytes_pair[0], alpha_bytes_pair[1],
            1,
            stream_raw,
            &pta_active,
            &ws_size_raw,
            &executor_raw)
        if pta_lookup and executor_raw != 0:
            try:
                runtime = _run_executor(
                    <uintptr_t>int(_addmm_exec_ptr), ws_size_raw, executor_raw,
                    stream_raw, dev_idx, runtime, "aclnnAddmm")
                return _make_npu_tensor_fast_large(out_ptr, numel, a_dtype, a_dev, out_shape, out_stride, isize, dev_idx, a._dtype_code)
            finally:
                if pta_active:
                    _ffi_pta_end_cache_lookup()
                    pta_active = False
                if beta_scalar_owned and beta_scalar:
                    _destroy_scalar_fn(int(beta_scalar))
                if alpha_scalar_owned and alpha_scalar:
                    _destroy_scalar_fn(int(alpha_scalar))

    try:
        ws_size, executor = _ffi_four_tensor_two_scalars_one_int8_op(
            _addmm_getws_ptr, _addmm_exec_ptr,
            self_t._shape_tuple, self_t._stride_tuple,
            a._shape_tuple, a._stride_tuple,
            b._shape_tuple, b._stride_tuple,
            out_shape, out_stride,
            1,
            dtype_code, dtype_code, dtype_code, dtype_code, 2,
            self_ptr, a_ptr, b_ptr, o_ptr,
            beta_scalar, alpha_scalar,
            stream_raw)
        if ws_size:
            workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            try:
                ret = _ffi_execute(
                    _addmm_exec_ptr, int(workspace_ptr), ws_size,
                    executor, stream_raw)
                if ret != 0:
                    raise RuntimeError(f"aclnnAddmm execute failed: {ret}")
            finally:
                if runtime is None:
                    runtime = _get_runtime_fast(dev_idx)
                runtime.defer_raw_free(workspace_ptr)
        if executor:
            _defer_executor_handle(<uintptr_t>int(executor))
    finally:
        if pta_active:
            _ffi_pta_end_cache_lookup()
        if beta_scalar_owned and beta_scalar:
            _destroy_scalar_fn(int(beta_scalar))
        if alpha_scalar_owned and alpha_scalar:
            _destroy_scalar_fn(int(alpha_scalar))

    return _make_npu_tensor_fast_large(out_ptr, numel, a_dtype, a_dev, out_shape, out_stride, isize, dev_idx, a._dtype_code)


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

    ws_size, executor = _ffi_binary_op_with_alpha(
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
            ret = _ffi_execute(
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

    ws_size, executor = _ffi_binary_op_no_alpha(
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
            ret = _ffi_execute(
                _div_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnDiv execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)

    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


# ---------------------------------------------------------------------------
# In-place arithmetic — hardwired add_/mul_/sub_/div_ that skip aclnn.py
# ---------------------------------------------------------------------------

def fast_add_inplace(a, b):
    """In-place add_(a, b) that calls _ffi.binary_op_with_alpha with output aliased to a."""
    _ensure_npu_imports()
    _ensure_ffi_binary()

    cdef int dev_idx
    _validate_npu_binary(a, b, "add_", &dev_idx)

    a_dtype = a.dtype
    stream = _get_stream_fast(dev_idx)

    py_a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    py_b_shape = (<TensorImpl>b)._shape_tuple if isinstance(b, TensorImpl) else b.shape
    cdef int a_ndim = len(py_a_shape)
    cdef int b_ndim = len(py_b_shape)
    if a_ndim > MAX_NDIM or b_ndim > MAX_NDIM:
        raise ValueError(f"ndim exceeds MAX_NDIM ({MAX_NDIM})")

    cdef int64_t[MAX_NDIM] a_shape_buf, b_shape_buf, out_shape_buf
    _fill_shape(py_a_shape, a_shape_buf, a_ndim)
    _fill_shape(py_b_shape, b_shape_buf, b_ndim)
    cdef int out_ndim
    with nogil:
        out_ndim = c_broadcast_shape(
            a_shape_buf, a_ndim, b_shape_buf, b_ndim, out_shape_buf)
    if out_ndim != a_ndim:
        raise ValueError("NPU add_ requires broadcastable to self shape")
    cdef int i
    for i in range(out_ndim):
        if out_shape_buf[i] != a_shape_buf[i]:
            raise ValueError("NPU add_ requires broadcastable to self shape")

    runtime = _get_runtime_fast(dev_idx)
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t alpha_handle = _get_alpha_one(dtype_code)
    cdef uintptr_t a_ptr = a._storage._untyped._device_ptr
    cdef uintptr_t b_ptr = b._storage._untyped._device_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_binary_op_with_alpha(
        _add_getws_ptr, _add_exec_ptr,
        py_a_shape, a.stride,
        py_b_shape, b.stride,
        py_a_shape, a.stride,
        dtype_code, 2,
        a_ptr, b_ptr, a_ptr,
        alpha_handle,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _add_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnAdd execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def fast_sub_inplace(a, b):
    """In-place sub_(a, b) that calls _ffi.binary_op_with_alpha with output aliased to a."""
    _ensure_npu_imports()
    _ensure_ffi_binary()

    cdef int dev_idx
    _validate_npu_binary(a, b, "sub_", &dev_idx)

    a_dtype = a.dtype
    stream = _get_stream_fast(dev_idx)

    py_a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    py_b_shape = (<TensorImpl>b)._shape_tuple if isinstance(b, TensorImpl) else b.shape
    cdef int a_ndim = len(py_a_shape)
    cdef int b_ndim = len(py_b_shape)
    if a_ndim > MAX_NDIM or b_ndim > MAX_NDIM:
        raise ValueError(f"ndim exceeds MAX_NDIM ({MAX_NDIM})")

    cdef int64_t[MAX_NDIM] a_shape_buf, b_shape_buf, out_shape_buf
    _fill_shape(py_a_shape, a_shape_buf, a_ndim)
    _fill_shape(py_b_shape, b_shape_buf, b_ndim)
    cdef int out_ndim
    with nogil:
        out_ndim = c_broadcast_shape(
            a_shape_buf, a_ndim, b_shape_buf, b_ndim, out_shape_buf)
    if out_ndim != a_ndim:
        raise ValueError("NPU sub_ requires broadcastable to self shape")
    cdef int i
    for i in range(out_ndim):
        if out_shape_buf[i] != a_shape_buf[i]:
            raise ValueError("NPU sub_ requires broadcastable to self shape")

    runtime = _get_runtime_fast(dev_idx)
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t alpha_handle = _get_alpha_one(dtype_code)
    cdef uintptr_t a_ptr = a._storage._untyped._device_ptr
    cdef uintptr_t b_ptr = b._storage._untyped._device_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_binary_op_with_alpha(
        _sub_getws_ptr, _sub_exec_ptr,
        py_a_shape, a.stride,
        py_b_shape, b.stride,
        py_a_shape, a.stride,
        dtype_code, 2,
        a_ptr, b_ptr, a_ptr,
        alpha_handle,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _sub_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnSub execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def fast_sgd_step(a, grad, lr):
    """In-place SGD update: a -= lr * grad using aclnnSub alpha scalar."""
    _ensure_npu_imports()
    _ensure_ffi_binary()
    _ensure_ffi_scalar_helpers()

    cdef int dev_idx
    _validate_npu_binary(a, grad, "sgd_step", &dev_idx)

    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    py_a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    py_g_shape = (<TensorImpl>grad)._shape_tuple if isinstance(grad, TensorImpl) else grad.shape
    py_a_stride = (<TensorImpl>a)._stride_tuple if isinstance(a, TensorImpl) else a.stride
    py_g_stride = (<TensorImpl>grad)._stride_tuple if isinstance(grad, TensorImpl) else grad.stride
    cdef int a_ndim = len(py_a_shape)
    cdef int g_ndim = len(py_g_shape)
    if a_ndim > MAX_NDIM or g_ndim > MAX_NDIM:
        raise ValueError(f"ndim exceeds MAX_NDIM ({MAX_NDIM})")

    cdef int64_t[MAX_NDIM] a_shape_buf, g_shape_buf, out_shape_buf
    _fill_shape(py_a_shape, a_shape_buf, a_ndim)
    _fill_shape(py_g_shape, g_shape_buf, g_ndim)
    cdef int out_ndim
    with nogil:
        out_ndim = c_broadcast_shape(
            a_shape_buf, a_ndim, g_shape_buf, g_ndim, out_shape_buf)
    if out_ndim != a_ndim:
        raise ValueError("NPU sgd_step requires grad broadcastable to param shape")
    cdef int i
    for i in range(out_ndim):
        if out_shape_buf[i] != a_shape_buf[i]:
            raise ValueError("NPU sgd_step requires grad broadcastable to param shape")

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr
    cdef uintptr_t g_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    if isinstance(grad, TensorImpl):
        g_ptr = <uintptr_t>(<TensorImpl>grad)._storage._untyped._device_ptr
    else:
        g_ptr = <uintptr_t>grad.storage().data_ptr()

    cdef object scalar_handle_obj = _create_scalar_fn(_scalar_bytes_fn(float(lr), a_dtype), dtype_code)
    cdef uintptr_t alpha_scalar = <uintptr_t>scalar_handle_obj
    cdef uintptr_t stream_raw = int(stream.stream)
    cdef object ws_size = 0
    cdef object executor = 0
    try:
        ws_size, executor = _ffi_binary_op_with_alpha(
            _sub_getws_ptr, _sub_exec_ptr,
            py_a_shape, py_a_stride,
            py_g_shape, py_g_stride,
            py_a_shape, py_a_stride,
            dtype_code, 2,
            a_ptr, g_ptr, a_ptr,
            alpha_scalar,
            stream_raw)

        if ws_size:
            workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            try:
                ret = _ffi_execute(
                    _sub_exec_ptr, int(workspace_ptr), ws_size,
                    executor, stream_raw)
                if ret != 0:
                    raise RuntimeError(f"aclnnSub execute failed: {ret}")
            finally:
                runtime.defer_raw_free(workspace_ptr)

        _defer_executor_fn(executor)
    finally:
        _destroy_scalar_fn(int(alpha_scalar))
    return a



def fast_sgd_step_many(params, grads, lr):
    """Batch in-place SGD updates for a parameter group."""
    cdef Py_ssize_t n = len(params)
    if n != len(grads):
        raise ValueError("params and grads must have the same length")
    cdef Py_ssize_t i
    for i in range(n):
        fast_sgd_step(params[i], grads[i], lr)
    return None



def fast_mul_inplace(a, b):
    """In-place mul_(a, b) that calls _ffi.binary_op_no_alpha with output aliased to a."""
    _ensure_npu_imports()
    _ensure_ffi_binary()

    cdef int dev_idx
    _validate_npu_binary(a, b, "mul_", &dev_idx)

    a_dtype = a.dtype
    stream = _get_stream_fast(dev_idx)

    py_a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    py_b_shape = (<TensorImpl>b)._shape_tuple if isinstance(b, TensorImpl) else b.shape
    cdef int a_ndim = len(py_a_shape)
    cdef int b_ndim = len(py_b_shape)
    if a_ndim > MAX_NDIM or b_ndim > MAX_NDIM:
        raise ValueError(f"ndim exceeds MAX_NDIM ({MAX_NDIM})")

    cdef int64_t[MAX_NDIM] a_shape_buf, b_shape_buf, out_shape_buf
    _fill_shape(py_a_shape, a_shape_buf, a_ndim)
    _fill_shape(py_b_shape, b_shape_buf, b_ndim)
    cdef int out_ndim
    with nogil:
        out_ndim = c_broadcast_shape(
            a_shape_buf, a_ndim, b_shape_buf, b_ndim, out_shape_buf)
    if out_ndim != a_ndim:
        raise ValueError("NPU mul_ requires broadcastable to self shape")
    cdef int i
    for i in range(out_ndim):
        if out_shape_buf[i] != a_shape_buf[i]:
            raise ValueError("NPU mul_ requires broadcastable to self shape")

    runtime = _get_runtime_fast(dev_idx)
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr = a._storage._untyped._device_ptr
    cdef uintptr_t b_ptr = b._storage._untyped._device_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_binary_op_no_alpha(
        _mul_getws_ptr, _mul_exec_ptr,
        py_a_shape, a.stride,
        py_b_shape, b.stride,
        py_a_shape, a.stride,
        dtype_code, 2,
        a_ptr, b_ptr, a_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _mul_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnMul execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def fast_mul_scalar(a, value):
    """NPU tensor * Python scalar via ACLNN Muls without host-to-device tensor scalar copy."""
    _ensure_npu_imports()

    cdef int dev_idx
    cdef TensorImpl source
    if not isinstance(a, TensorImpl) or (<TensorImpl>a)._device_type != 1:
        raise ValueError("NPU mul scalar expects an NPU tensor")
    dev_idx = (<TensorImpl>a)._device_index
    if dev_idx < 0:
        dev_idx = 0

    source = <TensorImpl>a
    py_shape = source._shape_tuple
    cdef int ndim = len(py_shape)
    if ndim > MAX_NDIM:
        raise ValueError(f"ndim exceeds MAX_NDIM ({MAX_NDIM})")

    cdef int64_t[MAX_NDIM] shape_buf, stride_buf
    _fill_shape(py_shape, shape_buf, ndim)
    with nogil:
        c_contiguous_stride(shape_buf, ndim, stride_buf)
    out_stride = _to_tuple(stride_buf, ndim)
    if source._stride_tuple != out_stride:
        source = <TensorImpl>a.contiguous()

    cdef int64_t n = source._c_numel
    if n < 1:
        n = 1
    cdef int itemsize = source._itemsize
    cdef int64_t alloc_size = n * itemsize
    stream = _get_stream_fast(dev_idx)
    cdef uintptr_t stream_raw = int(stream.stream)
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size, stream=stream.stream)

    from candle._backends.npu import aclnn as _aclnn
    runtime = _get_runtime_fast(dev_idx)
    _aclnn.mul_scalar(
        <uintptr_t>source._storage._untyped._device_ptr,
        value,
        <uintptr_t>out_ptr,
        source._shape_tuple,
        source._stride_tuple,
        source.dtype,
        runtime,
        stream=stream_raw,
    )
    return _make_npu_tensor_fast(
        <int64_t>out_ptr,
        source._c_numel,
        source.dtype,
        source._device_obj,
        source._shape_tuple,
        out_stride,
        itemsize,
    )


def fast_div_inplace(a, b):
    """In-place div_(a, b) that calls _ffi.binary_op_no_alpha with output aliased to a."""
    _ensure_npu_imports()
    _ensure_ffi_binary()

    cdef int dev_idx
    _validate_npu_binary(a, b, "div_", &dev_idx)

    a_dtype = a.dtype
    stream = _get_stream_fast(dev_idx)

    py_a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    py_b_shape = (<TensorImpl>b)._shape_tuple if isinstance(b, TensorImpl) else b.shape
    cdef int a_ndim = len(py_a_shape)
    cdef int b_ndim = len(py_b_shape)
    if a_ndim > MAX_NDIM or b_ndim > MAX_NDIM:
        raise ValueError(f"ndim exceeds MAX_NDIM ({MAX_NDIM})")

    cdef int64_t[MAX_NDIM] a_shape_buf, b_shape_buf, out_shape_buf
    _fill_shape(py_a_shape, a_shape_buf, a_ndim)
    _fill_shape(py_b_shape, b_shape_buf, b_ndim)
    cdef int out_ndim
    with nogil:
        out_ndim = c_broadcast_shape(
            a_shape_buf, a_ndim, b_shape_buf, b_ndim, out_shape_buf)
    if out_ndim != a_ndim:
        raise ValueError("NPU div_ requires broadcastable to self shape")
    cdef int i
    for i in range(out_ndim):
        if out_shape_buf[i] != a_shape_buf[i]:
            raise ValueError("NPU div_ requires broadcastable to self shape")

    runtime = _get_runtime_fast(dev_idx)
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr = a._storage._untyped._device_ptr
    cdef uintptr_t b_ptr = b._storage._untyped._device_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_binary_op_no_alpha(
        _div_getws_ptr, _div_exec_ptr,
        py_a_shape, a.stride,
        py_b_shape, b.stride,
        py_a_shape, a.stride,
        dtype_code, 2,
        a_ptr, b_ptr, a_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _div_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnDiv execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def fast_bitwise_and_inplace(a, b):
    """In-place bitwise_and_(a, b) — aliases output ptr to a via aclnnBitwiseAndTensor."""
    return _fast_bitwise_inplace_dispatch(a, b, "bitwise_and_",
                                          _bitwise_and_getws_ptr,
                                          _bitwise_and_exec_ptr,
                                          "aclnnBitwiseAndTensor")


def fast_bitwise_or_inplace(a, b):
    """In-place bitwise_or_(a, b) — aliases output ptr to a via aclnnBitwiseOrTensor."""
    return _fast_bitwise_inplace_dispatch(a, b, "bitwise_or_",
                                          _bitwise_or_getws_ptr,
                                          _bitwise_or_exec_ptr,
                                          "aclnnBitwiseOrTensor")


def fast_bitwise_xor_inplace(a, b):
    """In-place bitwise_xor_(a, b) — aliases output ptr to a via aclnnBitwiseXorTensor."""
    return _fast_bitwise_inplace_dispatch(a, b, "bitwise_xor_",
                                          _bitwise_xor_getws_ptr,
                                          _bitwise_xor_exec_ptr,
                                          "aclnnBitwiseXorTensor")


cdef object _fast_bitwise_inplace_dispatch(object a, object b, str name,
                                            object getws_ptr_unused,
                                            object exec_ptr_unused,
                                            str pretty):
    """Common in-place bitwise tensor op (a = a op b) reusing the existing
    out-of-place aclnnBitwise<And|Or|Xor>Tensor with output ptr aliased to a."""
    _ensure_npu_imports()
    _ensure_ffi_bitwise()

    cdef int dev_idx
    _validate_npu_binary(a, b, name, &dev_idx)

    a_dtype = a.dtype
    stream = _get_stream_fast(dev_idx)

    py_a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    py_b_shape = (<TensorImpl>b)._shape_tuple if isinstance(b, TensorImpl) else b.shape
    cdef int a_ndim = len(py_a_shape)
    cdef int b_ndim = len(py_b_shape)
    if a_ndim > MAX_NDIM or b_ndim > MAX_NDIM:
        raise ValueError(f"ndim exceeds MAX_NDIM ({MAX_NDIM})")

    cdef int64_t[MAX_NDIM] a_shape_buf, b_shape_buf, out_shape_buf
    _fill_shape(py_a_shape, a_shape_buf, a_ndim)
    _fill_shape(py_b_shape, b_shape_buf, b_ndim)
    cdef int out_ndim
    with nogil:
        out_ndim = c_broadcast_shape(
            a_shape_buf, a_ndim, b_shape_buf, b_ndim, out_shape_buf)
    if out_ndim != a_ndim:
        raise ValueError(f"NPU {name} requires broadcastable to self shape")
    cdef int i
    for i in range(out_ndim):
        if out_shape_buf[i] != a_shape_buf[i]:
            raise ValueError(f"NPU {name} requires broadcastable to self shape")

    runtime = _get_runtime_fast(dev_idx)
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr = a._storage._untyped._device_ptr
    cdef uintptr_t b_ptr = b._storage._untyped._device_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    if name == "bitwise_and_":
        getws_ptr = _bitwise_and_getws_ptr
        exec_ptr = _bitwise_and_exec_ptr
    elif name == "bitwise_or_":
        getws_ptr = _bitwise_or_getws_ptr
        exec_ptr = _bitwise_or_exec_ptr
    else:
        getws_ptr = _bitwise_xor_getws_ptr
        exec_ptr = _bitwise_xor_exec_ptr

    ws_size, executor = _ffi_ref.binary_two_inputs_op(
        getws_ptr, exec_ptr,
        py_a_shape, a.stride,
        py_b_shape, b.stride,
        py_a_shape, a.stride,
        dtype_code, dtype_code, dtype_code,
        2,
        int(a_ptr), int(b_ptr), int(a_ptr),
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"{pretty} execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def fast_pow(a, b):
    return fast_binary_op(a, b, None, "pow")


def fast_pow_tensor_scalar(a, exponent):
    return fast_pow(a, _npu_scalar_like(exponent, a))


def fast_pow_inplace(a, b):
    """In-place pow_(a, b) — aliases output ptr to a via aclnnPowTensorTensor.

    Reuses the existing out-of-place aclnnPowTensorTensor with the output
    pointer aliased to the `a` pointer. Caller must ensure b is broadcastable
    to a's shape.
    """
    _ensure_npu_imports()
    _ensure_ffi_pow()

    cdef int dev_idx
    _validate_npu_binary(a, b, "pow_", &dev_idx)

    a_dtype = a.dtype
    stream = _get_stream_fast(dev_idx)

    py_a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    py_b_shape = (<TensorImpl>b)._shape_tuple if isinstance(b, TensorImpl) else b.shape
    cdef int a_ndim = len(py_a_shape)
    cdef int b_ndim = len(py_b_shape)
    if a_ndim > MAX_NDIM or b_ndim > MAX_NDIM:
        raise ValueError(f"ndim exceeds MAX_NDIM ({MAX_NDIM})")

    cdef int64_t[MAX_NDIM] a_shape_buf, b_shape_buf, out_shape_buf
    _fill_shape(py_a_shape, a_shape_buf, a_ndim)
    _fill_shape(py_b_shape, b_shape_buf, b_ndim)
    cdef int out_ndim
    with nogil:
        out_ndim = c_broadcast_shape(
            a_shape_buf, a_ndim, b_shape_buf, b_ndim, out_shape_buf)
    if out_ndim != a_ndim:
        raise ValueError("NPU pow_ requires broadcastable to self shape")
    cdef int i
    for i in range(out_ndim):
        if out_shape_buf[i] != a_shape_buf[i]:
            raise ValueError("NPU pow_ requires broadcastable to self shape")

    runtime = _get_runtime_fast(dev_idx)
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr = a._storage._untyped._device_ptr
    cdef uintptr_t b_ptr = b._storage._untyped._device_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.binary_two_inputs_op(
        _pow_getws_ptr, _pow_exec_ptr,
        py_a_shape, a.stride,
        py_b_shape, b.stride,
        py_a_shape, a.stride,
        dtype_code, dtype_code, dtype_code,
        2,
        int(a_ptr), int(b_ptr), int(a_ptr),
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _pow_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnPowTensorTensor execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def fast_floor_divide(a, b):
    return fast_binary_op(a, b, None, "floor_divide")


def fast_logaddexp(a, b):
    return fast_binary_op(a, b, None, "logaddexp")


def fast_logaddexp2(a, b):
    return fast_binary_op(a, b, None, "logaddexp2")


def fast_atan2(a, b):
    return fast_binary_op(a, b, None, "atan2")


def fast_fmod(a, b):
    return fast_binary_op(a, b, None, "fmod")


def fast_remainder(a, b):
    return fast_binary_op(a, b, None, "remainder")


def fast_maximum(a, b):
    return fast_binary_op(a, b, None, "maximum")


def fast_minimum(a, b):
    return fast_binary_op(a, b, None, "minimum")


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
cdef object _clamp_getws_ptr = None      # cached Clamp getws pointer
cdef object _clamp_exec_ptr = None       # cached Clamp exec pointer
cdef object _inplace_copy_getws_ptr = None  # cached InplaceCopy getws pointer
cdef object _inplace_copy_exec_ptr = None   # cached InplaceCopy exec pointer
cdef object _gelu_getws_ptr = None       # cached Gelu getws pointer
cdef object _gelu_exec_ptr = None        # cached Gelu exec pointer
cdef object _gelu_backward_getws_ptr = None  # cached GeluBackward getws pointer
cdef object _gelu_backward_exec_ptr = None   # cached GeluBackward exec pointer
cdef object _silu_getws_ptr = None       # cached Silu getws pointer
cdef object _silu_exec_ptr = None        # cached Silu exec pointer
cdef object _silu_backward_getws_ptr = None  # cached SiluBackward getws pointer
cdef object _silu_backward_exec_ptr = None   # cached SiluBackward exec pointer
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
cdef object _trace_getws_ptr = None      # cached Trace getws pointer
cdef object _trace_exec_ptr = None       # cached Trace exec pointer
cdef object _inverse_getws_ptr = None    # cached Inverse getws pointer
cdef object _inverse_exec_ptr = None     # cached Inverse exec pointer


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


cdef inline void _ensure_ffi_addmm() except *:
    global _ffi_ref, _addmm_getws_ptr, _addmm_exec_ptr
    if _addmm_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _addmm_getws_ptr, _addmm_exec_ptr = _ffi_ref.resolve_op("Addmm")
    _ensure_ffi_scalar_helpers()


cdef inline void _ensure_ffi_reduce_sum() except *:
    global _ffi_ref, _reduce_sum_getws_ptr, _reduce_sum_exec_ptr
    if _reduce_sum_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _reduce_sum_getws_ptr, _reduce_sum_exec_ptr = _ffi_ref.resolve_op("ReduceSum")


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


cdef inline void _ensure_ffi_trace() except *:
    global _ffi_ref, _trace_getws_ptr, _trace_exec_ptr
    if _trace_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _trace_getws_ptr, _trace_exec_ptr = _ffi_ref.resolve_op("Trace")


cdef inline void _ensure_ffi_inverse() except *:
    global _ffi_ref, _inverse_getws_ptr, _inverse_exec_ptr
    if _inverse_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _inverse_getws_ptr, _inverse_exec_ptr = _ffi_ref.resolve_op("Inverse")




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


cdef inline void _ensure_ffi_clamp() except *:
    global _ffi_ref, _clamp_getws_ptr, _clamp_exec_ptr
    if _clamp_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _clamp_getws_ptr, _clamp_exec_ptr = _ffi_ref.resolve_op("Clamp")
    _ensure_ffi_scalar_helpers()


cdef inline void _ensure_ffi_inplace_copy() except *:
    global _ffi_ref, _inplace_copy_getws_ptr, _inplace_copy_exec_ptr
    if _inplace_copy_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _inplace_copy_getws_ptr, _inplace_copy_exec_ptr = _ffi_ref.resolve_op("InplaceCopy")


cdef inline void _ensure_ffi_gelu() except *:
    global _ffi_ref, _gelu_getws_ptr, _gelu_exec_ptr
    if _gelu_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _gelu_getws_ptr, _gelu_exec_ptr = _ffi_ref.resolve_op("Gelu")


cdef inline void _ensure_ffi_gelu_backward() except *:
    global _ffi_ref, _gelu_backward_getws_ptr, _gelu_backward_exec_ptr
    if _gelu_backward_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _gelu_backward_getws_ptr, _gelu_backward_exec_ptr = _ffi_ref.resolve_op("GeluBackward")


cdef inline void _ensure_ffi_silu() except *:
    global _ffi_ref, _silu_getws_ptr, _silu_exec_ptr
    if _silu_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _silu_getws_ptr, _silu_exec_ptr = _ffi_ref.resolve_op("Silu")


cdef inline void _ensure_ffi_silu_backward() except *:
    global _ffi_ref, _silu_backward_getws_ptr, _silu_backward_exec_ptr
    if _silu_backward_getws_ptr is not None:
        return
    if _ffi_ref is None:
        _ensure_ffi_binary()
    _silu_backward_getws_ptr, _silu_backward_exec_ptr = _ffi_ref.resolve_op("SiluBackward")


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

    if isinstance(a, TensorImpl):
        a_dev = (<TensorImpl>a)._device_obj
        a_dtype = (<TensorImpl>a)._dtype_obj
    else:
        a_dev = a.device
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
            ret = _ffi_execute(
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
            ret = _ffi_execute(
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

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
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

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
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

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
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

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
                _abs_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnAbs execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_abs_inplace(a):
    """In-place abs_(a) using aclnnAbs with output aliased to input."""
    _ensure_npu_imports()
    _ensure_ffi_abs()

    cdef int dev_idx = a.device.index or 0
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    a_stride = a.stride
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_unary_op(
        _abs_getws_ptr, _abs_exec_ptr,
        a_shape, a_stride,
        a_shape, a_stride,
        dtype_code, dtype_code, 2,
        a_ptr, a_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _abs_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnAbs execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def _npu_contiguous_copy(a):
    from candle._backends.npu.ops.shape import contiguous
    return contiguous(a)


def fast_neg(a):
    """Optimized out-of-place neg(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_neg()

    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)
    stream_obj = stream.stream

    out_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    out_stride = _npu_runtime._contiguous_stride(out_shape)
    cdef int64_t n = 1
    for dim in out_shape:
        n *= dim

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_fneg = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fneg, stream=stream_obj)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fneg, stream=stream_obj)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream_obj)
    cdef object src_for_neg = a
    cdef object src_shape = out_shape
    cdef object src_stride = a.stride
    cdef object workspace_ptr
    cdef int ret
    cdef int64_t temp_ptr = 0
    cdef bint temp_allocated = False
    cdef object temp_allocator = None

    try:
        try:
            if isinstance(a, TensorImpl) and _can_use_native_offset_unary_descriptor(<TensorImpl>a):
                # torch_npu passes aclCreateTensor the base storage pointer plus the
                # logical view's storage_offset/storage size.  CANN accepts that
                # descriptor for RoPE-style half views; passing an already adjusted
                # effective pointer with offset=0 is what triggers 561103.
                a_ptr = _npu_storage_base_ptr_fast(a)
                ws_size, executor = _ffi_unary_op_with_input_storage(
                    <uintptr_t>_neg_getws_ptr, <uintptr_t>_neg_exec_ptr,
                    src_shape, src_stride,
                    (_npu_storage_numel_fast(a, isize),), (<TensorImpl>a)._c_offset,
                    out_shape, out_stride,
                    dtype_code, dtype_code, 2,
                    a_ptr, o_ptr,
                    stream_raw)
            else:
                a_ptr = _npu_data_ptr_fast(a, isize)
                ws_size, executor = _ffi_unary_op(
                    _neg_getws_ptr, _neg_exec_ptr,
                    src_shape, src_stride,
                    out_shape, out_stride,
                    dtype_code, dtype_code, 2,
                    a_ptr, o_ptr,
                    stream_raw)
        except RuntimeError as exc:
            if "GetWorkspaceSize failed: 561103" not in str(exc):
                _get_allocator_fn_ref(dev_idx).free(out_ptr, stream=stream_obj)
                raise
            # TODO: re-enable native strided-view neg when CANN fixes aclnnNeg 561103.
            src_for_neg = _npu_contiguous_copy(a)
            if isinstance(src_for_neg, TensorImpl):
                a_ptr = _npu_data_ptr_fast(src_for_neg, isize)
                src_shape = (<TensorImpl>src_for_neg)._shape_tuple
            else:
                a_ptr = _npu_data_ptr_fast(src_for_neg, isize)
                src_shape = src_for_neg.shape
            src_stride = src_for_neg.stride
            ws_size, executor = _ffi_unary_op(
                _neg_getws_ptr, _neg_exec_ptr,
                src_shape, src_stride,
                out_shape, out_stride,
                dtype_code, dtype_code, 2,
                a_ptr, o_ptr,
                stream_raw)

        if ws_size:
            workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            try:
                ret = _ffi_execute(
                    _neg_exec_ptr, int(workspace_ptr), ws_size,
                    executor, stream_raw)
                if ret != 0:
                    raise RuntimeError(f"aclnnNeg execute failed: {ret}")
            finally:
                runtime.defer_raw_free(workspace_ptr)

        _defer_executor_fn(executor)
        return _make_npu_tensor_fast(out_ptr, n, a_dtype, a_dev, out_shape, out_stride, isize)
    finally:
        if temp_allocated:
            temp_allocator.free(temp_ptr, stream=stream_obj)


def fast_neg_inplace(a):
    """In-place neg_(a) using aclnnNeg with output aliased to input."""
    _ensure_npu_imports()
    _ensure_ffi_neg()

    cdef int dev_idx = a.device.index or 0
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    a_stride = a.stride
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_unary_op(
        _neg_getws_ptr, _neg_exec_ptr,
        a_shape, a_stride,
        a_shape, a_stride,
        dtype_code, dtype_code, 2,
        a_ptr, a_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _neg_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnNeg execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def fast_sign(a):
    """Optimized out-of-place sign(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_sign()

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

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
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

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
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

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
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

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
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

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
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

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
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

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
                _bitwise_not_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnBitwiseNot execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_bitwise_not_inplace(a):
    """In-place bitwise_not_(a) using aclnnBitwiseNot with output aliased to input."""
    _ensure_npu_imports()
    _ensure_ffi_bitwise_not()

    cdef int dev_idx = a.device.index or 0
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    a_stride = a.stride
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_unary_op(
        _bitwise_not_getws_ptr, _bitwise_not_exec_ptr,
        a_shape, a_stride,
        a_shape, a_stride,
        dtype_code, dtype_code, 2,
        a_ptr, a_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _bitwise_not_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnBitwiseNot execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def fast_square(a):
    """Optimized out-of-place square(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_square()

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

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
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

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
                _exp_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnExp execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_exp_inplace(a):
    """In-place exp_(a) using aclnnExp with output aliased to input."""
    _ensure_npu_imports()
    _ensure_ffi_exp()

    cdef int dev_idx = a.device.index or 0
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    a_stride = a.stride
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_unary_op(
        _exp_getws_ptr, _exp_exec_ptr,
        a_shape, a_stride,
        a_shape, a_stride,
        dtype_code, dtype_code, 2,
        a_ptr, a_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _exp_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnExp execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def fast_expm1(a):
    """Optimized out-of-place expm1(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_expm1()

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

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
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

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
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

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
                _log_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnLog execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_log_inplace(a):
    """In-place log_(a) using aclnnLog with output aliased to input."""
    _ensure_npu_imports()
    _ensure_ffi_log()

    cdef int dev_idx = a.device.index or 0
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    a_stride = a.stride
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_unary_op(
        _log_getws_ptr, _log_exec_ptr,
        a_shape, a_stride,
        a_shape, a_stride,
        dtype_code, dtype_code, 2,
        a_ptr, a_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _log_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnLog execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


cpdef object fast_rsqrt(object a):
    """Optimized out-of-place rsqrt(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_rsqrt()

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

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
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

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
                _sqrt_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnSqrt execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_sqrt_inplace(a):
    """In-place sqrt_(a) using aclnnSqrt with output aliased to input."""
    _ensure_npu_imports()
    _ensure_ffi_sqrt()

    cdef int dev_idx = a.device.index or 0
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    a_stride = a.stride
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_unary_op(
        _sqrt_getws_ptr, _sqrt_exec_ptr,
        a_shape, a_stride,
        a_shape, a_stride,
        dtype_code, dtype_code, 2,
        a_ptr, a_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _sqrt_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnSqrt execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def fast_hypot(a, b):
    """NPU hypot(a, b) implemented as an on-device Cython composite."""
    return fast_sqrt(fast_add(fast_mul(a, a), fast_mul(b, b)))


def fast_fmin(a, b):
    """NPU fmin(a, b) routed through aclnnMinimum.

    Note: PyTorch fmin differs from min in NaN handling (fmin(x, NaN) = x).
    We currently treat fmin as minimum because the previous SWhere-based
    composite segfaults on 310B; NaN-aware semantics will be revisited.
    """
    return fast_binary_op(a, b, None, "minimum")


def fast_fmax(a, b):
    """NPU fmax(a, b) routed through aclnnMaximum.

    Note: PyTorch fmax differs from max in NaN handling (fmax(x, NaN) = x).
    We currently treat fmax as maximum because the previous SWhere-based
    composite segfaults on 310B; NaN-aware semantics will be revisited.
    """
    return fast_binary_op(a, b, None, "maximum")


def _npu_scalar_like(value, a):
    from candle._backends.npu.ops._helpers import _scalar_to_npu_tensor
    return _scalar_to_npu_tensor(value, a)


def fast_relu6(a):
    zero = _npu_scalar_like(0.0, a)
    six = _npu_scalar_like(6.0, a)
    return fast_binary_op(fast_binary_op(a, zero, None, "maximum"), six, None, "minimum")


def fast_selu(a):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return fast_mul(fast_elu(a, alpha), _npu_scalar_like(scale, a))


def fast_celu(a, alpha=1.0):
    inv_alpha = _npu_scalar_like(1.0 / alpha, a)
    alpha_t = _npu_scalar_like(alpha, a)
    one = _npu_scalar_like(1.0, a)
    zero = _npu_scalar_like(0.0, a)
    exp_part = fast_sub(fast_exp(fast_mul(a, inv_alpha)), one)
    neg_part = fast_mul(alpha_t, fast_binary_op(exp_part, zero, None, "minimum"))
    pos_part = fast_binary_op(a, zero, None, "maximum")
    return fast_add(pos_part, neg_part)


def fast_threshold(a, threshold_val, value):
    thresh_t = _npu_scalar_like(threshold_val, a)
    value_t = _npu_scalar_like(value, a)
    return fast_where(fast_binary_op(a, thresh_t, None, "gt"), a, value_t)


def fast_hardshrink(a, lambd=0.5):
    zero = _npu_scalar_like(0.0, a)
    lambd_t = _npu_scalar_like(lambd, a)
    return fast_where(fast_binary_op(fast_abs(a), lambd_t, None, "gt"), a, zero)


def fast_softshrink(a, lambd=0.5):
    zero = _npu_scalar_like(0.0, a)
    lambd_t = _npu_scalar_like(lambd, a)
    neg_lambd_t = _npu_scalar_like(-lambd, a)
    pos_mask = fast_binary_op(a, lambd_t, None, "gt")
    neg_mask = fast_binary_op(a, neg_lambd_t, None, "lt")
    result = fast_where(pos_mask, fast_sub(a, lambd_t), zero)
    return fast_where(neg_mask, fast_add(a, lambd_t), result)


def _fast_clamp06(t):
    zero = _npu_scalar_like(0.0, t)
    six = _npu_scalar_like(6.0, t)
    return fast_binary_op(fast_binary_op(t, zero, None, "maximum"), six, None, "minimum")


def fast_hardswish(a):
    three = _npu_scalar_like(3.0, a)
    six = _npu_scalar_like(6.0, a)
    return fast_div(fast_mul(a, _fast_clamp06(fast_add(a, three))), six)


def fast_hardsigmoid(a):
    three = _npu_scalar_like(3.0, a)
    six = _npu_scalar_like(6.0, a)
    return fast_div(_fast_clamp06(fast_add(a, three)), six)


def fast_softsign(a):
    one = _npu_scalar_like(1.0, a)
    return fast_div(a, fast_add(one, fast_abs(a)))


def fast_rrelu(a, lower=0.125, upper=0.3333333333333333, training=False):
    zero = _npu_scalar_like(0.0, a)
    slope_t = _npu_scalar_like((lower + upper) / 2.0, a)
    return fast_where(fast_binary_op(a, zero, None, "gt"), a, fast_mul(a, slope_t))


def fast_frac(a):
    return fast_add(a, fast_neg(fast_trunc(a)))


def fast_reciprocal(a):
    return fast_div(_npu_scalar_like(1.0, a), a)


def fast_isinf(a):
    finite = fast_isfinite(a)
    recip_finite = fast_isfinite(fast_reciprocal(a))
    return fast_binary_op(fast_logical_not(finite), recip_finite, None, "logical_and")


def fast_isnan(a):
    finite = fast_isfinite(a)
    recip_finite = fast_isfinite(fast_reciprocal(a))
    return fast_binary_op(fast_logical_not(finite), fast_logical_not(recip_finite), None, "logical_and")


def fast_isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    diff = fast_abs(fast_sub(a, b))
    tol = fast_add(_npu_scalar_like(float(atol), diff), fast_mul(_npu_scalar_like(float(rtol), diff), fast_abs(b)))
    close = fast_binary_op(diff, tol, None, "le")
    if equal_nan:
        nan_both = fast_binary_op(fast_isnan(a), fast_isnan(b), None, "logical_and")
        return fast_binary_op(close, nan_both, None, "logical_or")
    nan_any = fast_binary_op(fast_isnan(a), fast_isnan(b), None, "logical_or")
    return fast_binary_op(close, fast_logical_not(nan_any), None, "logical_and")


def fast_special_sinc(a):
    import math
    pi_a = fast_mul(a, _npu_scalar_like(math.pi, a))
    return fast_where(fast_binary_op(a, _npu_scalar_like(0.0, a), None, "eq"),
                      _npu_scalar_like(1.0, a), fast_div(fast_sin(pi_a), pi_a))


def fast_special_erfcx(a):
    return fast_mul(fast_exp(fast_mul(a, a)), fast_erfc(a))


def fast_special_logit(a, eps=None):
    one = _npu_scalar_like(1.0, a)
    if eps is not None:
        lo = _npu_scalar_like(float(eps), a)
        hi = _npu_scalar_like(1.0 - float(eps), a)
        a = fast_binary_op(fast_binary_op(a, lo, None, "maximum"), hi, None, "minimum")
    return fast_log(fast_div(a, fast_sub(one, a)))


def fast_special_ndtr(a):
    import math
    return fast_mul(_npu_scalar_like(0.5, a), fast_erfc(fast_mul(a, _npu_scalar_like(-1.0 / math.sqrt(2.0), a))))


def fast_special_log_ndtr(a):
    return fast_log(fast_special_ndtr(a))


def fast_special_xlogy(a, b):
    zero = _npu_scalar_like(0.0, a)
    result = fast_mul(a, fast_log(fast_binary_op(b, _npu_scalar_like(1e-38, b), None, "maximum")))
    return fast_where(fast_binary_op(a, zero, None, "eq"), zero, result)


def fast_special_xlog1py(a, b):
    zero = _npu_scalar_like(0.0, a)
    result = fast_mul(a, fast_log1p(b))
    return fast_where(fast_binary_op(a, zero, None, "eq"), zero, result)


def fast_sin(a):
    """Optimized out-of-place sin(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_sin()

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

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
                _sin_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnSin execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_sin_inplace(a):
    """In-place sin_(a) using aclnnSin with output aliased to input."""
    _ensure_npu_imports()
    _ensure_ffi_sin()

    cdef int dev_idx = a.device.index or 0
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    a_stride = a.stride
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_unary_op(
        _sin_getws_ptr, _sin_exec_ptr,
        a_shape, a_stride,
        a_shape, a_stride,
        dtype_code, dtype_code, 2,
        a_ptr, a_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _sin_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnSin execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def fast_cos(a):
    """Optimized out-of-place cos(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_cos()

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

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
                _cos_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnCos execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_cos_inplace(a):
    """In-place cos_(a) using aclnnCos with output aliased to input."""
    _ensure_npu_imports()
    _ensure_ffi_cos()

    cdef int dev_idx = a.device.index or 0
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    a_stride = a.stride
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_unary_op(
        _cos_getws_ptr, _cos_exec_ptr,
        a_shape, a_stride,
        a_shape, a_stride,
        dtype_code, dtype_code, 2,
        a_ptr, a_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _cos_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnCos execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def fast_tan(a):
    """Optimized out-of-place tan(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_tan()

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

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
                _tan_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnTan execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_tan_inplace(a):
    """In-place tan_(a) using aclnnTan with output aliased to input."""
    _ensure_npu_imports()
    _ensure_ffi_tan()

    cdef int dev_idx = a.device.index or 0
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    a_stride = a.stride
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_unary_op(
        _tan_getws_ptr, _tan_exec_ptr,
        a_shape, a_stride,
        a_shape, a_stride,
        dtype_code, dtype_code, 2,
        a_ptr, a_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _tan_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnTan execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def fast_tanh(a):
    """Optimized out-of-place tanh(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_tanh()

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

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
                _tanh_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnTanh execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)



def fast_tanh_inplace(a):
    """In-place tanh_(a) using aclnnTanh with output aliased to input."""
    _ensure_npu_imports()
    _ensure_ffi_tanh()

    cdef int dev_idx = a.device.index or 0
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    a_stride = a.stride
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_unary_op(
        _tanh_getws_ptr, _tanh_exec_ptr,
        a_shape, a_stride,
        a_shape, a_stride,
        dtype_code, dtype_code, 2,
        a_ptr, a_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _tanh_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnTanh execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def fast_sigmoid(a):
    """Optimized out-of-place sigmoid(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_sigmoid()

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

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
                _sigmoid_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnSigmoid execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_sigmoid_inplace(a):
    """In-place sigmoid_(a) using aclnnSigmoid with output aliased to input."""
    _ensure_npu_imports()
    _ensure_ffi_sigmoid()

    cdef int dev_idx = a.device.index or 0
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    a_stride = a.stride
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_unary_op(
        _sigmoid_getws_ptr, _sigmoid_exec_ptr,
        a_shape, a_stride,
        a_shape, a_stride,
        dtype_code, dtype_code, 2,
        a_ptr, a_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _sigmoid_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnSigmoid execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def fast_relu(a):
    """Optimized out-of-place relu(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_relu()

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

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
                _relu_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnRelu execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_relu_inplace(a):
    """In-place relu_(a) using aclnnRelu with output aliased to input."""
    _ensure_npu_imports()
    _ensure_ffi_relu()

    cdef int dev_idx = a.device.index or 0
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    a_stride = a.stride
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_unary_op(
        _relu_getws_ptr, _relu_exec_ptr,
        a_shape, a_stride,
        a_shape, a_stride,
        dtype_code, dtype_code, 2,
        a_ptr, a_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _relu_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnRelu execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def fast_leaky_relu(a, negative_slope):
    """Optimized out-of-place leaky_relu(a) that calls _ffi.leaky_relu_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_leaky_relu()

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
                ret = _ffi_execute(
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
                ret = _ffi_execute(
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


def fast_leaky_relu_inplace(a, negative_slope):
    """In-place leaky_relu_(a, negative_slope) using aclnnLeakyRelu with output aliased to input."""
    _ensure_npu_imports()
    _ensure_ffi_leaky_relu()

    cdef int dev_idx = a.device.index or 0
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    a_stride = a.stride
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()

    scalar_handle = _create_scalar_fn(_scalar_bytes_fn(negative_slope, a_dtype), dtype_code)
    cdef uintptr_t stream_raw = int(stream.stream)
    try:
        ws_size, executor = _ffi_ref.leaky_relu_op(
            _leaky_relu_getws_ptr, _leaky_relu_exec_ptr,
            a_shape, a_stride,
            a_shape, a_stride,
            dtype_code, 2,
            a_ptr, a_ptr,
            scalar_handle,
            stream_raw)

        if ws_size:
            workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            try:
                ret = _ffi_execute(
                    _leaky_relu_exec_ptr, int(workspace_ptr), ws_size,
                    executor, stream_raw)
                if ret != 0:
                    raise RuntimeError(f"aclnnLeakyRelu execute failed: {ret}")
            finally:
                runtime.defer_raw_free(workspace_ptr)

        _defer_executor_fn(executor)
    finally:
        _destroy_scalar_fn(int(scalar_handle))

    return a


def fast_elu_inplace(a, alpha):
    """In-place elu_(a, alpha) using aclnnElu with output aliased to input."""
    _ensure_npu_imports()
    _ensure_ffi_elu()

    cdef int dev_idx = a.device.index or 0
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    a_stride = a.stride
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()

    alpha_scalar = _create_scalar_fn(_scalar_bytes_fn(alpha, a_dtype), dtype_code)
    scale_scalar = _create_scalar_fn(_scalar_bytes_fn(1.0, a_dtype), dtype_code)
    input_scale_scalar = _create_scalar_fn(_scalar_bytes_fn(1.0, a_dtype), dtype_code)
    cdef uintptr_t stream_raw = int(stream.stream)
    try:
        ws_size, executor = _ffi_ref.tensor_three_scalars_op(
            _elu_getws_ptr, _elu_exec_ptr,
            a_shape, a_stride,
            a_shape, a_stride,
            dtype_code, dtype_code, 2,
            a_ptr, a_ptr,
            alpha_scalar, scale_scalar, input_scale_scalar,
            stream_raw)

        if ws_size:
            workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            try:
                ret = _ffi_execute(
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

    return a


def fast_clamp(a, min_val=None, max_val=None):
    """Optimized clamp(a, min_val, max_val) that calls _ffi.clamp_optional_scalars_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_clamp()

    if min_val is None and max_val is None:
        raise ValueError("clamp requires min or max")

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
    cdef int64_t alloc_size_fclamp = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fclamp, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fclamp, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr

    cdef uintptr_t min_scalar = 0
    cdef uintptr_t max_scalar = 0
    if min_val is not None:
        min_scalar = _create_scalar_fn(_scalar_bytes_fn(min_val, a_dtype), dtype_code)
    if max_val is not None:
        max_scalar = _create_scalar_fn(_scalar_bytes_fn(max_val, a_dtype), dtype_code)
    cdef uintptr_t stream_raw = int(stream.stream)
    try:
        ws_size, executor = _ffi_ref.clamp_optional_scalars_op(
            _clamp_getws_ptr, _clamp_exec_ptr,
            a.shape, a.stride,
            out_shape, out_stride,
            dtype_code, 2,
            a_ptr, o_ptr,
            min_scalar, max_scalar,
            stream_raw)

        if ws_size:
            workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            try:
                ret = _ffi_execute(
                    _clamp_exec_ptr, int(workspace_ptr), ws_size,
                    executor, stream_raw)
                if ret != 0:
                    raise RuntimeError(f"aclnnClamp execute failed: {ret}")
            finally:
                runtime.defer_raw_free(workspace_ptr)

        _defer_executor_fn(executor)
    finally:
        if min_scalar:
            _destroy_scalar_fn(int(min_scalar))
        if max_scalar:
            _destroy_scalar_fn(int(max_scalar))

    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)



def fast_clamp_min(a, min_val):
    return fast_clamp(a, min_val, None)



def fast_clamp_max(a, max_val):
    return fast_clamp(a, None, max_val)


def fast_clamp_inplace(a, min_val=None, max_val=None):
    """In-place clamp_(a, min_val, max_val) reusing _ffi.clamp_optional_scalars_op."""
    _ensure_npu_imports()
    _ensure_ffi_clamp()


    cdef int dev_idx = a.device.index or 0
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    a_stride = a.stride
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()

    cdef uintptr_t min_scalar = 0
    cdef uintptr_t max_scalar = 0
    if min_val is not None:
        min_scalar = _create_scalar_fn(_scalar_bytes_fn(min_val, a_dtype), dtype_code)
    if max_val is not None:
        max_scalar = _create_scalar_fn(_scalar_bytes_fn(max_val, a_dtype), dtype_code)
    cdef uintptr_t stream_raw = int(stream.stream)
    try:
        ws_size, executor = _ffi_ref.clamp_optional_scalars_op(
            _clamp_getws_ptr, _clamp_exec_ptr,
            a_shape, a_stride,
            a_shape, a_stride,
            dtype_code, 2,
            a_ptr, a_ptr,
            min_scalar, max_scalar,
            stream_raw)

        if ws_size:
            workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            try:
                ret = _ffi_execute(
                    _clamp_exec_ptr, int(workspace_ptr), ws_size,
                    executor, stream_raw)
                if ret != 0:
                    raise RuntimeError(f"aclnnClamp execute failed: {ret}")
            finally:
                runtime.defer_raw_free(workspace_ptr)

        _defer_executor_fn(executor)
    finally:
        if min_scalar:
            _destroy_scalar_fn(int(min_scalar))
        if max_scalar:
            _destroy_scalar_fn(int(max_scalar))

    return a


def fast_copy_inplace(a, src):
    """In-place copy_(a, src) using _ffi.inplace_copy_op."""
    _ensure_npu_imports()
    _ensure_ffi_inplace_copy()


    cdef int dev_idx = a.device.index or 0
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    src_shape = (<TensorImpl>src)._shape_tuple if isinstance(src, TensorImpl) else src.shape
    cdef int dtype_code_dst = _dtype_to_acl_code(a.dtype)
    cdef int dtype_code_src = _dtype_to_acl_code(src.dtype)
    cdef uintptr_t a_ptr, src_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    if isinstance(src, TensorImpl):
        src_ptr = <uintptr_t>(<TensorImpl>src)._storage._untyped._device_ptr
    else:
        src_ptr = <uintptr_t>src.storage().data_ptr()
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_ref.inplace_copy_op(
        _inplace_copy_getws_ptr, _inplace_copy_exec_ptr,
        a_shape, a.stride,
        src_shape, src.stride,
        dtype_code_dst, dtype_code_src, 2,
        a_ptr, src_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _inplace_copy_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnInplaceCopy execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def fast_zero_stride_contiguous(a):
    """Materialize an all-zero-stride NPU view as a dense contiguous tensor.

    This is the hot leaf-grad path for full-sum backward.  It bypasses the
    Python backend contiguous wrapper while preserving the same ACLNN
    InplaceCopy semantics and PTA executor-cache behavior.
    """
    _ensure_npu_imports()
    _ensure_ffi_inplace_copy()

    cdef TensorImpl t
    cdef int ndim
    cdef int i
    cdef int dev_idx
    cdef int isize
    cdef int dtype_code
    cdef int64_t n
    cdef int64_t storage_numel
    cdef int64_t alloc_size
    cdef int64_t[MAX_NDIM] shape_buf, out_stride_buf
    cdef object out_shape
    cdef object out_stride
    cdef object out_ptr
    cdef object stream_obj
    cdef object runtime = None
    cdef object workspace_ptr
    cdef object ret
    cdef object ws_size
    cdef object executor = 0
    cdef object pta_state
    cdef uintptr_t src_ptr
    cdef uintptr_t o_ptr
    cdef uintptr_t stream_raw
    cdef uint64_t ws_size_raw = 0
    cdef uintptr_t executor_raw = 0
    cdef bint pta_active = False
    cdef bint executor_owned = False
    cdef int ret_i

    if not isinstance(a, TensorImpl):
        return None
    t = <TensorImpl>a
    if t._device_type != 1:
        return None
    ndim = t._ndim
    if ndim == 0 or ndim > MAX_NDIM:
        return None
    for i in range(ndim):
        if t._c_stride[i] != 0:
            return None

    dev_idx = t._device_index
    stream_obj = _get_stream_obj_fast(dev_idx)
    stream_raw = _get_stream_raw_fast(dev_idx)
    out_shape = t._shape_tuple
    _fill_shape(out_shape, shape_buf, ndim)
    with nogil:
        c_contiguous_stride(shape_buf, ndim, out_stride_buf)
    out_stride = _to_tuple(out_stride_buf, ndim)

    n = t._c_numel
    storage_numel = n if n > 0 else 1
    isize = t._itemsize
    alloc_size = storage_numel * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc_large_cached(alloc_size, stream_obj)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size, stream=stream_obj)

    dtype_code = _tensor_dtype_to_acl_code(t)
    src_ptr = <uintptr_t>t._storage._untyped._device_ptr + <uintptr_t>(t._c_offset * t._itemsize)
    o_ptr = <uintptr_t>int(out_ptr)

    try:
        pta_state = _ffi_pta_begin_inplace_copy_cache_lookup(
            out_shape,
            out_stride,
            t._shape_tuple,
            t._stride_tuple,
            dtype_code,
            dtype_code,
            o_ptr,
            src_ptr,
            stream_raw,
        )
        if pta_state is not None:
            pta_active = bool(pta_state[0])
            ws_size_raw = <uint64_t>int(pta_state[1])
            executor_raw = <uintptr_t>int(pta_state[2])
            if executor_raw != 0:
                runtime = _run_executor(
                    <uintptr_t>int(_inplace_copy_exec_ptr), ws_size_raw, executor_raw,
                    stream_raw, dev_idx, runtime, "aclnnInplaceCopy")
                return _make_npu_tensor_fast_large(
                    out_ptr, storage_numel, t._dtype_obj, t._device_obj,
                    out_shape, out_stride, isize, dev_idx, t._dtype_code)

        ws_size, executor = _ffi_ref.inplace_copy_op(
            _inplace_copy_getws_ptr, _inplace_copy_exec_ptr,
            out_shape, out_stride,
            t._shape_tuple, t._stride_tuple,
            dtype_code, dtype_code, 2,
            o_ptr, src_ptr,
            stream_raw)
        executor_owned = True
        if ws_size:
            workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            try:
                ret = _ffi_execute(
                    _inplace_copy_exec_ptr, <uintptr_t>int(workspace_ptr), ws_size,
                    executor, stream_raw)
                if ret != 0:
                    raise RuntimeError(f"aclnnInplaceCopy execute failed: {ret}")
            finally:
                if runtime is None:
                    runtime = _get_runtime_fast(dev_idx)
                runtime.defer_raw_free(workspace_ptr)
        return _make_npu_tensor_fast_large(
            out_ptr, storage_numel, t._dtype_obj, t._device_obj,
            out_shape, out_stride, isize, dev_idx, t._dtype_code)
    finally:
        if pta_active:
            _ffi_pta_end_cache_lookup()
        if executor_owned:
            _defer_executor_handle(<uintptr_t>int(executor))



def fast_fill_inplace(a, value):
    """In-place fill_(a, value) by broadcasting a scalar tensor via InplaceCopy."""
    src = _npu_scalar_like(float(value), a)
    return fast_copy_inplace(a, src)



def fast_hardtanh(a, min_val, max_val):
    """Optimized out-of-place hardtanh(a) that calls _ffi.tensor_two_scalars_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_hardtanh()

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
                ret = _ffi_execute(
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
            ret = _ffi_execute(
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
            ret = _ffi_execute(
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
            ret = _ffi_execute(
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
            ret = _ffi_execute(
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
                ret = _ffi_execute(
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
            ret = _ffi_execute(
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
            ret = _ffi_execute(
                _log_softmax_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnLogSoftmax execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


cpdef object fast_gelu(object a):
    """Optimized out-of-place gelu(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_gelu()

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
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr + <uintptr_t>((<TensorImpl>a)._c_offset * (<TensorImpl>a)._itemsize)
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
                _gelu_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnGelu execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_handle(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


cpdef object fast_gelu_exact(TensorImpl a):
    """Gelu for already-validated exact base NPU tensors.

    Mirrors fast_silu_exact: large-pool cached allocation, raw-handle executor
    defer, and the torch_npu-aligned PTA executor cache.  aclnnGelu has no
    strided-view limitation observed for the MLP shapes, but to stay safe we
    materialize a contiguous copy when the cached stride is non-canonical.
    """
    _ensure_npu_imports()
    _ensure_ffi_gelu()

    cdef int dev_idx = a._device_index
    cdef object a_dev = a._device_obj
    cdef object a_dtype = a._dtype_obj
    cdef object runtime = None
    cdef object stream_obj = _get_stream_obj_fast(dev_idx)
    cdef object in_shape = a._shape_tuple
    cdef object in_stride = a._stride_tuple
    cdef object out_shape = in_shape
    cdef int ndim = len(out_shape)
    cdef int64_t[MAX_NDIM] shape_buf, out_stride_buf
    cdef object out_stride
    cdef bint input_is_contiguous
    cdef bint strict_contiguous
    cdef int64_t n
    cdef object src_for_gelu = a
    cdef int isize
    cdef int64_t alloc_size_fgelu
    cdef object out_ptr
    cdef int dtype_code
    cdef uintptr_t a_ptr, o_ptr
    cdef uintptr_t stream_raw
    cdef bint pta_active = False
    cdef uint64_t ws_size_raw = 0
    cdef uintptr_t executor_raw = 0
    cdef int pta_lookup
    cdef int ret_i
    cdef object ws_size
    cdef object executor = 0
    cdef object workspace_ptr
    cdef object ret

    if ndim > MAX_NDIM:
        raise ValueError(f"ndim exceeds MAX_NDIM ({MAX_NDIM})")
    strict_contiguous = _tensor_has_strict_contiguous_stride(a)
    if strict_contiguous:
        out_stride = in_stride
        input_is_contiguous = True
    else:
        _fill_shape(out_shape, shape_buf, ndim)
        with nogil:
            c_contiguous_stride(shape_buf, ndim, out_stride_buf)
        out_stride = _to_tuple(out_stride_buf, ndim)
        input_is_contiguous = in_stride == out_stride
    n = a._c_numel

    if not input_is_contiguous:
        src_for_gelu = _npu_contiguous_copy(a)
        if isinstance(src_for_gelu, TensorImpl):
            in_shape = (<TensorImpl>src_for_gelu)._shape_tuple
            in_stride = (<TensorImpl>src_for_gelu)._stride_tuple
        else:
            in_shape = src_for_gelu.shape
            in_stride = src_for_gelu.stride

    isize = a._itemsize
    alloc_size_fgelu = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc_large_cached(alloc_size_fgelu, stream_obj)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fgelu, stream=stream_obj)

    dtype_code = _tensor_dtype_to_acl_code(a)
    if isinstance(src_for_gelu, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>src_for_gelu)._storage._untyped._device_ptr + <uintptr_t>((<TensorImpl>src_for_gelu)._c_offset * (<TensorImpl>src_for_gelu)._itemsize)
    else:
        a_ptr = <uintptr_t>src_for_gelu.storage().data_ptr()
    o_ptr = out_ptr
    stream_raw = _get_stream_raw_fast(dev_idx)

    if _use_gelu_pta_cache and _pta_unary_begin_fn is not None:
        pta_lookup = _ffi_pta_begin_unary_cache_lookup_raw(
            b"aclnnGelu",
            in_shape, in_stride,
            out_shape, out_stride,
            dtype_code, dtype_code,
            a_ptr, o_ptr,
            stream_raw,
            &pta_active,
            &ws_size_raw,
            &executor_raw)
        if pta_lookup and executor_raw != 0:
            try:
                runtime = _run_executor(_gelu_exec_ptr, ws_size_raw, executor_raw,
                                        stream_raw, dev_idx, runtime, "aclnnGelu")
                return _make_npu_tensor_fast_large(out_ptr, n, a_dtype, a_dev, out_shape, out_stride, isize, dev_idx, a._dtype_code)
            finally:
                if pta_active:
                    _ffi_pta_end_cache_lookup()
                    pta_active = False

    try:
        ws_size, executor = _ffi_unary_op(
            _gelu_getws_ptr, _gelu_exec_ptr,
            in_shape, in_stride,
            out_shape, out_stride,
            dtype_code, dtype_code, 2,
            a_ptr, o_ptr,
            stream_raw)
        if ws_size:
            workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            try:
                ret = _ffi_execute(
                    _gelu_exec_ptr, int(workspace_ptr), ws_size,
                    executor, stream_raw)
                if ret != 0:
                    raise RuntimeError(f"aclnnGelu execute failed: {ret}")
            finally:
                if runtime is None:
                    runtime = _get_runtime_fast(dev_idx)
                runtime.defer_raw_free(workspace_ptr)
    finally:
        if executor:
            _defer_executor_handle(executor)
        if pta_active:
            _ffi_pta_end_cache_lookup()

    return _make_npu_tensor_fast_large(out_ptr, n, a_dtype, a_dev, out_shape, out_stride, isize, dev_idx, a._dtype_code)


cpdef object fast_silu(object a):
    """Optimized out-of-place silu(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_silu()

    cdef int dev_idx = (<TensorImpl>a)._device_index if isinstance(a, TensorImpl) else (a.device.index or 0)
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    in_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    in_stride = (<TensorImpl>a)._stride_tuple if isinstance(a, TensorImpl) else a.stride
    out_shape = in_shape
    cdef int ndim = len(out_shape)
    if ndim > MAX_NDIM:
        raise ValueError(f"ndim exceeds MAX_NDIM ({MAX_NDIM})")
    cdef int64_t[MAX_NDIM] shape_buf, out_stride_buf
    _fill_shape(out_shape, shape_buf, ndim)
    with nogil:
        c_contiguous_stride(shape_buf, ndim, out_stride_buf)
    out_stride = _to_tuple(out_stride_buf, ndim)
    cdef bint input_is_contiguous = in_stride == out_stride
    cdef int64_t n
    if isinstance(a, TensorImpl):
        n = (<TensorImpl>a)._c_numel
    else:
        n = 1
        for dim in out_shape:
            n *= dim

    cdef object src_for_silu = a
    if not input_is_contiguous:
        # TODO: re-enable native strided-view silu when CANN fixes aclnnSilu 561103.
        src_for_silu = _npu_contiguous_copy(a)
        if isinstance(src_for_silu, TensorImpl):
            in_shape = (<TensorImpl>src_for_silu)._shape_tuple
            in_stride = (<TensorImpl>src_for_silu)._stride_tuple
        else:
            in_shape = src_for_silu.shape
            in_stride = src_for_silu.stride

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_fsilu = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fsilu, stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fsilu, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(src_for_silu, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>src_for_silu)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>src_for_silu.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)
    cdef bint pta_active = False

    # Try PTA executor cache (torch_npu-aligned hit_cache_v2 path)
    if _use_silu_pta_cache and _pta_unary_begin_fn is not None:
        state = _ffi_pta_begin_unary_cache_lookup(
            b"aclnnSilu",
            in_shape, in_stride,
            out_shape, out_stride,
            dtype_code, dtype_code,
            a_ptr, o_ptr,
            stream_raw)
        if state is not None:
            pta_active = bool(state[0])
            ws_size = state[1]
            executor = state[2]
            if executor:
                try:
                    # PTA hit_cache_v2 rebinds current addresses with
                    # AddTensorAddrToCachedList during lookup, so cached Silu
                    # executors are valid for freshly allocated outputs.
                    if ws_size:
                        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
                        if ret != 0:
                            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
                        try:
                            ret = _ffi_execute(
                                _silu_exec_ptr, int(workspace_ptr), ws_size,
                                executor, stream_raw)
                            if ret != 0:
                                raise RuntimeError(f"aclnnSilu execute failed: {ret}")
                        finally:
                            runtime.defer_raw_free(workspace_ptr)
                    else:
                        ret = _ffi_execute(_silu_exec_ptr, 0, 0, executor, stream_raw)
                        if ret != 0:
                            raise RuntimeError(f"aclnnSilu execute failed: {ret}")
                    return _make_npu_tensor_fast(out_ptr, n, a_dtype, a_dev, out_shape, out_stride, isize)
                finally:
                    if pta_active:
                        _ffi_pta_end_cache_lookup()
                        pta_active = False
            # else: PTA miss — fall through with pta_active still set so the
            # GetWorkspaceSize path below runs inside the open PTA context and
            # the miss-path finally closes it.

    try:
        ws_size, executor = _ffi_unary_op(
            _silu_getws_ptr, _silu_exec_ptr,
            in_shape, in_stride,
            out_shape, out_stride,
            dtype_code, dtype_code, 2,
            a_ptr, o_ptr,
            stream_raw)

        if ws_size:
            workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            try:
                ret = _ffi_execute(
                    _silu_exec_ptr, int(workspace_ptr), ws_size,
                    executor, stream_raw)
                if ret != 0:
                    raise RuntimeError(f"aclnnSilu execute failed: {ret}")
            finally:
                runtime.defer_raw_free(workspace_ptr)

        _defer_executor_handle(executor)
    finally:
        if pta_active:
            _ffi_pta_end_cache_lookup()

    return _make_npu_tensor_fast(out_ptr, n, a_dtype, a_dev, out_shape, out_stride, isize)


cpdef object fast_silu_exact(TensorImpl a):
    """Silu for already-validated exact base NPU tensors."""
    _ensure_npu_imports()
    _ensure_ffi_silu()

    cdef int dev_idx = a._device_index
    cdef object a_dev = a._device_obj
    cdef object a_dtype = a._dtype_obj
    cdef object runtime = None
    cdef object stream_obj = _get_stream_obj_fast(dev_idx)
    cdef object in_shape = a._shape_tuple
    cdef object in_stride = a._stride_tuple
    cdef object out_shape = in_shape
    cdef int ndim = len(out_shape)
    cdef int64_t[MAX_NDIM] shape_buf, out_stride_buf
    cdef object out_stride
    cdef bint input_is_contiguous
    cdef bint strict_contiguous
    cdef int64_t n
    cdef object src_for_silu = a
    cdef int isize
    cdef int64_t alloc_size_fsilu
    cdef object out_ptr
    cdef int dtype_code
    cdef uintptr_t a_ptr, o_ptr
    cdef uintptr_t stream_raw
    cdef bint pta_active = False
    cdef uint64_t ws_size_raw = 0
    cdef uintptr_t executor_raw = 0
    cdef int pta_lookup
    cdef int ret_i
    cdef object ws_size
    cdef object executor
    cdef object workspace_ptr
    cdef object ret

    if ndim > MAX_NDIM:
        raise ValueError(f"ndim exceeds MAX_NDIM ({MAX_NDIM})")
    strict_contiguous = _tensor_has_strict_contiguous_stride(a)
    if strict_contiguous:
        out_stride = in_stride
        input_is_contiguous = True
    else:
        _fill_shape(out_shape, shape_buf, ndim)
        with nogil:
            c_contiguous_stride(shape_buf, ndim, out_stride_buf)
        out_stride = _to_tuple(out_stride_buf, ndim)
        input_is_contiguous = in_stride == out_stride
    n = a._c_numel

    if not input_is_contiguous:
        # TODO: re-enable native strided-view silu when CANN fixes aclnnSilu 561103.
        src_for_silu = _npu_contiguous_copy(a)
        if isinstance(src_for_silu, TensorImpl):
            in_shape = (<TensorImpl>src_for_silu)._shape_tuple
            in_stride = (<TensorImpl>src_for_silu)._stride_tuple
        else:
            in_shape = src_for_silu.shape
            in_stride = src_for_silu.stride

    isize = a._itemsize
    alloc_size_fsilu = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc_large_cached(alloc_size_fsilu, stream_obj)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_fsilu, stream=stream_obj)

    dtype_code = _tensor_dtype_to_acl_code(a)
    if isinstance(src_for_silu, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>src_for_silu)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>src_for_silu.storage().data_ptr()
    o_ptr = out_ptr
    stream_raw = _get_stream_raw_fast(dev_idx)

    if _use_silu_pta_cache and _pta_unary_begin_fn is not None:
        pta_lookup = _ffi_pta_begin_unary_cache_lookup_raw(
            b"aclnnSilu",
            in_shape, in_stride,
            out_shape, out_stride,
            dtype_code, dtype_code,
            a_ptr, o_ptr,
            stream_raw,
            &pta_active,
            &ws_size_raw,
            &executor_raw)
        if pta_lookup and executor_raw != 0:
            try:
                if ws_size_raw:
                    workspace_ptr, ret = _acl_rt_malloc_fn(ws_size_raw, 0)
                    if ret != 0:
                        raise RuntimeError(f"acl.rt.malloc failed: {ret}")
                    try:
                        ret_i = _ffi_execute(
                            _silu_exec_ptr, <uintptr_t>int(workspace_ptr), ws_size_raw,
                            executor_raw, stream_raw)
                        if ret_i != 0:
                            raise RuntimeError(f"aclnnSilu execute failed: {ret_i}")
                    finally:
                        if runtime is None:
                            runtime = _get_runtime_fast(dev_idx)
                        runtime.defer_raw_free(workspace_ptr)
                else:
                    ret_i = _ffi_execute(_silu_exec_ptr, 0, 0, executor_raw, stream_raw)
                    if ret_i != 0:
                        raise RuntimeError(f"aclnnSilu execute failed: {ret_i}")
                return _make_npu_tensor_fast_large(out_ptr, n, a_dtype, a_dev, out_shape, out_stride, isize, dev_idx, a._dtype_code)
            finally:
                if pta_active:
                    _ffi_pta_end_cache_lookup()
                    pta_active = False

    try:
        ws_size, executor = _ffi_unary_op(
            _silu_getws_ptr, _silu_exec_ptr,
            in_shape, in_stride,
            out_shape, out_stride,
            dtype_code, dtype_code, 2,
            a_ptr, o_ptr,
            stream_raw)
        if ws_size:
            workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            try:
                ret = _ffi_execute(
                    _silu_exec_ptr, int(workspace_ptr), ws_size,
                    executor, stream_raw)
                if ret != 0:
                    raise RuntimeError(f"aclnnSilu execute failed: {ret}")
            finally:
                if runtime is None:
                    runtime = _get_runtime_fast(dev_idx)
                runtime.defer_raw_free(workspace_ptr)
        _defer_executor_handle(executor)
    finally:
        if pta_active:
            _ffi_pta_end_cache_lookup()

    return _make_npu_tensor_fast_large(out_ptr, n, a_dtype, a_dev, out_shape, out_stride, isize, dev_idx, a._dtype_code)


cpdef object fast_cast(object a, object dst_dtype):
    """Fast contiguous NPU dtype cast via cached Cython ACLNN Cast FFI."""
    _ensure_npu_imports()
    _ensure_ffi_cast()

    if not isinstance(a, TensorImpl):
        return None
    cdef TensorImpl src = <TensorImpl>a
    if src._device_type != 1:
        return None
    if src._dtype_obj == dst_dtype:
        return a
    if not _tensor_has_strict_contiguous_stride(src):
        return None

    cdef int dev_idx = src._device_index
    cdef object a_dev = src._device_obj
    cdef object in_shape = src._shape_tuple
    cdef object in_stride = src._stride_tuple
    cdef object out_shape = in_shape
    cdef object out_stride = in_stride
    cdef int64_t n = src._c_numel
    if n == 0:
        return None

    cdef int dst_itemsize = c_dtype_itemsize(dst_dtype)
    cdef int dst_tensor_code = _dtype_to_tensor_code(dst_dtype)
    cdef int src_acl_code = _tensor_dtype_to_acl_code(src)
    cdef int dst_acl_code = _dtype_to_acl_code(dst_dtype)
    cdef int64_t alloc_size = n * dst_itemsize
    cdef object stream_obj = _get_stream_obj_fast(dev_idx)
    cdef object out_ptr
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc_large_cached(alloc_size, stream_obj)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size, stream=stream_obj)

    cdef uintptr_t src_ptr = _npu_data_ptr_fast(a, src._itemsize)
    cdef uintptr_t dst_ptr = <uintptr_t>int(out_ptr)
    cdef uintptr_t stream_raw = _get_stream_raw_fast(dev_idx)
    cdef bint pta_active = False
    cdef uint64_t ws_size_raw = 0
    cdef uintptr_t executor_raw = 0
    cdef int pta_lookup
    cdef object runtime = None
    cdef object ws_size
    cdef object executor = 0

    if _use_cast_pta_cache and _pta_unary_begin_fn is not None:
        pta_lookup = _ffi_pta_begin_unary_cache_lookup_raw(
            b"aclnnCast",
            in_shape, in_stride,
            out_shape, out_stride,
            src_acl_code, dst_acl_code,
            src_ptr, dst_ptr,
            stream_raw,
            &pta_active,
            &ws_size_raw,
            &executor_raw)
        if pta_lookup and executor_raw != 0:
            try:
                runtime = _run_executor(_cast_exec_ptr, ws_size_raw, executor_raw,
                                        stream_raw, dev_idx, runtime, "aclnnCast")
                return _make_npu_tensor_fast_large(
                    dst_ptr, n, dst_dtype, a_dev, out_shape, out_stride,
                    dst_itemsize, dev_idx, dst_tensor_code)
            finally:
                if pta_active:
                    _ffi_pta_end_cache_lookup()
                    pta_active = False

    try:
        ws_size, executor = _ffi_cast_op(
            _cast_getws_ptr, _cast_exec_ptr,
            in_shape, in_stride,
            out_shape, out_stride,
            src_acl_code, dst_acl_code, 2,
            src_ptr, dst_ptr,
            stream_raw)
        if ws_size:
            runtime = _run_executor(_cast_exec_ptr, <uint64_t>ws_size, <uintptr_t>executor,
                                    stream_raw, dev_idx, runtime, "aclnnCast")
        if executor:
            _defer_executor_handle(<uintptr_t>executor)
    finally:
        if pta_active:
            _ffi_pta_end_cache_lookup()

    return _make_npu_tensor_fast_large(
        dst_ptr, n, dst_dtype, a_dev, out_shape, out_stride,
        dst_itemsize, dev_idx, dst_tensor_code)


cpdef object fast_gelu_backward(object grad, object saved_input):
    """Optimized GELU backward via aclnnGeluBackward with cached Cython FFI."""
    _ensure_npu_imports()
    _ensure_ffi_gelu_backward()

    if not (isinstance(grad, TensorImpl) and isinstance(saved_input, TensorImpl)):
        raise ValueError("fast_gelu_backward expects base TensorImpl operands")
    cdef TensorImpl grad_t = <TensorImpl>grad
    cdef TensorImpl input_t = <TensorImpl>saved_input
    if grad_t._device_type != 1 or input_t._device_type != 1:
        raise ValueError("NPU gelu_backward expects NPU tensors")
    if grad_t._device_index != input_t._device_index:
        raise ValueError("NPU gelu_backward requires tensors on the same device")
    if grad_t._dtype_code != input_t._dtype_code:
        raise ValueError("NPU gelu_backward requires matching dtypes")

    cdef int dev_idx = grad_t._device_index
    cdef object a_dev = grad_t._device_obj
    cdef object a_dtype = grad_t._dtype_obj
    cdef object runtime = None
    cdef object stream_obj = _get_stream_obj_fast(dev_idx)
    cdef object py_grad_shape = grad_t._shape_tuple
    cdef object py_self_shape = input_t._shape_tuple
    cdef object grad_stride = grad_t._stride_tuple
    cdef object input_stride = input_t._stride_tuple
    if py_grad_shape != py_self_shape:
        raise ValueError("NPU gelu_backward requires matching grad and input shapes")
    cdef int ndim = grad_t._ndim
    if ndim > MAX_NDIM:
        raise ValueError(f"ndim exceeds MAX_NDIM ({MAX_NDIM})")

    cdef int64_t[MAX_NDIM] shape_buf, out_stride_buf
    _fill_shape(py_grad_shape, shape_buf, ndim)
    cdef int64_t n
    with nogil:
        c_contiguous_stride(shape_buf, ndim, out_stride_buf)
        n = c_numel(shape_buf, ndim)
    cdef object out_shape = py_grad_shape
    cdef object out_stride = _to_tuple(out_stride_buf, ndim)

    cdef int isize = grad_t._itemsize
    cdef int64_t alloc_size = n * isize
    cdef object out_ptr
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc_large_cached(alloc_size, stream_obj)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size, stream=stream_obj)

    cdef int dtype_code = _tensor_dtype_to_acl_code(grad_t)
    cdef uintptr_t grad_ptr = <uintptr_t>grad_t._storage._untyped._device_ptr + <uintptr_t>(grad_t._c_offset * grad_t._itemsize)
    cdef uintptr_t self_ptr = <uintptr_t>input_t._storage._untyped._device_ptr + <uintptr_t>(input_t._c_offset * input_t._itemsize)
    cdef uintptr_t o_ptr = <uintptr_t>int(out_ptr)
    cdef uintptr_t stream_raw = _get_stream_raw_fast(dev_idx)
    cdef bint pta_active = False
    cdef uint64_t ws_size_raw = 0
    cdef uintptr_t executor_raw = 0
    cdef int pta_lookup
    cdef int ret_i
    cdef object ws_size
    cdef object executor = 0
    cdef object workspace_ptr
    cdef object ret

    if _use_gelu_backward_pta_cache and _pta_binary_begin_fn is not None:
        pta_lookup = _ffi_pta_begin_binary_cache_lookup_raw(
            b"aclnnGeluBackward",
            py_grad_shape, grad_stride,
            py_self_shape, input_stride,
            out_shape, out_stride,
            dtype_code,
            grad_ptr, self_ptr, o_ptr,
            stream_raw,
            &pta_active,
            &ws_size_raw,
            &executor_raw)
        if pta_lookup and executor_raw != 0:
            try:
                runtime = _run_executor(_gelu_backward_exec_ptr, ws_size_raw, executor_raw,
                                        stream_raw, dev_idx, runtime, "aclnnGeluBackward")
                return _make_npu_tensor_fast_large(out_ptr, n, a_dtype, a_dev, out_shape, out_stride, isize, dev_idx, grad_t._dtype_code)
            finally:
                if pta_active:
                    _ffi_pta_end_cache_lookup()
                    pta_active = False

    try:
        ws_size, executor = _ffi_binary_op_no_alpha(
            _gelu_backward_getws_ptr, _gelu_backward_exec_ptr,
            py_grad_shape, grad_stride,
            py_self_shape, input_stride,
            out_shape, out_stride,
            dtype_code, 2,
            grad_ptr, self_ptr, o_ptr,
            stream_raw)

        if ws_size:
            workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            try:
                ret = _ffi_execute(
                    _gelu_backward_exec_ptr, int(workspace_ptr), ws_size,
                    executor, stream_raw)
                if ret != 0:
                    raise RuntimeError(f"aclnnGeluBackward execute failed: {ret}")
            finally:
                if runtime is None:
                    runtime = _get_runtime_fast(dev_idx)
                runtime.defer_raw_free(workspace_ptr)

        _defer_executor_handle(executor)
    finally:
        if pta_active:
            _ffi_pta_end_cache_lookup()

    return _make_npu_tensor_fast_large(out_ptr, n, a_dtype, a_dev, out_shape, out_stride, isize, dev_idx, grad_t._dtype_code)


cpdef object fast_silu_backward(object grad, object saved_input):
    """Optimized silu backward via aclnnSiluBackward with cached Cython FFI."""
    _ensure_npu_imports()
    _ensure_ffi_silu_backward()

    cdef int dev_idx
    _validate_npu_binary(grad, saved_input, "silu_backward", &dev_idx)
    a_dev = _device_obj_fast(grad)
    a_dtype = grad.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    py_grad_shape = (<TensorImpl>grad)._shape_tuple if isinstance(grad, TensorImpl) else grad.shape
    py_self_shape = (<TensorImpl>saved_input)._shape_tuple if isinstance(saved_input, TensorImpl) else saved_input.shape
    if py_grad_shape != py_self_shape:
        raise ValueError("NPU silu_backward requires matching grad and input shapes")
    cdef int ndim = len(py_grad_shape)
    if ndim > MAX_NDIM:
        raise ValueError(f"ndim exceeds MAX_NDIM ({MAX_NDIM})")

    cdef int64_t[MAX_NDIM] shape_buf, out_stride_buf
    _fill_shape(py_grad_shape, shape_buf, ndim)
    cdef int64_t n
    with nogil:
        c_contiguous_stride(shape_buf, ndim, out_stride_buf)
        n = c_numel(shape_buf, ndim)
    out_shape = py_grad_shape
    out_stride = _to_tuple(out_stride_buf, ndim)

    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t grad_ptr, self_ptr, o_ptr
    if isinstance(grad, TensorImpl):
        grad_ptr = <uintptr_t>(<TensorImpl>grad)._storage._untyped._device_ptr
    else:
        grad_ptr = <uintptr_t>grad.storage().data_ptr()
    if isinstance(saved_input, TensorImpl):
        self_ptr = <uintptr_t>(<TensorImpl>saved_input)._storage._untyped._device_ptr
    else:
        self_ptr = <uintptr_t>saved_input.storage().data_ptr()
    o_ptr = out_ptr

    cdef uintptr_t stream_raw = int(stream.stream)
    cdef bint pta_active = False
    cdef bint pta_cache_miss = False
    cdef object pta_key = None
    cdef object pointer_key = None

    if _use_silu_backward_pta_cache and _pta_binary_begin_fn is not None:
        state = _ffi_pta_begin_binary_cache_lookup(
            b"aclnnSiluBackward",
            py_grad_shape, grad.stride,
            py_self_shape, saved_input.stride,
            out_shape, out_stride,
            dtype_code,
            grad_ptr, self_ptr, o_ptr,
            stream_raw)
        if state is not None:
            pta_active = bool(state[0])
            ws_size = state[1]
            executor = state[2]
            pta_key = (py_grad_shape, grad.stride, py_self_shape, saved_input.stride,
                       out_shape, out_stride, dtype_code)
            pointer_key = (grad_ptr, self_ptr, o_ptr)
            if executor:
                if _silu_backward_pta_pointer_keys.get(pta_key) == pointer_key:
                    try:
                        if ws_size:
                            workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
                            if ret != 0:
                                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
                            try:
                                ret = _ffi_execute(
                                    _silu_backward_exec_ptr, int(workspace_ptr), ws_size,
                                    executor, stream_raw)
                                if ret != 0:
                                    raise RuntimeError(f"aclnnSiluBackward execute failed: {ret}")
                            finally:
                                runtime.defer_raw_free(workspace_ptr)
                        else:
                            ret = _ffi_execute(_silu_backward_exec_ptr, 0, 0, executor, stream_raw)
                            if ret != 0:
                                raise RuntimeError(f"aclnnSiluBackward execute failed: {ret}")
                        return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)
                    finally:
                        if pta_active:
                            _ffi_pta_end_cache_lookup()
                            pta_active = False
                if pta_active:
                    _ffi_pta_end_cache_lookup()
                    pta_active = False
            else:
                pta_cache_miss = pta_active

    try:
        ws_size, executor = _ffi_binary_op_no_alpha(
            _silu_backward_getws_ptr, _silu_backward_exec_ptr,
            py_grad_shape, grad.stride,
            py_self_shape, saved_input.stride,
            out_shape, out_stride,
            dtype_code, 2,
            grad_ptr, self_ptr, o_ptr,
            stream_raw)

        if ws_size:
            workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            try:
                ret = _ffi_execute(
                    _silu_backward_exec_ptr, int(workspace_ptr), ws_size,
                    executor, stream_raw)
                if ret != 0:
                    raise RuntimeError(f"aclnnSiluBackward execute failed: {ret}")
            finally:
                runtime.defer_raw_free(workspace_ptr)

        if pta_cache_miss and pta_key is not None:
            _silu_backward_pta_pointer_keys[pta_key] = pointer_key

        _defer_executor_handle(executor)
    finally:
        if pta_active:
            _ffi_pta_end_cache_lookup()

    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)

def fast_mish(a):
    """Optimized out-of-place mish(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_mish()

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

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
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

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
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

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
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

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
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

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
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

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
                _floor_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnFloor execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_floor_inplace(a):
    """In-place floor_(a) using aclnnFloor with output aliased to input."""
    _ensure_npu_imports()
    _ensure_ffi_floor()

    cdef int dev_idx = a.device.index or 0
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    a_stride = a.stride
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_unary_op(
        _floor_getws_ptr, _floor_exec_ptr,
        a_shape, a_stride,
        a_shape, a_stride,
        dtype_code, dtype_code, 2,
        a_ptr, a_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _floor_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnFloor execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def fast_ceil(a):
    """Optimized out-of-place ceil(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_ceil()

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

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
                _ceil_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnCeil execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_ceil_inplace(a):
    """In-place ceil_(a) using aclnnCeil with output aliased to input."""
    _ensure_npu_imports()
    _ensure_ffi_ceil()

    cdef int dev_idx = a.device.index or 0
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    a_stride = a.stride
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_unary_op(
        _ceil_getws_ptr, _ceil_exec_ptr,
        a_shape, a_stride,
        a_shape, a_stride,
        dtype_code, dtype_code, 2,
        a_ptr, a_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _ceil_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnCeil execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def fast_round(a):
    """Optimized out-of-place round(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_round()

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

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
                _round_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnRound execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)



def fast_round_inplace(a):
    """In-place round_(a) using aclnnRound with output aliased to input."""
    _ensure_npu_imports()
    _ensure_ffi_round()

    cdef int dev_idx = a.device.index or 0
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    a_stride = a.stride
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_unary_op(
        _round_getws_ptr, _round_exec_ptr,
        a_shape, a_stride,
        a_shape, a_stride,
        dtype_code, dtype_code, 2,
        a_ptr, a_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _round_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnRound execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def fast_trunc(a):
    """Optimized out-of-place trunc(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_trunc()

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

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
                _trunc_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnTrunc execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_trunc_inplace(a):
    """In-place trunc_(a) using aclnnTrunc with output aliased to input."""
    _ensure_npu_imports()
    _ensure_ffi_trunc()

    cdef int dev_idx = a.device.index or 0
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    a_stride = a.stride
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_unary_op(
        _trunc_getws_ptr, _trunc_exec_ptr,
        a_shape, a_stride,
        a_shape, a_stride,
        dtype_code, dtype_code, 2,
        a_ptr, a_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _trunc_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnTrunc execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def fast_log2(a):
    """Optimized out-of-place log2(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_log2()

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

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
                _log2_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnLog2 execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_log2_inplace(a):
    """In-place log2_(a) using aclnnLog2 with output aliased to input."""
    _ensure_npu_imports()
    _ensure_ffi_log2()

    cdef int dev_idx = a.device.index or 0
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    a_stride = a.stride
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_unary_op(
        _log2_getws_ptr, _log2_exec_ptr,
        a_shape, a_stride,
        a_shape, a_stride,
        dtype_code, dtype_code, 2,
        a_ptr, a_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _log2_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnLog2 execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def fast_log10(a):
    """Optimized out-of-place log10(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_log10()

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

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
                _log10_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnLog10 execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)


def fast_log10_inplace(a):
    """In-place log10_(a) using aclnnLog10 with output aliased to input."""
    _ensure_npu_imports()
    _ensure_ffi_log10()

    cdef int dev_idx = a.device.index or 0
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    a_stride = a.stride
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_unary_op(
        _log10_getws_ptr, _log10_exec_ptr,
        a_shape, a_stride,
        a_shape, a_stride,
        dtype_code, dtype_code, 2,
        a_ptr, a_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _log10_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnLog10 execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def fast_expm1_inplace(a):
    """In-place expm1_(a) using aclnnExpm1 with output aliased to input."""
    _ensure_npu_imports()
    _ensure_ffi_expm1()

    cdef int dev_idx = a.device.index or 0
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    a_stride = a.stride
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_unary_op(
        _expm1_getws_ptr, _expm1_exec_ptr,
        a_shape, a_stride,
        a_shape, a_stride,
        dtype_code, dtype_code, 2,
        a_ptr, a_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _expm1_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnExpm1 execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def fast_log1p_inplace(a):
    """In-place log1p_(a) using aclnnLog1p with output aliased to input."""
    _ensure_npu_imports()
    _ensure_ffi_log1p()

    cdef int dev_idx = a.device.index or 0
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    a_stride = a.stride
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_unary_op(
        _log1p_getws_ptr, _log1p_exec_ptr,
        a_shape, a_stride,
        a_shape, a_stride,
        dtype_code, dtype_code, 2,
        a_ptr, a_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _log1p_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnLog1p execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def fast_exp2_inplace(a):
    """In-place exp2_(a) using aclnnExp2 with output aliased to input."""
    _ensure_npu_imports()
    _ensure_ffi_exp2()

    cdef int dev_idx = a.device.index or 0
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    a_stride = a.stride
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_unary_op(
        _exp2_getws_ptr, _exp2_exec_ptr,
        a_shape, a_stride,
        a_shape, a_stride,
        dtype_code, dtype_code, 2,
        a_ptr, a_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _exp2_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnExp2 execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def fast_erf_inplace(a):
    """In-place erf_(a) using aclnnErf with output aliased to input."""
    _ensure_npu_imports()
    _ensure_ffi_erf()

    cdef int dev_idx = a.device.index or 0
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    a_stride = a.stride
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_unary_op(
        _erf_getws_ptr, _erf_exec_ptr,
        a_shape, a_stride,
        a_shape, a_stride,
        dtype_code, dtype_code, 2,
        a_ptr, a_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _erf_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnErf execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def fast_erfc_inplace(a):
    """In-place erfc_(a) using aclnnErfc with output aliased to input."""
    _ensure_npu_imports()
    _ensure_ffi_erfc()

    cdef int dev_idx = a.device.index or 0
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    a_stride = a.stride
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_unary_op(
        _erfc_getws_ptr, _erfc_exec_ptr,
        a_shape, a_stride,
        a_shape, a_stride,
        dtype_code, dtype_code, 2,
        a_ptr, a_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _erfc_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnErfc execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def fast_asin_inplace(a):
    """In-place asin_(a) using aclnnAsin with output aliased to input."""
    _ensure_npu_imports()
    _ensure_ffi_asin()

    cdef int dev_idx = a.device.index or 0
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    a_stride = a.stride
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_unary_op(
        _asin_getws_ptr, _asin_exec_ptr,
        a_shape, a_stride,
        a_shape, a_stride,
        dtype_code, dtype_code, 2,
        a_ptr, a_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _asin_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnAsin execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def fast_acos_inplace(a):
    """In-place acos_(a) using aclnnAcos with output aliased to input."""
    _ensure_npu_imports()
    _ensure_ffi_acos()

    cdef int dev_idx = a.device.index or 0
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    a_stride = a.stride
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_unary_op(
        _acos_getws_ptr, _acos_exec_ptr,
        a_shape, a_stride,
        a_shape, a_stride,
        dtype_code, dtype_code, 2,
        a_ptr, a_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _acos_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnAcos execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def fast_atan_inplace(a):
    """In-place atan_(a) using aclnnAtan with output aliased to input."""
    _ensure_npu_imports()
    _ensure_ffi_atan()

    cdef int dev_idx = a.device.index or 0
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    a_stride = a.stride
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_unary_op(
        _atan_getws_ptr, _atan_exec_ptr,
        a_shape, a_stride,
        a_shape, a_stride,
        dtype_code, dtype_code, 2,
        a_ptr, a_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _atan_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnAtan execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def fast_sinh_inplace(a):
    """In-place sinh_(a) using aclnnSinh with output aliased to input."""
    _ensure_npu_imports()
    _ensure_ffi_sinh()

    cdef int dev_idx = a.device.index or 0
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    a_stride = a.stride
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_unary_op(
        _sinh_getws_ptr, _sinh_exec_ptr,
        a_shape, a_stride,
        a_shape, a_stride,
        dtype_code, dtype_code, 2,
        a_ptr, a_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _sinh_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnSinh execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def fast_cosh_inplace(a):
    """In-place cosh_(a) using aclnnCosh with output aliased to input."""
    _ensure_npu_imports()
    _ensure_ffi_cosh()

    cdef int dev_idx = a.device.index or 0
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    a_stride = a.stride
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_unary_op(
        _cosh_getws_ptr, _cosh_exec_ptr,
        a_shape, a_stride,
        a_shape, a_stride,
        dtype_code, dtype_code, 2,
        a_ptr, a_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _cosh_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnCosh execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def fast_asinh_inplace(a):
    """In-place asinh_(a) using aclnnAsinh with output aliased to input."""
    _ensure_npu_imports()
    _ensure_ffi_asinh()

    cdef int dev_idx = a.device.index or 0
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    a_stride = a.stride
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_unary_op(
        _asinh_getws_ptr, _asinh_exec_ptr,
        a_shape, a_stride,
        a_shape, a_stride,
        dtype_code, dtype_code, 2,
        a_ptr, a_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _asinh_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnAsinh execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def fast_acosh_inplace(a):
    """In-place acosh_(a) using aclnnAcosh with output aliased to input."""
    _ensure_npu_imports()
    _ensure_ffi_acosh()

    cdef int dev_idx = a.device.index or 0
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    a_stride = a.stride
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_unary_op(
        _acosh_getws_ptr, _acosh_exec_ptr,
        a_shape, a_stride,
        a_shape, a_stride,
        dtype_code, dtype_code, 2,
        a_ptr, a_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _acosh_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnAcosh execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def fast_atanh_inplace(a):
    """In-place atanh_(a) using aclnnAtanh with output aliased to input."""
    _ensure_npu_imports()
    _ensure_ffi_atanh()

    cdef int dev_idx = a.device.index or 0
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    a_stride = a.stride
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_unary_op(
        _atanh_getws_ptr, _atanh_exec_ptr,
        a_shape, a_stride,
        a_shape, a_stride,
        dtype_code, dtype_code, 2,
        a_ptr, a_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _atanh_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnAtanh execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def fast_rsqrt_inplace(a):
    """In-place rsqrt_(a) using aclnnRsqrt with output aliased to input."""
    _ensure_npu_imports()
    _ensure_ffi_rsqrt()

    cdef int dev_idx = a.device.index or 0
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    a_stride = a.stride
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_unary_op(
        _rsqrt_getws_ptr, _rsqrt_exec_ptr,
        a_shape, a_stride,
        a_shape, a_stride,
        dtype_code, dtype_code, 2,
        a_ptr, a_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _rsqrt_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnRsqrt execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def fast_square_inplace(a):
    """In-place square_(a) using aclnnSquare with output aliased to input."""
    _ensure_npu_imports()
    _ensure_ffi_square()

    cdef int dev_idx = a.device.index or 0
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    a_stride = a.stride
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_unary_op(
        _square_getws_ptr, _square_exec_ptr,
        a_shape, a_stride,
        a_shape, a_stride,
        dtype_code, dtype_code, 2,
        a_ptr, a_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _square_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnSquare execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def fast_digamma_inplace(a):
    """In-place digamma_(a) using aclnnDigamma with output aliased to input."""
    _ensure_npu_imports()
    _ensure_ffi_digamma()

    cdef int dev_idx = a.device.index or 0
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    a_stride = a.stride
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_unary_op(
        _digamma_getws_ptr, _digamma_exec_ptr,
        a_shape, a_stride,
        a_shape, a_stride,
        dtype_code, dtype_code, 2,
        a_ptr, a_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _digamma_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnDigamma execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def fast_lgamma_inplace(a):
    """In-place lgamma_(a) using aclnnLgamma with output aliased to input."""
    _ensure_npu_imports()
    _ensure_ffi_lgamma()

    cdef int dev_idx = a.device.index or 0
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    a_stride = a.stride
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_unary_op(
        _lgamma_getws_ptr, _lgamma_exec_ptr,
        a_shape, a_stride,
        a_shape, a_stride,
        dtype_code, dtype_code, 2,
        a_ptr, a_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _lgamma_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnLgamma execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def fast_sign_inplace(a):
    """In-place sign_(a) using aclnnSign with output aliased to input."""
    _ensure_npu_imports()
    _ensure_ffi_sign()

    cdef int dev_idx = a.device.index or 0
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    a_stride = a.stride
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_unary_op(
        _sign_getws_ptr, _sign_exec_ptr,
        a_shape, a_stride,
        a_shape, a_stride,
        dtype_code, dtype_code, 2,
        a_ptr, a_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _sign_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnSign execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def fast_silu_inplace(a):
    """In-place silu_(a) using aclnnSilu with output aliased to input."""
    _ensure_npu_imports()
    _ensure_ffi_silu()

    cdef int dev_idx = a.device.index or 0
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    a_stride = a.stride
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_unary_op(
        _silu_getws_ptr, _silu_exec_ptr,
        a_shape, a_stride,
        a_shape, a_stride,
        dtype_code, dtype_code, 2,
        a_ptr, a_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _silu_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnSilu execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def fast_mish_inplace(a):
    """In-place mish_(a) using aclnnMish with output aliased to input."""
    _ensure_npu_imports()
    _ensure_ffi_mish()

    cdef int dev_idx = a.device.index or 0
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    a_stride = a.stride
    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_unary_op(
        _mish_getws_ptr, _mish_exec_ptr,
        a_shape, a_stride,
        a_shape, a_stride,
        dtype_code, dtype_code, 2,
        a_ptr, a_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _mish_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnMish execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def fast_exp2(a):
    """Optimized out-of-place exp2(a) that calls _ffi.unary_op directly."""
    _ensure_npu_imports()
    _ensure_ffi_exp2()

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

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
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

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
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

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
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

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
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

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
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

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
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

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
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

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
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

    ws_size, executor = _ffi_unary_op(
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
            ret = _ffi_execute(
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
                ret = _ffi_execute(
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
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fa, stream.stream)
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
                ret = _ffi_execute(
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
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_fa, stream.stream)
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
                ret = _ffi_execute(
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



def fast_addcmul_inplace(a, b, c, value):
    """In-place addcmul_(a, b, c, value) — aliases output ptr to a via aclnnAddcmul."""
    _ensure_npu_imports()
    _ensure_ffi_addcmul()

    cdef int dev_idx
    _validate_npu_binary(a, b, "addcmul_", &dev_idx)
    if c.device.type != "npu":
        raise ValueError("NPU addcmul_ expects NPU tensors")
    if c.dtype != a.dtype:
        raise ValueError("NPU addcmul_ requires matching dtypes")

    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    py_a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    py_b_shape = (<TensorImpl>b)._shape_tuple if isinstance(b, TensorImpl) else b.shape
    py_c_shape = (<TensorImpl>c)._shape_tuple if isinstance(c, TensorImpl) else c.shape

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, b_ptr, c_ptr
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

    scalar_handle = _create_scalar_fn(_scalar_bytes_fn(value, a_dtype), dtype_code)
    cdef uintptr_t stream_raw = int(stream.stream)
    try:
        ws_size, executor = _ffi_ref.three_tensor_scalar_op(
            _addcmul_getws_ptr, _addcmul_exec_ptr,
            py_a_shape, a.stride,
            py_b_shape, b.stride,
            py_c_shape, c.stride,
            py_a_shape, a.stride,
            dtype_code, dtype_code, dtype_code, dtype_code, 2,
            a_ptr, b_ptr, c_ptr, a_ptr,
            scalar_handle,
            stream_raw)

        if ws_size:
            workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            try:
                ret = _ffi_execute(
                    _addcmul_exec_ptr, int(workspace_ptr), ws_size,
                    executor, stream_raw)
                if ret != 0:
                    raise RuntimeError(f"aclnnAddcmul execute failed: {ret}")
            finally:
                runtime.defer_raw_free(workspace_ptr)

        _defer_executor_fn(executor)
    finally:
        _destroy_scalar_fn(int(scalar_handle))

    return a


def fast_addcdiv_inplace(a, b, c, value):
    """In-place addcdiv_(a, b, c, value) — aliases output ptr to a via aclnnAddcdiv."""
    _ensure_npu_imports()
    _ensure_ffi_addcdiv()

    cdef int dev_idx
    _validate_npu_binary(a, b, "addcdiv_", &dev_idx)
    if c.device.type != "npu":
        raise ValueError("NPU addcdiv_ expects NPU tensors")
    if c.dtype != a.dtype:
        raise ValueError("NPU addcdiv_ requires matching dtypes")

    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    py_a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    py_b_shape = (<TensorImpl>b)._shape_tuple if isinstance(b, TensorImpl) else b.shape
    py_c_shape = (<TensorImpl>c)._shape_tuple if isinstance(c, TensorImpl) else c.shape

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, b_ptr, c_ptr
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

    scalar_handle = _create_scalar_fn(_scalar_bytes_fn(value, a_dtype), dtype_code)
    cdef uintptr_t stream_raw = int(stream.stream)
    try:
        ws_size, executor = _ffi_ref.three_tensor_scalar_op(
            _addcdiv_getws_ptr, _addcdiv_exec_ptr,
            py_a_shape, a.stride,
            py_b_shape, b.stride,
            py_c_shape, c.stride,
            py_a_shape, a.stride,
            dtype_code, dtype_code, dtype_code, dtype_code, 2,
            a_ptr, b_ptr, c_ptr, a_ptr,
            scalar_handle,
            stream_raw)

        if ws_size:
            workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            try:
                ret = _ffi_execute(
                    _addcdiv_exec_ptr, int(workspace_ptr), ws_size,
                    executor, stream_raw)
                if ret != 0:
                    raise RuntimeError(f"aclnnAddcdiv execute failed: {ret}")
            finally:
                runtime.defer_raw_free(workspace_ptr)

        _defer_executor_fn(executor)
    finally:
        _destroy_scalar_fn(int(scalar_handle))

    return a


def fast_lerp_tensor_inplace(a, b, weight):
    """In-place lerp_(a, b, weight_tensor) — aliases output ptr to a via aclnnLerp."""
    _ensure_npu_imports()
    _ensure_ffi_lerp()

    cdef int dev_idx
    _validate_npu_binary(a, b, "lerp_", &dev_idx)
    if weight.device.type != "npu":
        raise ValueError("NPU lerp_ expects NPU tensors")
    if weight.dtype != a.dtype:
        raise ValueError("NPU lerp_ requires matching dtypes")

    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    py_a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    py_b_shape = (<TensorImpl>b)._shape_tuple if isinstance(b, TensorImpl) else b.shape
    py_w_shape = (<TensorImpl>weight)._shape_tuple if isinstance(weight, TensorImpl) else weight.shape

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, b_ptr, w_ptr
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

    cdef uintptr_t stream_raw = int(stream.stream)
    ws_size, executor = _ffi_ref.four_tensor_op(
        _lerp_getws_ptr, _lerp_exec_ptr,
        py_a_shape, a.stride,
        py_b_shape, b.stride,
        py_w_shape, weight.stride,
        py_a_shape, a.stride,
        dtype_code, dtype_code, dtype_code, dtype_code, 2,
        a_ptr, b_ptr, w_ptr, a_ptr,
        stream_raw)

    if ws_size:
        workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
        if ret != 0:
            raise RuntimeError(f"acl.rt.malloc failed: {ret}")
        try:
            ret = _ffi_execute(
                _lerp_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnLerp execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return a


def fast_lerp_scalar_inplace(a, b, value):
    """In-place lerp_(a, b, scalar) — aliases output ptr to a via aclnnLerps."""
    _ensure_npu_imports()
    _ensure_ffi_lerps()

    cdef int dev_idx
    _validate_npu_binary(a, b, "lerp_", &dev_idx)

    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    py_a_shape = (<TensorImpl>a)._shape_tuple if isinstance(a, TensorImpl) else a.shape
    py_b_shape = (<TensorImpl>b)._shape_tuple if isinstance(b, TensorImpl) else b.shape

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, b_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    if isinstance(b, TensorImpl):
        b_ptr = <uintptr_t>(<TensorImpl>b)._storage._untyped._device_ptr
    else:
        b_ptr = <uintptr_t>b.storage().data_ptr()

    scalar_handle = _create_scalar_fn(_scalar_bytes_fn(value, a_dtype), dtype_code)
    cdef uintptr_t stream_raw = int(stream.stream)
    try:
        ws_size, executor = _ffi_ref.two_tensor_scalar_op(
            _lerps_getws_ptr, _lerps_exec_ptr,
            py_a_shape, a.stride,
            py_b_shape, b.stride,
            py_a_shape, a.stride,
            dtype_code, dtype_code, dtype_code, 2,
            a_ptr, b_ptr, a_ptr,
            scalar_handle,
            stream_raw)

        if ws_size:
            workspace_ptr, ret = _acl_rt_malloc_fn(ws_size, 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            try:
                ret = _ffi_execute(
                    _lerps_exec_ptr, int(workspace_ptr), ws_size,
                    executor, stream_raw)
                if ret != 0:
                    raise RuntimeError(f"aclnnLerps execute failed: {ret}")
            finally:
                runtime.defer_raw_free(workspace_ptr)

        _defer_executor_fn(executor)
    finally:
        _destroy_scalar_fn(int(scalar_handle))

    return a



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

    # 2. Allocator drain: pending events + device synchronize + return cached blocks.
    #    This already provides torch.npu.synchronize() device-wide semantics; doing
    #    an extra aclrtSynchronizeStream first doubles the hot benchmark sync cost.
    alloc = _get_allocator_fn_ref(device_id)
    alloc.synchronize()

    # 3. Flush deferred executors (usually empty, ~0.2 us)
    _flush_executors_fn()

    # 4. Process all three deferred free lists from runtime
    frees = runtime._deferred_frees
    if frees:
        runtime._deferred_frees = []
        for ptr in frees:
            alloc.free_synchronized(ptr)

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


def fast_adam_step(param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq,
                   step, double lr, double beta1, double beta2,
                   double eps, double weight_decay,
                   bint amsgrad, bint maximize):
    """Cython entry point for AdamW step. Replaces the Python orchestration
    that lived in `_backends/npu/ops/optim.py::_adam_step_op`. The actual
    aclnn kernel invocation continues to flow through `aclnn.apply_adam_w_v2`
    which delegates to the Cython `_aclnn_ffi.six_tensor_five_floats_two_bools_op`.
    """
    _ensure_npu_imports()

    if param.device.type != "npu":
        raise ValueError("NPU adam_step expects NPU param tensor")

    cdef int dev_idx = param.device.index or 0
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    p_s = param.storage()
    g_s = grad.storage()
    ea_s = exp_avg.storage()
    eas_s = exp_avg_sq.storage()

    import numpy as _np
    step_np = _np.array([float(step)], dtype=_np.float32)
    step_ptr, _ = _npu_runtime._copy_cpu_to_npu(step_np, runtime=runtime)

    max_v_ptr = None
    if amsgrad and max_exp_avg_sq is not None:
        max_v_ptr = max_exp_avg_sq.storage().data_ptr()

    from candle._backends.npu import aclnn as _aclnn_mod
    _aclnn_mod.apply_adam_w_v2(
        p_s.data_ptr(), ea_s.data_ptr(), eas_s.data_ptr(),
        max_v_ptr, g_s.data_ptr(), step_ptr,
        param.shape, param.stride, (1,), (1,),
        param.dtype,
        lr, beta1, beta2,
        weight_decay, eps,
        amsgrad, maximize,
        runtime=runtime, stream=stream.stream,
    )
    return param


def fast_trace(a):
    """Cython entry point for aclnnTrace. Replaces the Python orchestration
    that lived in `_backends/npu/ops/linalg.py::trace_op`. Sum of diagonal
    elements of a 2-D matrix — output is a 0-d scalar tensor.
    """
    _ensure_npu_imports()
    _ensure_ffi_trace()

    cdef int dev_idx = a.device.index or 0
    a_dev = _device_obj_fast(a)
    a_dtype = a.dtype
    runtime = _get_runtime_fast(dev_idx)
    stream = _get_stream_fast(dev_idx)

    out_shape = ()
    out_stride = ()
    cdef int isize = c_dtype_itemsize(a_dtype)
    cdef int64_t alloc_size_trace = isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_trace, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_trace, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_unary_op(
        _trace_getws_ptr, _trace_exec_ptr,
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
            ret = _ffi_execute(
                _trace_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnTrace execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, 1, a_dtype, a_dev, out_shape, out_stride)


def fast_inverse(a):
    """Cython entry point for aclnnInverse. Replaces the Python orchestration
    that lived in `_backends/npu/ops/linalg.py::linalg_inv`. Output shares
    the input's shape and stride; uses the contiguous shape/stride for the
    output FFI argument.
    """
    _ensure_npu_imports()
    _ensure_ffi_inverse()

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
    cdef int64_t alloc_size_inv = n * isize
    if dev_idx == 0:
        _ensure_allocator_dev0()
        out_ptr = _fast_allocator_dev0.malloc(alloc_size_inv, stream=stream.stream)
    else:
        out_ptr = _get_allocator_fn_ref(dev_idx).malloc(alloc_size_inv, stream=stream.stream)

    cdef int dtype_code = _dtype_to_acl_code(a_dtype)
    cdef uintptr_t a_ptr, o_ptr
    if isinstance(a, TensorImpl):
        a_ptr = <uintptr_t>(<TensorImpl>a)._storage._untyped._device_ptr
    else:
        a_ptr = <uintptr_t>a.storage().data_ptr()
    o_ptr = out_ptr
    cdef uintptr_t stream_raw = int(stream.stream)

    ws_size, executor = _ffi_unary_op(
        _inverse_getws_ptr, _inverse_exec_ptr,
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
            ret = _ffi_execute(
                _inverse_exec_ptr, int(workspace_ptr), ws_size,
                executor, stream_raw)
            if ret != 0:
                raise RuntimeError(f"aclnnInverse execute failed: {ret}")
        finally:
            runtime.defer_raw_free(workspace_ptr)

    _defer_executor_fn(executor)
    return _cy_make_npu_tensor(out_ptr, n, a_dtype, a_dev, out_shape, out_stride)
