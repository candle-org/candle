# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython fast-path for NPU storage creation.

Replaces np.dtype(to_numpy_dtype(dtype)).itemsize with a C switch,
and constructs the NPU storage wrappers directly in Cython.

NPU path is intentionally Cython-only — no Python fallback.
"""

from libc.stdint cimport int64_t


# ---------------------------------------------------------------------------
# C-level dtype itemsize (no numpy needed)
# ---------------------------------------------------------------------------

cdef int _c_dtype_itemsize(object dtype):
    """Return byte size from a candle dtype object — C switch, no dict."""
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
    return 4


# ---------------------------------------------------------------------------
# NPU storage creation — hard require Cython storage classes
# ---------------------------------------------------------------------------

cdef object _FastNPUStorage_cls = None
cdef object _FastTypedStorage_cls = None


cdef inline void _ensure_fast_storage():
    """Load FastNPUStorage/FastTypedStorage cdef classes.

    NPU path is Cython-only by design. If `_npu_storage` is unavailable,
    import should fail loudly instead of silently falling back to Python.
    """
    global _FastNPUStorage_cls, _FastTypedStorage_cls
    if _FastNPUStorage_cls is not None:
        return
    from candle._cython._npu_storage import FastNPUStorage, FastTypedStorage  # pylint: disable=import-error,no-name-in-module
    _FastNPUStorage_cls = FastNPUStorage
    _FastTypedStorage_cls = FastTypedStorage


def cy_npu_storage_from_ptr(int64_t device_ptr, int64_t size,
                            object dtype, object device=None):
    """Create typed NPU storage from a raw device pointer.

    NPU path is Cython-only: this function always constructs
    FastNPUStorage + FastTypedStorage.
    """
    _ensure_fast_storage()

    cdef int itemsize = _c_dtype_itemsize(dtype)
    cdef int64_t nbytes = size * itemsize

    if device is None:
        from candle._device import device as _Device
        device = _Device("npu")

    untyped = _FastNPUStorage_cls(device_ptr, nbytes, device)
    return _FastTypedStorage_cls(untyped, dtype, size)


# ---------------------------------------------------------------------------
# Tensor class cache — loaded once for cy_make_npu_tensor
# ---------------------------------------------------------------------------

cdef object _Tensor_cls = None
cdef object _StrideTuple_cls = None


cdef inline void _ensure_tensor_cls():
    global _Tensor_cls, _StrideTuple_cls
    if _Tensor_cls is not None:
        return
    from candle._tensor import Tensor, _StrideTuple
    _Tensor_cls = Tensor
    _StrideTuple_cls = _StrideTuple


cdef int _c_dtype_code(object dtype):
    """Map a candle dtype to the TensorImpl._dtype_code integer (C switch)."""
    cdef str name = getattr(dtype, "name", None)
    if name is None:
        s = str(dtype)
        parts = s.split(".")
        name = parts[len(parts) - 1]
    if name == "float32":  return 0
    if name == "float16":  return 1
    if name == "float64":  return 2
    if name == "bfloat16": return 3
    if name == "int32":    return 4
    if name == "int64":    return 5
    if name == "int16":    return 6
    if name == "int8":     return 7
    if name == "uint8":    return 8
    if name == "bool":     return 9
    return -1


# Dispatch key bit constants (must match _tensor_impl.pyx)
DEF _DK_NPU = 1 << 13   # 8192


def cy_make_npu_tensor(int64_t device_ptr, int64_t n_elements,
                       object dtype, object device,
                       tuple shape, object stride):
    """Construct an NPU Tensor entirely in Cython, skipping Python __init__.

    Equivalent to::

        storage = npu_typed_storage_from_ptr(device_ptr, n_elements, dtype, device)
        return Tensor(storage, shape, stride)

    but bypasses the Python __init__ overhead (~27 us savings).

    Uses Python property setters for shape/stride (they call the cdef C-array
    helpers internally) and direct public field writes for all scalar fields.
    """
    _ensure_fast_storage()
    _ensure_tensor_cls()

    cdef int itemsize = _c_dtype_itemsize(dtype)
    cdef int64_t nbytes = n_elements * itemsize

    # 1. FastNPUStorage + FastTypedStorage
    untyped = _FastNPUStorage_cls(device_ptr, nbytes, device)
    typed = _FastTypedStorage_cls(untyped, dtype, n_elements)

    # 2. Tensor via __new__ (skips Python __init__)
    #    Tensor is a Python subclass of TensorImpl (cdef class); all fields
    #    live directly on the object — there is no _impl indirection.
    t = _Tensor_cls.__new__(_Tensor_cls)

    # 3. Shape + stride via Python property setters.
    #    These internally call the cdef _set_shape/_set_stride methods which
    #    populate the C arrays (_c_shape, _c_stride) and Python tuple caches.
    t.shape = shape              # sets _c_shape[], _c_numel, _ndim, _shape_tuple
    t.stride = _StrideTuple_cls(stride)  # sets _c_stride[], _stride_tuple

    # 4. Device fields (public cdef attributes — direct C write, no call overhead).
    #    NPU device_type is always 1 in this fast path.
    t._device_obj = device
    t._device_type = 1           # DeviceType.NPU
    cdef object _idx = getattr(device, "index", None)
    t._device_index = <int>(_idx if _idx is not None else 0)

    # 5. Dtype fields (public cdef attributes — direct C write).
    t._dtype_obj = dtype
    t._itemsize = itemsize
    t._dtype_code = _c_dtype_code(dtype)

    # 6. Pre-computed dispatch keys: NPU + no requires_grad = _DK_NPU only.
    t._dispatch_keys = _DK_NPU

    # 7. Storage
    t._storage = typed

    # 8. Autograd / misc fields (mirrors Tensor.__init__).
    #    All are cdef public — direct C writes.
    t._c_offset = 0
    t.requires_grad = False
    t.grad = None
    t.grad_fn = None
    t._pending = False
    t._retain_grad = False
    t._backward_hooks = None
    t._version_value = 0
    t._vc_proxy = None
    t._base = None
    t._view_meta = None

    return t
