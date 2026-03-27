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
# Stride tuple class cache — loaded once for cy_make_npu_tensor
# ---------------------------------------------------------------------------

cdef object _StrideTuple_cls = None


cdef inline void _ensure_tensor_cls():
    global _StrideTuple_cls
    if _StrideTuple_cls is not None:
        return
    from candle._tensor import _StrideTuple
    _StrideTuple_cls = _StrideTuple


def cy_make_npu_tensor(int64_t device_ptr, int64_t n_elements,
                       object dtype, object device,
                       tuple shape, object stride):
    """Construct an NPU Tensor entirely in Cython via the unified tensor factory.

    Equivalent to::

        storage = npu_typed_storage_from_ptr(device_ptr, n_elements, dtype, device)
        return Tensor(storage, shape, stride)

    Routes through cy_make_tensor_from_storage so all tensor births share a
    single initialisation path.
    """
    from candle._cython._tensor_impl import cy_make_tensor_from_storage

    _ensure_fast_storage()
    _ensure_tensor_cls()

    cdef int itemsize = _c_dtype_itemsize(dtype)
    cdef int64_t nbytes = n_elements * itemsize

    # 1. FastNPUStorage + FastTypedStorage (unchanged)
    untyped = _FastNPUStorage_cls(device_ptr, nbytes, device)
    typed = _FastTypedStorage_cls(untyped, dtype, n_elements)

    # 2. Delegate all field initialisation to the unified factory
    return cy_make_tensor_from_storage(
        typed,
        shape,
        _StrideTuple_cls(stride),
        0,
        False,
    )
