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
