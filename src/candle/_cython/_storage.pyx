# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython fast-path for NPU storage creation.

Replaces np.dtype(to_numpy_dtype(dtype)).itemsize with a C switch,
and inlines the TypedStorage construction.
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
    return 4  # fallback


# ---------------------------------------------------------------------------
# Fast storage creation
# ---------------------------------------------------------------------------

# Cached Python class references — for non-NPU paths and fallback
cdef object _NPUUntypedStorage_cls = None
cdef object _TypedStorage_cls = None

cdef inline void _ensure_storage_classes():
    global _NPUUntypedStorage_cls, _TypedStorage_cls
    if _NPUUntypedStorage_cls is None:
        from candle._storage import _NPUUntypedStorage, TypedStorage
        _NPUUntypedStorage_cls = _NPUUntypedStorage
        _TypedStorage_cls = TypedStorage


# Fast NPU storage: Cython cdef classes preferred over Python classes
cdef object _FastNPUStorage_cls = None
cdef object _FastTypedStorage_cls = None

cdef inline void _ensure_fast_storage():
    """Load FastNPUStorage/FastTypedStorage cdef classes (with Python fallback)."""
    global _FastNPUStorage_cls, _FastTypedStorage_cls
    if _FastNPUStorage_cls is not None:
        return
    try:
        from candle._cython._npu_storage import FastNPUStorage, FastTypedStorage  # pylint: disable=import-error,no-name-in-module
        _FastNPUStorage_cls = FastNPUStorage
        _FastTypedStorage_cls = FastTypedStorage
    except ImportError:
        _ensure_storage_classes()
        _FastNPUStorage_cls = _NPUUntypedStorage_cls
        _FastTypedStorage_cls = _TypedStorage_cls


def cy_npu_storage_from_ptr(int64_t device_ptr, int64_t size,
                             object dtype, object device=None):
    """Create TypedStorage from device pointer — fast path.

    Uses FastNPUStorage + FastTypedStorage (Cython cdef classes with __dealloc__)
    when available, falling back to Python _NPUUntypedStorage + TypedStorage.

    Replaces:
    - np.dtype(to_numpy_dtype(dtype)).itemsize  -> C switch
    - _NPUUntypedStorage(ptr, nbytes, device)   -> direct construction
    - TypedStorage(untyped, dtype, size)         -> direct construction
    """
    _ensure_fast_storage()

    cdef int itemsize = _c_dtype_itemsize(dtype)
    cdef int64_t nbytes = size * itemsize
    untyped = _FastNPUStorage_cls(device_ptr, nbytes, device)
    return _FastTypedStorage_cls(untyped, dtype, size)
