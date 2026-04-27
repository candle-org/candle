# cython: language_level=3, boundscheck=False, wraparound=False
from libc.stdint cimport int64_t
from libc.stdlib cimport malloc, free
from libc.string cimport memset

cdef class StorageImpl:
    def __cinit__(self):
        self._data_ptr = NULL
        self._nbytes = 0
        self._device_type = 0
        self._device_index = -1
        self._owner = None
        self._resizable = False
        self._shared = False
        self._filename = None
        self._sharing_mechanism = None
        self._fd = None
        self._cleanup_finalizer = None

    def __dealloc__(self):
        if self._owner is None and self._data_ptr != NULL and self._device_type == 0 and self._resizable:
            free(self._data_ptr)
            self._data_ptr = NULL

    @staticmethod
    def from_numpy(object arr, bint shared=False, object filename=None, object sharing_mechanism=None, object cleanup_finalizer=None, object fd=None):
        import numpy as np

        cdef StorageImpl s = StorageImpl.__new__(StorageImpl)
        if not isinstance(arr, np.ndarray):
            raise TypeError("expected numpy.ndarray")
        s._data_ptr = <void*><size_t>arr.ctypes.data
        s._nbytes = <int64_t>arr.nbytes
        s._device_type = 0
        s._device_index = -1
        s._owner = arr
        s._resizable = False
        s._shared = shared
        s._filename = filename
        s._sharing_mechanism = sharing_mechanism
        s._fd = fd
        s._cleanup_finalizer = cleanup_finalizer
        return s

    @staticmethod
    def alloc_cpu(int64_t nbytes):
        cdef StorageImpl s = StorageImpl.__new__(StorageImpl)
        s._data_ptr = malloc(<size_t>nbytes)
        if s._data_ptr == NULL:
            raise MemoryError(f"Failed to allocate {nbytes} bytes")
        memset(s._data_ptr, 0, <size_t>nbytes)
        s._nbytes = nbytes
        s._device_type = 0
        s._device_index = -1
        s._owner = None
        s._resizable = True
        s._shared = False
        s._filename = None
        s._sharing_mechanism = None
        s._cleanup_finalizer = None
        return s

    @staticmethod
    def from_device_ptr(size_t ptr, int64_t nbytes, int device_type,
                        int device_index, object owner=None):
        if owner is not None:
            try:
                import numpy as np
                if isinstance(owner, np.ndarray) and device_type != 0:
                    raise TypeError("numpy-backed owner is only valid for CPU storage")
            except ImportError:
                pass
        cdef StorageImpl s = StorageImpl.__new__(StorageImpl)
        s._data_ptr = <void*>ptr
        s._nbytes = nbytes
        s._device_type = device_type
        s._device_index = device_index
        s._owner = owner
        s._resizable = False
        s._shared = False
        s._filename = None
        s._sharing_mechanism = None
        s._cleanup_finalizer = None
        return s

    cpdef size_t data_ptr(self):
        return <size_t>self._data_ptr

    cpdef int64_t nbytes(self):
        return self._nbytes

    cpdef int device_type(self):
        return self._device_type

    cpdef int device_index(self):
        return self._device_index

    cpdef bint resizable(self):
        return self._resizable

    cpdef bint is_shared(self):
        return self._shared

    cpdef object filename(self):
        return self._filename

    cpdef object sharing_mechanism(self):
        return self._sharing_mechanism

    cpdef object shared_fd(self):
        return self._fd

    cpdef object owner(self):
        return self._owner

    cpdef object typed_view(self, object dtype, int64_t size):
        from candle._dtype import to_numpy_dtype
        import numpy as np

        if self._owner is None:
            raise RuntimeError("storage has no Python-visible backing owner")
        return np.frombuffer(self._owner, dtype=to_numpy_dtype(dtype), count=int(size))
