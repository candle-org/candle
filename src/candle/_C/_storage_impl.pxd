from libc.stdint cimport int64_t

cdef class StorageImpl:
    cdef void* _data_ptr
    cdef int64_t _nbytes
    cdef int _device_type
    cdef int _device_index
    cdef object _owner
    cdef bint _resizable
    cdef bint _shared
    cdef object _filename
    cdef object _sharing_mechanism
    cdef object _fd
    cdef object _cleanup_finalizer

    cpdef size_t data_ptr(self)
    cpdef int64_t nbytes(self)
    cpdef int device_type(self)
    cpdef int device_index(self)
    cpdef bint resizable(self)
    cpdef bint is_shared(self)
    cpdef object filename(self)
    cpdef object sharing_mechanism(self)
    cpdef object shared_fd(self)
    cpdef object owner(self)
    cpdef object typed_view(self, object dtype, int64_t size)
