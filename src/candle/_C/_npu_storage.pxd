from libc.stdint cimport int64_t


cdef class FastNPUStorage:
    cdef public int64_t _device_ptr
    cdef public int64_t _nbytes
    cdef public object device
    cdef object _alloc
    cdef public bint _large_fast_free


cdef class FastTypedStorage:
    cdef public FastNPUStorage _untyped
    cdef public object _dtype
    cdef public int64_t _size
