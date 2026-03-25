from libc.stdint cimport int64_t

cdef class StorageImpl:
    cdef void* _data_ptr
    cdef int64_t _nbytes
    cdef int _device_type       # 0=cpu, 1=npu, 2=cuda, 3=mps, 4=meta
    cdef int _device_index      # -1 = unset
    cdef object _owner          # prevent GC of backing memory
    cdef bint _resizable

    cpdef size_t data_ptr(self)
    cpdef int64_t nbytes(self)
    cpdef int device_type(self)
    cpdef int device_index(self)
    cpdef bint resizable(self)
