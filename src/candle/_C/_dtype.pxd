cdef class FastDType:
    cdef public int code
    cdef public int itemsize
    cdef public bint is_floating_point
    cdef public bint is_complex
    cdef public bint is_signed
    cdef public bint _is_quantized
    cdef public str name
    cdef public object _numpy_dtype
