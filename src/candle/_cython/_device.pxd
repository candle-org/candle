cdef class FastDevice:
    cdef public int type_code
    cdef int _index
    cdef str _type_str

    cdef void _assign_type_code(self)
