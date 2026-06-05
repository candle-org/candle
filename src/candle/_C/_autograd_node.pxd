cdef class AccumulateGrad:
    cdef public object tensor
    cdef public dict _hooks
    cdef public dict _prehooks
    cdef public object _metadata
    cdef public str _name
    cdef dict __dict__
    cdef object __weakref__


cdef class Node:
    cdef public object backward
    cdef public tuple inputs
    cdef public list _saved_tensors_list
    cdef public dict _saved_fields
    cdef public dict _hooks
    cdef public dict _prehooks
    cdef public tuple _next_functions_cache
    cdef public object _metadata
    cdef public str _name
    cdef public object _anomaly_trace
    cdef public object _anomaly_parent
    cdef dict __dict__
    cdef object __weakref__

    cpdef tuple _freeze_next_functions(self)
