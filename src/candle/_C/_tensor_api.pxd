cdef bint _npu_functionalize_active_flag
cdef bint _npu_pipeline_active_flag
cdef bint _npu_profiler_active_flag
cdef bint _npu_autocast_active_flag

cpdef object tensor_add(object self, object other)
cpdef object tensor_sub(object self, object other)
cpdef object tensor_mul(object self, object other)
cpdef object tensor_truediv(object self, object other)
cpdef object tensor_itruediv(object self, object other)
