from libc.stdint cimport int32_t, uint64_t, uintptr_t

cpdef object pta_begin_add_cache_lookup(
    object self_shape, object self_stride,
    object other_shape, object other_stride,
    object out_shape, object out_stride,
    int32_t dtype_code,
    uintptr_t self_ptr, uintptr_t other_ptr, uintptr_t out_ptr,
    bytes alpha_scalar_bytes, int32_t alpha_dtype_code,
    uintptr_t stream,
)

cdef int pta_begin_add_cache_lookup_raw(
    object self_shape, object self_stride,
    object other_shape, object other_stride,
    object out_shape, object out_stride,
    int32_t dtype_code,
    uintptr_t self_ptr, uintptr_t other_ptr, uintptr_t out_ptr,
    bytes alpha_scalar_bytes, int32_t alpha_dtype_code,
    uintptr_t stream,
    bint* pta_active_out,
    uint64_t* ws_size_out,
    uintptr_t* executor_out,
) except -1

cpdef object pta_begin_binary_cache_lookup(
    bytes op_name,
    object self_shape, object self_stride,
    object other_shape, object other_stride,
    object out_shape, object out_stride,
    int32_t dtype_code,
    uintptr_t self_ptr, uintptr_t other_ptr, uintptr_t out_ptr,
    uintptr_t stream,
)

cdef int pta_begin_binary_cache_lookup_raw(
    bytes op_name,
    object self_shape, object self_stride,
    object other_shape, object other_stride,
    object out_shape, object out_stride,
    int32_t dtype_code,
    uintptr_t self_ptr, uintptr_t other_ptr, uintptr_t out_ptr,
    uintptr_t stream,
    bint* pta_active_out,
    uint64_t* ws_size_out,
    uintptr_t* executor_out,
) except -1

cpdef object pta_begin_unary_cache_lookup(
    bytes op_name,
    object self_shape, object self_stride,
    object out_shape, object out_stride,
    int32_t self_dtype_code, int32_t out_dtype_code,
    uintptr_t self_ptr, uintptr_t out_ptr,
    uintptr_t stream,
)

cdef int pta_begin_unary_cache_lookup_raw(
    bytes op_name,
    object self_shape, object self_stride,
    object out_shape, object out_stride,
    int32_t self_dtype_code, int32_t out_dtype_code,
    uintptr_t self_ptr, uintptr_t out_ptr,
    uintptr_t stream,
    bint* pta_active_out,
    uint64_t* ws_size_out,
    uintptr_t* executor_out,
) except -1

cpdef void pta_end_cache_lookup()

cpdef int execute(
    uintptr_t exec_ptr,
    uintptr_t workspace_ptr,
    uint64_t workspace_size,
    uintptr_t executor,
    uintptr_t stream,
)

cpdef object binary_op_with_alpha(
    uintptr_t getws_ptr, uintptr_t exec_ptr,
    object self_shape, object self_stride,
    object other_shape, object other_stride,
    object out_shape, object out_stride,
    int32_t dtype_code, int32_t fmt,
    uintptr_t self_ptr, uintptr_t other_ptr, uintptr_t out_ptr,
    uintptr_t alpha_scalar,
    uintptr_t stream,
)

cpdef object binary_op_no_alpha(
    uintptr_t getws_ptr, uintptr_t exec_ptr,
    object self_shape, object self_stride,
    object other_shape, object other_stride,
    object out_shape, object out_stride,
    int32_t dtype_code, int32_t fmt,
    uintptr_t self_ptr, uintptr_t other_ptr, uintptr_t out_ptr,
    uintptr_t stream,
)

cpdef object unary_op(
    uintptr_t getws_ptr, uintptr_t exec_ptr,
    object self_shape, object self_stride,
    object out_shape, object out_stride,
    int32_t self_dtype_code, int32_t out_dtype_code, int32_t fmt,
    uintptr_t self_ptr, uintptr_t out_ptr,
    uintptr_t stream,
)
