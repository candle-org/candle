from libc.stdint cimport int8_t, int32_t, uint64_t, uintptr_t

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

cdef int pta_begin_addmm_cache_lookup_raw(
    object input_shape, object input_stride,
    object mat1_shape, object mat1_stride,
    object mat2_shape, object mat2_stride,
    object out_shape, object out_stride,
    int32_t input_dtype_code,
    int32_t mat1_dtype_code,
    int32_t mat2_dtype_code,
    int32_t out_dtype_code,
    uintptr_t input_ptr,
    uintptr_t mat1_ptr,
    uintptr_t mat2_ptr,
    uintptr_t out_ptr,
    bytes beta_scalar_bytes,
    int32_t beta_dtype_code,
    bytes alpha_scalar_bytes,
    int32_t alpha_dtype_code,
    int8_t cube_math_type,
    uintptr_t stream,
    bint* pta_active_out,
    uint64_t* ws_size_out,
    uintptr_t* executor_out,
) except -1

cdef int pta_begin_reduce_sum_cache_lookup_raw(
    object self_shape, object self_stride,
    object out_shape, object out_stride,
    object dims_tuple, bint keepdim,
    int32_t dtype_code,
    uintptr_t self_ptr, uintptr_t out_ptr,
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

cpdef object reduce_sum_op(
    uintptr_t getws_ptr, uintptr_t exec_ptr,
    object self_shape, object self_stride,
    object out_shape, object out_stride,
    object dims_tuple, bint keepdim,
    int32_t dtype_code, int32_t fmt,
    uintptr_t self_ptr, uintptr_t out_ptr,
    uintptr_t stream,
)

cpdef object binary_two_inputs_with_int8_op(
    uintptr_t getws_ptr, uintptr_t exec_ptr,
    object self_shape, object self_stride,
    object other_shape, object other_stride,
    object out_shape, object out_stride,
    int8_t extra_flag,
    int32_t self_dtype_code, int32_t other_dtype_code, int32_t out_dtype_code, int32_t fmt,
    uintptr_t self_ptr, uintptr_t other_ptr, uintptr_t out_ptr,
    uintptr_t stream,
)

cpdef object four_tensor_two_scalars_one_int8_op(
    uintptr_t getws_ptr, uintptr_t exec_ptr,
    object a_shape, object a_stride,
    object b_shape, object b_stride,
    object c_shape, object c_stride,
    object out_shape, object out_stride,
    int8_t cube_math_type,
    int32_t a_dtype_code, int32_t b_dtype_code, int32_t c_dtype_code, int32_t out_dtype_code, int32_t fmt,
    uintptr_t a_ptr, uintptr_t b_ptr, uintptr_t c_ptr, uintptr_t out_ptr,
    uintptr_t scalar_a, uintptr_t scalar_b,
    uintptr_t stream,
)
