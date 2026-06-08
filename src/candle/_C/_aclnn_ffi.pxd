from libc.stdint cimport int8_t, int32_t, int64_t, uint64_t, uintptr_t

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

cdef int pta_begin_sdpa_flash_cache_lookup_raw(
    object q_shape, object q_stride,
    object k_shape, object k_stride,
    object v_shape, object v_stride,
    object out_shape, object out_stride,
    object aux_shape, object aux_stride,
    int32_t dtype_code,
    uintptr_t q_ptr, uintptr_t k_ptr, uintptr_t v_ptr,
    uintptr_t out_ptr, uintptr_t softmax_max_ptr, uintptr_t softmax_sum_ptr,
    double scale_value, double keep_prob,
    int64_t pre_tokens, int64_t next_tokens, int64_t head_num,
    int64_t sparse_mode, int64_t inner_precise,
    uintptr_t stream,
    bint* pta_active_out,
    uint64_t* ws_size_out,
    uintptr_t* executor_out,
) except -1

cdef int pta_begin_sdpa_flash_grad_cache_lookup_raw(
    object q_shape, object q_stride,
    object k_shape, object k_stride,
    object v_shape, object v_stride,
    object grad_shape, object grad_stride,
    object out_shape, object out_stride,
    object aux_shape, object aux_stride,
    object dq_shape, object dq_stride,
    object dk_shape, object dk_stride,
    object dv_shape, object dv_stride,
    int32_t dtype_code,
    uintptr_t q_ptr, uintptr_t k_ptr, uintptr_t v_ptr,
    uintptr_t grad_ptr, uintptr_t out_ptr,
    uintptr_t softmax_max_ptr, uintptr_t softmax_sum_ptr,
    uintptr_t dq_ptr, uintptr_t dk_ptr, uintptr_t dv_ptr,
    double scale_value, double keep_prob,
    int64_t pre_tokens, int64_t next_tokens, int64_t head_num,
    int64_t sparse_mode, int64_t inner_precise,
    uintptr_t stream,
    bint* pta_active_out,
    uint64_t* ws_size_out,
    uintptr_t* executor_out,
) except -1

cdef int pta_begin_sdpa_flash_grad_v2_cache_lookup_raw(
    object q_shape, object q_stride,
    object k_shape, object k_stride,
    object v_shape, object v_stride,
    object grad_shape, object grad_stride,
    object out_shape, object out_stride,
    object aux_shape, object aux_stride,
    object dq_shape, object dq_stride,
    object dk_shape, object dk_stride,
    object dv_shape, object dv_stride,
    int32_t dtype_code,
    uintptr_t q_ptr, uintptr_t k_ptr, uintptr_t v_ptr,
    uintptr_t grad_ptr, uintptr_t out_ptr,
    uintptr_t softmax_max_ptr, uintptr_t softmax_sum_ptr,
    uintptr_t dq_ptr, uintptr_t dk_ptr, uintptr_t dv_ptr,
    double scale_value, double keep_prob,
    int64_t pre_tokens, int64_t next_tokens, int64_t head_num,
    int64_t sparse_mode, int64_t inner_precise, int64_t pse_type,
    uintptr_t stream,
    bint* pta_active_out,
    uint64_t* ws_size_out,
    uintptr_t* executor_out,
) except -1

cdef int pta_begin_sdpa_flash_grad_storage_cache_lookup_raw(
    bytes op_name,
    object q_shape, object q_stride,
    object k_shape, object k_stride,
    object v_shape, object v_stride,
    object grad_shape, object grad_stride,
    object out_shape, object out_stride,
    object aux_shape, object aux_stride,
    object dq_shape, object dq_stride,
    object dk_shape, object dk_stride,
    object dv_shape, object dv_stride,
    object dq_storage_shape, object dk_storage_shape, object dv_storage_shape,
    int64_t dq_storage_offset, int64_t dk_storage_offset, int64_t dv_storage_offset,
    int32_t dtype_code,
    uintptr_t q_ptr, uintptr_t k_ptr, uintptr_t v_ptr,
    uintptr_t grad_ptr, uintptr_t out_ptr,
    uintptr_t softmax_max_ptr, uintptr_t softmax_sum_ptr,
    uintptr_t dq_storage_ptr, uintptr_t dk_storage_ptr, uintptr_t dv_storage_ptr,
    double scale_value, double keep_prob,
    int64_t pre_tokens, int64_t next_tokens, int64_t head_num,
    int64_t sparse_mode, int64_t inner_precise, int64_t pse_type,
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

cdef void* create_tensor_raw(
    const int64_t* shape, const int64_t* stride, uint64_t ndim,
    int32_t dtype_code, int32_t fmt, void* data_ptr,
) noexcept nogil

cdef void* create_tensor_raw_with_storage(
    const int64_t* shape, const int64_t* stride, uint64_t ndim,
    const int64_t* storage_dims, uint64_t storage_ndim,
    int64_t storage_offset,
    int32_t dtype_code, int32_t fmt, void* data_ptr,
) noexcept nogil

cdef int32_t destroy_tensor_raw(void* handle) noexcept nogil

cpdef object pta_begin_inplace_copy_cache_lookup(
    object dst_shape, object dst_stride,
    object src_shape, object src_stride,
    int32_t dst_dtype_code, int32_t src_dtype_code,
    uintptr_t dst_ptr, uintptr_t src_ptr,
    uintptr_t stream,
)

cpdef object pta_begin_split_with_size_cache_lookup(
    object self_shape, object self_stride,
    object split_sizes,
    object out_ptrs, object out_shapes, object out_strides,
    int64_t dim,
    int32_t dtype_code,
    uintptr_t self_ptr,
    uintptr_t stream,
)

cpdef object pta_begin_stack_cache_lookup(
    object tensor_ptrs, object shapes, object strides,
    int64_t dim,
    object out_shape, object out_stride,
    int32_t dtype_code,
    uintptr_t out_ptr,
    uintptr_t stream,
)

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

cpdef object layer_norm_op(
    uintptr_t getws_ptr, uintptr_t exec_ptr,
    object input_shape, object input_stride,
    object out_shape, object out_stride,
    object stats_shape, object stats_stride,
    object weight_shape, object weight_stride,
    object bias_shape, object bias_stride,
    object normalized_shape, double eps,
    int32_t dtype_code, int32_t fmt,
    uintptr_t input_ptr, uintptr_t weight_ptr, uintptr_t bias_ptr,
    uintptr_t out_ptr, uintptr_t mean_ptr, uintptr_t rstd_ptr,
    uintptr_t stream,
)

cpdef object split_with_size_op(
    uintptr_t getws_ptr, uintptr_t exec_ptr,
    object self_shape, object self_stride,
    object split_sizes,
    object out_ptrs, object out_shapes, object out_strides,
    int64_t dim,
    int32_t dtype_code, int32_t fmt,
    uintptr_t self_ptr,
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
