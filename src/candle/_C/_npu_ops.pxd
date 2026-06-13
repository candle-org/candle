from libc.stdint cimport int64_t
from candle._C._tensor_impl cimport TensorImpl

cpdef object fast_add(object a, object b)
cpdef object fast_add_exact(TensorImpl a, TensorImpl b)
cpdef object fast_add_scalar_exact(TensorImpl a, object value)
cpdef object fast_add_scalar_inplace_exact(TensorImpl a, object value)
cpdef object fast_sub_exact(TensorImpl a, TensorImpl b)
cpdef object fast_div_exact(TensorImpl a, TensorImpl b)
cpdef object fast_mul(object a, object b)
cpdef object fast_mul_exact(TensorImpl a, TensorImpl b)
cpdef object fast_mul_scalar_exact(TensorImpl a, object value)
cpdef object fast_sub_scalar_exact(TensorImpl a, object value)
cpdef object fast_div_scalar_exact(TensorImpl a, object value)
cpdef object fast_rsub_scalar_exact(TensorImpl a, object value)
cpdef object fast_rdiv_scalar_exact(TensorImpl a, object value)
cpdef object fast_matmul(object a, object b)
cpdef object fast_matmul_exact(TensorImpl a, TensorImpl b)
cpdef object fast_sdpa_flash_attention(object query, object key, object value, double scale_value, object attn_mask=*)
cpdef object fast_packed_qkv_projection_forward(object packed, int64_t embed,
                                                int64_t heads, bint batch_first)
cpdef object fast_sdpa_flash_attention_backward(object grad_out, object query, object key, object value,
                                                object output, object softmax_max, object softmax_sum,
                                                double scale_value, object attn_mask=*)
cpdef object fast_packed_qkv_projection_backward(object grad_q, object grad_k, object grad_v,
                                                 object input_shape, int64_t embed,
                                                 int64_t heads, bint batch_first)
cpdef object fast_sdpa_flash_attention_backward_packed_qkv(object grad_out, object query, object key, object value,
                                                           object output, object softmax_max, object softmax_sum,
                                                           double scale_value, object input_shape,
                                                           int64_t embed, int64_t heads, bint batch_first)
cpdef object fast_sum(object a, object dim=*, bint keepdim=*)
cpdef object fast_mm_mat1_backward(object grad, object mat2, object alpha=*)
cpdef object fast_mm_mat2_backward(object grad, object mat1, object alpha=*)
cpdef object fast_addmm(object bias, object mat1, object mat2, object beta=*, object alpha=*)
cpdef object fast_layer_norm(object input, object normalized_shape, object weight=*, object bias=*, object eps=*)
cpdef object fast_layer_norm_backward(object grad, object saved_input, object backward_data,
                                      object normalized_shape, object weight=*, object bias=*)
cpdef object fast_gelu(object a)
cpdef object fast_gelu_exact(TensorImpl a)
cpdef object fast_silu(object a)
cpdef object fast_silu_exact(TensorImpl a)
cpdef object fast_rsqrt(object a)
cpdef object fast_cast(object a, object dst_dtype)
cpdef object fast_gelu_backward(object grad, object saved_input)
cpdef object fast_silu_backward(object grad, object saved_input)
cpdef object fast_neg(object a)
cpdef object fast_copy_small_inner_contiguous_view(object view)
cpdef object fast_cat_small_last_dim(object tensors, int64_t dim, object out_shape, object out_stride,
                                     int64_t out_ptr, object first)
cpdef object fast_last_dim_slice_view(object tensor, object key)
