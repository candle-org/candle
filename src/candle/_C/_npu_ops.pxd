from candle._C._tensor_impl cimport TensorImpl

cpdef object fast_add(object a, object b)
cpdef object fast_add_exact(TensorImpl a, TensorImpl b)
cpdef object fast_mul(object a, object b)
cpdef object fast_mul_exact(TensorImpl a, TensorImpl b)
cpdef object fast_silu(object a)
cpdef object fast_silu_exact(TensorImpl a)
cpdef object fast_silu_backward(object grad, object saved_input)
