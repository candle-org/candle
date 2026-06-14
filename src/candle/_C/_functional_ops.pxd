# cython: language_level=3
"""Public Cython API for _functional_ops module.

Exposes autograd attach functions for use in other Cython modules
(e.g., _tensor_impl.pyx for training-mode forward fast paths).
"""

# Autograd attach functions for training-mode fast paths
cpdef object attach_npu_add_grad(object result, object a, object b)
cpdef object attach_npu_mul_grad(object result, object a, object b)
