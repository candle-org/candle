"""Common matmul/addmm backward helper ops.

These helpers mirror PyTorch's generated autograd formulas for strided dense
matrices. They are metadata/composite helpers: device computation is delegated
back through Candle dispatch so NPU/MPS/CUDA tensors stay on-device.
"""


def sym_strides(input):  # pylint: disable=redefined-builtin
    """Return symbolic strides for generated autograd formulas."""
    return input.stride


def layout(input):  # pylint: disable=redefined-builtin
    """Return tensor layout for generated autograd formulas."""
    return getattr(input, "layout", "strided")


def _scale_if_needed(result, alpha):
    if alpha == 1:
        return result
    from ..._dispatch import dispatch
    return dispatch("mul", result.device.type, result, alpha)


def mm_mat1_backward(grad, mat2, mat1_shape, mat1_strides=None, mat1_layout=None, alpha=1):
    """Gradient for mat1 in addmm/mm: grad @ mat2.T."""
    del mat1_strides, mat1_layout
    from ..._dispatch import dispatch
    if len(grad.shape) == 1 and len(mat1_shape) == 2 and mat1_shape[0] == 1:
        grad = dispatch("reshape", grad.device.type, grad, (1, grad.shape[0]))
    mat2_t = dispatch("transpose", mat2.device.type, mat2, 0, 1)
    return _scale_if_needed(dispatch("matmul", grad.device.type, grad, mat2_t), alpha)


def mm_mat2_backward(grad, mat1, mat2_shape, mat2_strides=None, mat2_layout=None, alpha=1):
    """Gradient for mat2 in addmm/mm: mat1.T @ grad."""
    del mat2_strides, mat2_layout
    from ..._dispatch import dispatch
    if len(grad.shape) == 1 and len(mat2_shape) == 2 and len(mat1.shape) == 2 and mat1.shape[0] == 1:
        grad = dispatch("reshape", grad.device.type, grad, (1, grad.shape[0]))
    mat1_t = dispatch("transpose", mat1.device.type, mat1, 0, 1)
    return _scale_if_needed(dispatch("matmul", grad.device.type, mat1_t, grad), alpha)
