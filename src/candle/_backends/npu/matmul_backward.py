"""NPU-specific matmul/addmm backward helper ops."""

from ..common.matmul_backward import (
    layout,
    mm_mat1_backward as _common_mm_mat1_backward,
    mm_mat2_backward as _common_mm_mat2_backward,
    sym_strides,
)


def mm_mat1_backward(grad, mat2, mat1_shape, mat1_strides=None, mat1_layout=None, alpha=1):
    """Gradient for mat1 in addmm/mm: grad @ mat2.T."""
    if alpha == 1:
        try:
            from ..._C._npu_ops import fast_mm_mat1_backward  # pylint: disable=import-error,no-name-in-module
            return fast_mm_mat1_backward(grad, mat2, alpha)
        except (ImportError, ValueError):
            pass
    return _common_mm_mat1_backward(grad, mat2, mat1_shape, mat1_strides, mat1_layout, alpha)


def mm_mat2_backward(grad, mat1, mat2_shape, mat2_strides=None, mat2_layout=None, alpha=1):
    """Gradient for mat2 in addmm/mm: mat1.T @ grad."""
    if alpha == 1:
        try:
            from ..._C._npu_ops import fast_mm_mat2_backward  # pylint: disable=import-error,no-name-in-module
            return fast_mm_mat2_backward(grad, mat1, alpha)
        except (ImportError, ValueError):
            pass
    return _common_mm_mat2_backward(grad, mat1, mat2_shape, mat2_strides, mat2_layout, alpha)
