import math
import ctypes
import struct
import numpy as np

from ._helpers import (
    _can_use_gpu, _metal_buf, _kernel_suffix, _scalar_fmt, _itemsize,
    _alloc_output_buf, _metal_buf_to_bytes, _from_metal_buffer,
    _get_dispatcher, _dispatch_unary_gpu, _dispatch_unary_predicate_gpu,
    _scalar_value, _dispatch_binary_gpu,
    _to_numpy, _from_numpy,
    _compute_reduce_dims, _reduce_shape, _gpu_reduce_single_dim,
    _normalize_tensor_sequence_args,
    _can_use_blas, _blas_gemm,
    float32_dtype, float16_dtype, float64_dtype,
    int32_dtype, int64_dtype, bool_dtype,
    to_numpy_dtype, Tensor,
    mps_typed_storage_from_numpy, _MPSUntypedStorage, TypedStorage,
    _accel,
)
from .math import mul, add, sub, abs as gpu_abs, sqrt as gpu_sqrt, pow as gpu_pow, div


def matmul(a, b):
    # GPU path: MPSMatrixMultiplication for 2D contiguous float32/float16
    if (_can_use_gpu(a) and _can_use_gpu(b)
            and len(a.shape) == 2 and len(b.shape) == 2):
        from ..mps_kernels import mps_matmul_gpu, _mps_dtype_code
        np_dt = to_numpy_dtype(a.dtype)
        dtype_code = _mps_dtype_code(np_dt)
        if dtype_code is not None:
            M, K = a.shape
            K2, N = b.shape
            if K == K2:
                out_buf = mps_matmul_gpu(
                    _metal_buf(a), _metal_buf(b),
                    M, K, N, dtype_code, _itemsize(a.dtype))
                if out_buf is not None:
                    from ...._tensor import _compute_strides
                    out_shape = (M, N)
                    out_stride = _compute_strides(out_shape)
                    return _from_metal_buffer(out_buf, out_shape, out_stride,
                                             a.dtype, a.device)
    # GPU path: 3D batched matmul — loop GPU 2D matmul per batch, then stack
    if (_can_use_gpu(a) and _can_use_gpu(b)
            and len(a.shape) == 3 and len(b.shape) == 3):
        from .shape import stack as gpu_stack, select
        batch = a.shape[0]
        slices = []
        for i in range(batch):
            ai = select(a, 0, i)  # a[i] — 2D view
            bi = select(b, 0, i)  # b[i] — 2D view
            slices.append(matmul(ai, bi))
        return gpu_stack(slices, dim=0)
    # CPU path (Accelerate BLAS or numpy)
    a_np = _to_numpy(a)
    b_np = _to_numpy(b)
    if a_np.ndim == 2 and b_np.ndim == 2:
        a_c = np.ascontiguousarray(a_np)
        b_c = np.ascontiguousarray(b_np)
        if _can_use_blas(a_c) and _can_use_blas(b_c):
            return _from_numpy(_blas_gemm(a_c, b_c, a.dtype), a.dtype, a.device)
    if a_np.ndim == 3 and b_np.ndim == 3:
        a_c = np.ascontiguousarray(a_np)
        b_c = np.ascontiguousarray(b_np)
        if a_c.dtype in (np.float32, np.float64) and _accel.available():
            batch = a_c.shape[0]
            M, K = a_c.shape[1], a_c.shape[2]
            N = b_c.shape[2]
            out = np.empty((batch, M, N), dtype=a_c.dtype)
            for i in range(batch):
                out[i] = _blas_gemm(a_c[i], b_c[i], a.dtype)
            return _from_numpy(out, a.dtype, a.device)
    return _from_numpy(a_np @ b_np, a.dtype, a.device)

def mm(a, b):
    return matmul(a, b)

def bmm(a, b):
    return matmul(a, b)

def dot(a, b):
    if a.dtype != b.dtype:
        raise RuntimeError("dot: expected both vectors to have same dtype")
    # GPU composite: sum(mul(a, b))
    if _can_use_gpu(a) and _can_use_gpu(b):
        from .reduce import sum_
        return sum_(mul(a, b))
    # CPU path (Accelerate BLAS or numpy)
    a_np = np.ascontiguousarray(_to_numpy(a))
    b_np = np.ascontiguousarray(_to_numpy(b))
    if _can_use_blas(a_np) and _can_use_blas(b_np):
        n = a_np.size
        if a_np.dtype == np.float32:
            val = _accel.cblas_sdot(n, a_np.ctypes.data, 1, b_np.ctypes.data, 1)
        else:
            val = _accel.cblas_ddot(n, a_np.ctypes.data, 1, b_np.ctypes.data, 1)
        return _from_numpy(np.array(val, dtype=a_np.dtype), a.dtype, a.device)
    return _from_numpy(np.dot(a_np, b_np), a.dtype, a.device)

def outer(a, b):
    # GPU composite: matmul(a.reshape(-1,1), b.reshape(1,-1))
    if _can_use_gpu(a) and _can_use_gpu(b):
        return matmul(a.reshape(-1, 1), b.reshape(1, -1))
    return _from_numpy(np.outer(_to_numpy(a), _to_numpy(b)), a.dtype, a.device)

def inner(a, b):
    # GPU composite: for 1D tensors, same as dot
    if _can_use_gpu(a) and _can_use_gpu(b) and len(a.shape) == 1 and len(b.shape) == 1:
        return dot(a, b)
    return _from_numpy(np.inner(_to_numpy(a), _to_numpy(b)), a.dtype, a.device)

def mv(a, b):
    # GPU composite: matmul(a, b.reshape(-1,1)).reshape(-1)
    if _can_use_gpu(a) and _can_use_gpu(b) and len(a.shape) == 2 and len(b.shape) == 1:
        return matmul(a, b.reshape(-1, 1)).reshape(-1)
    # CPU path (Accelerate BLAS or numpy)
    a_np = np.ascontiguousarray(_to_numpy(a))
    b_np = np.ascontiguousarray(_to_numpy(b))
    if a_np.ndim == 2 and b_np.ndim == 1 and _can_use_blas(a_np) and _can_use_blas(b_np):
        M, N = a_np.shape
        out = np.empty(M, dtype=a_np.dtype)
        if a_np.dtype == np.float32:
            _accel.cblas_sgemv(111, M, N, 1.0,
                               a_np.ctypes.data, N, b_np.ctypes.data, 1,
                               0.0, out.ctypes.data, 1)
        else:
            return _from_numpy(np.dot(a_np, b_np), a.dtype, a.device)
        return _from_numpy(out, a.dtype, a.device)
    return _from_numpy(np.dot(a_np, b_np), a.dtype, a.device)

def cross(a, b, dim=-1):
    if a.dtype != b.dtype:
        raise RuntimeError("cross: expected both inputs to have same dtype")
    # GPU composite: extract 3 components via narrow, compute cross product
    if _can_use_gpu(a) and _can_use_gpu(b):
        from .shape import narrow, cat
        ndim = len(a.shape)
        d = dim if dim >= 0 else dim + ndim
        a0 = narrow(a, d, 0, 1)
        a1 = narrow(a, d, 1, 1)
        a2 = narrow(a, d, 2, 1)
        b0 = narrow(b, d, 0, 1)
        b1 = narrow(b, d, 1, 1)
        b2 = narrow(b, d, 2, 1)
        c0 = sub(mul(a1, b2), mul(a2, b1))
        c1 = sub(mul(a2, b0), mul(a0, b2))
        c2 = sub(mul(a0, b1), mul(a1, b0))
        return cat([c0, c1, c2], dim=d)
    a_np = np.moveaxis(_to_numpy(a), dim, -1)
    b_np = np.moveaxis(_to_numpy(b), dim, -1)
    out = np.cross(a_np, b_np)
    out = np.moveaxis(out, -1, dim)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)

def tensordot(a, b, dims=2):
    if a.dtype != b.dtype:
        raise RuntimeError("tensordot: expected both inputs to have same dtype")
    # GPU composite for integer dims: reshape to 2D → matmul → reshape
    if isinstance(dims, int) and _can_use_gpu(a) and _can_use_gpu(b):
        ndim_a = len(a.shape)
        ndim_b = len(b.shape)
        # Contract last `dims` of a with first `dims` of b
        a_free = a.shape[:ndim_a - dims]
        a_contract = a.shape[ndim_a - dims:]
        b_contract = b.shape[:dims]
        b_free = b.shape[dims:]
        if a_contract != b_contract:
            raise RuntimeError("tensordot: contraction dimensions mismatch")
        contract_size = 1
        for s in a_contract:
            contract_size *= s
        a_rows = 1
        for s in a_free:
            a_rows *= s
        b_cols = 1
        for s in b_free:
            b_cols *= s
        a_2d = a.contiguous().reshape(a_rows, contract_size)
        b_2d = b.contiguous().reshape(contract_size, b_cols)
        result_2d = matmul(a_2d, b_2d)
        out_shape = a_free + b_free
        if not out_shape:
            out_shape = (1,)
            return result_2d.reshape(out_shape).squeeze(0)
        return result_2d.reshape(out_shape)
    # CPU/numpy fallback (also handles axis-list dims)
    a_np = _to_numpy(a)
    b_np = _to_numpy(b)
    if isinstance(dims, int):
        out = np.tensordot(a_np, b_np, axes=dims)
    elif isinstance(dims, (list, tuple)) and len(dims) == 2:
        out = np.tensordot(a_np, b_np, axes=dims)
    else:
        out = np.tensordot(a_np, b_np, axes=dims)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)

def einsum(equation, *operands):
    if len(operands) == 1 and isinstance(operands[0], (list, tuple)):
        operands = operands[0]
    # GPU composite for common 2-operand patterns
    if len(operands) == 2 and _can_use_gpu(operands[0]) and _can_use_gpu(operands[1]):
        from .reduce import sum_
        a, b = operands
        eq = equation.replace(' ', '')
        if eq in ('ij,jk->ik', 'bij,bjk->bik'):
            return matmul(a, b)
        if eq == 'ij,ij->ij':
            return mul(a, b)
        if eq == 'ij,ij->':
            return sum_(mul(a, b))
        if eq == 'ij,ij->i':
            return sum_(mul(a, b), dim=-1)
    ops_np = [_to_numpy(op) for op in operands]
    out = np.einsum(equation, *ops_np)
    return _from_numpy(np.ascontiguousarray(out), operands[0].dtype, operands[0].device)


# ---------------------------------------------------------------------------
# Group 2: Logical ops
# ---------------------------------------------------------------------------

def addmm(input, mat1, mat2, beta=1, alpha=1):
    """addmm: beta * input + alpha * (mat1 @ mat2)."""
    # GPU composite path: matmul(GPU) then scale+add(GPU)
    if (_can_use_gpu(mat1) and _can_use_gpu(mat2)
            and len(mat1.shape) == 2 and len(mat2.shape) == 2
            and mat1.dtype in (float32_dtype, float16_dtype)):
        mm_result = matmul(mat1, mat2)
        if alpha != 1:
            mm_result = mul(mm_result, alpha)
        if beta == 0:
            return mm_result
        if beta != 1:
            input = mul(input, beta)
        return add(input, mm_result)
    inp = _to_numpy(input)
    m1 = np.ascontiguousarray(_to_numpy(mat1))
    m2 = np.ascontiguousarray(_to_numpy(mat2))
    if _can_use_blas(m1) and _can_use_blas(m2):
        M, K = m1.shape
        N = m2.shape[1]
        # out = alpha * m1 @ m2
        out = np.empty((M, N), dtype=m1.dtype)
        if m1.dtype == np.float32:
            # Prepare C = beta * input first, then sgemm adds alpha*A*B
            out[:] = (np.broadcast_to(inp, (M, N)) * beta).astype(np.float32)
            _accel.cblas_sgemm(111, 111, M, N, K, float(alpha),
                               m1.ctypes.data, K, m2.ctypes.data, N,
                               1.0, out.ctypes.data, N)
        else:
            out[:] = (np.broadcast_to(inp, (M, N)) * beta).astype(np.float64)
            _accel.cblas_dgemm(111, 111, M, N, K, float(alpha),
                               m1.ctypes.data, K, m2.ctypes.data, N,
                               1.0, out.ctypes.data, N)
        return _from_numpy(out, input.dtype, input.device)
    out = beta * inp + alpha * np.dot(m1, m2)
    return _from_numpy(np.ascontiguousarray(out), input.dtype, input.device)

def baddbmm(input, batch1, batch2, beta=1, alpha=1):
    """Batch matrix-matrix product: beta * input + alpha * (batch1 @ batch2)."""
    if batch1.ndim != 3 or batch2.ndim != 3:
        raise RuntimeError("baddbmm: batch1 and batch2 must be 3-D tensors")
    # GPU composite: bmm(GPU) then scale+add(GPU)
    if (_can_use_gpu(batch1) and _can_use_gpu(batch2)
            and batch1.dtype in (float32_dtype, float16_dtype)):
        bmm_result = matmul(batch1, batch2)
        if alpha != 1:
            bmm_result = mul(bmm_result, alpha)
        if beta == 0:
            return bmm_result
        if beta != 1:
            input = mul(input, beta)
        return add(input, bmm_result)
    input_np = _to_numpy(input)
    batch1_np = _to_numpy(batch1)
    batch2_np = _to_numpy(batch2)
    bmm_result = batch1_np @ batch2_np
    out = beta * input_np + alpha * bmm_result
    return _from_numpy(np.ascontiguousarray(out), input.dtype, input.device)

def linalg_qr(a, mode='reduced'):
    """QR decomposition on CPU via numpy."""
    arr = _to_numpy(a)
    np_mode = mode
    q, r = np.linalg.qr(arr, mode=np_mode)
    return _from_numpy(q, a.dtype, a.device), _from_numpy(r, a.dtype, a.device)



# ---------------------------------------------------------------------------
# Tensor indexing / selection ops
# ---------------------------------------------------------------------------

def linalg_inv(a):
    """Inverse of a square matrix."""
    arr = _to_numpy(a).astype(np.float64)
    out = np.linalg.inv(arr)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)

def linalg_vector_norm(a, ord=2, dim=None, keepdim=False):
    """Vector norm."""
    # GPU composite for common orders
    if _can_use_gpu(a) and ord != 0:
        from .reduce import sum_, amax, amin
        abs_a = gpu_abs(a)
        if ord == 2:
            return gpu_sqrt(sum_(mul(a, a), dim=dim, keepdim=keepdim))
        if ord == 1:
            return sum_(abs_a, dim=dim, keepdim=keepdim)
        if ord == float('inf'):
            return amax(abs_a, dim=dim, keepdim=keepdim)
        if ord == float('-inf'):
            return amin(abs_a, dim=dim, keepdim=keepdim)
        # General p-norm: pow(sum(pow(abs(a), p), dim), 1/p)
        powered = gpu_pow(abs_a, ord)
        summed = sum_(powered, dim=dim, keepdim=keepdim)
        return gpu_pow(summed, 1.0 / ord)
    # CPU/numpy fallback (also handles ord=0)
    arr = _to_numpy(a).astype(np.float64)
    if dim is not None:
        if isinstance(dim, (list, tuple)):
            axis = tuple(dim)
        else:
            axis = dim
    else:
        axis = None
    if ord == float('inf'):
        out = np.max(np.abs(arr), axis=axis, keepdims=keepdim)
    elif ord == float('-inf'):
        out = np.min(np.abs(arr), axis=axis, keepdims=keepdim)
    elif ord == 0:
        out = np.sum(arr != 0, axis=axis, keepdims=keepdim).astype(np.float64)
    else:
        out = np.sum(np.abs(arr) ** ord, axis=axis, keepdims=keepdim) ** (1.0 / ord)
    return _from_numpy(np.ascontiguousarray(np.atleast_1d(out).astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)


# ---------------------------------------------------------------------------
# torch.fft ops
# ---------------------------------------------------------------------------

def linalg_norm(a, ord=None, dim=None, keepdim=False):
    """Vector or matrix norm."""
    # GPU composite: delegate to vector_norm or matrix_norm
    if _can_use_gpu(a):
        if isinstance(dim, (list, tuple)) and len(dim) == 2:
            return linalg_matrix_norm(a, ord=ord if ord is not None else 'fro',
                                      dim=dim, keepdim=keepdim)
        # vector norm: ord=None means ord=2
        return linalg_vector_norm(a, ord=ord if ord is not None else 2,
                                  dim=dim, keepdim=keepdim)
    arr = _to_numpy(a).astype(np.float64)
    if dim is not None:
        if isinstance(dim, (list, tuple)):
            axis = tuple(dim)
        else:
            axis = dim
    else:
        axis = None
    out = np.linalg.norm(arr, ord=ord, axis=axis, keepdims=keepdim)
    return _from_numpy(np.ascontiguousarray(np.atleast_1d(out).astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)

def linalg_matrix_norm(a, ord='fro', dim=(-2, -1), keepdim=False):
    """Matrix norm."""
    # GPU composite for Frobenius norm
    if _can_use_gpu(a) and ord == 'fro':
        from .reduce import sum_
        return gpu_sqrt(sum_(mul(a, a), dim=list(dim), keepdim=keepdim))
    arr = _to_numpy(a).astype(np.float64)
    if isinstance(dim, (list, tuple)) and len(dim) == 2:
        axis = tuple(dim)
    else:
        axis = dim
    if ord == 'fro':
        out = np.sqrt(np.sum(arr ** 2, axis=axis, keepdims=keepdim))
    elif ord == 'nuc':
        # Nuclear norm = sum of singular values
        out = np.sum(np.linalg.svd(arr, compute_uv=False), axis=-1, keepdims=keepdim)
    else:
        out = np.linalg.norm(arr, ord=ord, axis=axis, keepdims=keepdim)
    return _from_numpy(np.ascontiguousarray(np.atleast_1d(out).astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)

def linalg_multi_dot(tensors):
    """Efficiently multiply 2+ matrices using chained matmul."""
    # GPU composite: left-to-right chain of GPU matmul
    if all(_can_use_gpu(t) for t in tensors):
        result = tensors[0]
        for t in tensors[1:]:
            result = matmul(result, t)
        return result
    arrays = [_to_numpy(t).astype(np.float64) for t in tensors]
    out = np.linalg.multi_dot(arrays)
    dt = tensors[0].dtype
    dev = tensors[0].device
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(dt))), dt, dev)

def linalg_matrix_power(a, n):
    """Matrix raised to integer power n."""
    arr = _to_numpy(a).astype(np.float64)
    out = np.linalg.matrix_power(arr, n)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)

def linalg_vander(x, N=None):
    """Vandermonde matrix."""
    arr = _to_numpy(x)
    n = N if N is not None else len(arr)
    out = np.vander(arr, N=n, increasing=True)
    return _from_numpy(np.ascontiguousarray(out), x.dtype, x.device)

def linalg_cholesky(a, upper=False):
    """Cholesky decomposition."""
    arr = _to_numpy(a).astype(np.float64)
    L = np.linalg.cholesky(arr)
    if upper:
        # Transpose the last two dims
        L = np.swapaxes(L, -2, -1).conj()
    return _from_numpy(np.ascontiguousarray(L.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)

def linalg_cond(a, p=None):
    """Condition number of a matrix."""
    arr = _to_numpy(a).astype(np.float64)
    out = np.linalg.cond(arr, p=p)
    return _from_numpy(np.ascontiguousarray(np.atleast_1d(out).astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)

def linalg_det(a):
    """Determinant of a square matrix."""
    arr = _to_numpy(a).astype(np.float64)
    out = np.linalg.det(arr)
    return _from_numpy(np.ascontiguousarray(np.atleast_1d(out).astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)

def linalg_slogdet(a):
    """Sign and log absolute value of determinant."""
    from collections import namedtuple
    arr = _to_numpy(a).astype(np.float64)
    sign, logabsdet = np.linalg.slogdet(arr)
    SlogdetResult = namedtuple("SlogdetResult", ["sign", "logabsdet"])
    dt = to_numpy_dtype(a.dtype)
    return SlogdetResult(
        _from_numpy(np.ascontiguousarray(np.atleast_1d(sign).astype(dt)), a.dtype, a.device),
        _from_numpy(np.ascontiguousarray(np.atleast_1d(logabsdet).astype(dt)), a.dtype, a.device),
    )

def linalg_eig(a):
    """Eigenvalue decomposition of a square matrix."""
    from ...._dtype import complex128 as complex128_dtype
    arr = _to_numpy(a).astype(np.float64)
    eigenvalues, eigenvectors = np.linalg.eig(arr)
    return (
        _from_numpy(np.ascontiguousarray(eigenvalues), complex128_dtype, a.device),
        _from_numpy(np.ascontiguousarray(eigenvectors), complex128_dtype, a.device),
    )

def linalg_eigh(a, UPLO='L'):
    """Eigenvalue decomposition of a symmetric/Hermitian matrix."""
    arr = _to_numpy(a).astype(np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(arr, UPLO=UPLO)
    return (
        _from_numpy(np.ascontiguousarray(eigenvalues.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device),
        _from_numpy(np.ascontiguousarray(eigenvectors.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device),
    )

def linalg_eigvals(a):
    """Eigenvalues of a square matrix."""
    from ...._dtype import complex128 as complex128_dtype
    arr = _to_numpy(a).astype(np.float64)
    out = np.linalg.eigvals(arr)
    return _from_numpy(np.ascontiguousarray(out), complex128_dtype, a.device)

def linalg_eigvalsh(a, UPLO='L'):
    """Eigenvalues of a symmetric/Hermitian matrix."""
    arr = _to_numpy(a).astype(np.float64)
    out = np.linalg.eigvalsh(arr, UPLO=UPLO)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)

def linalg_householder_product(input, tau):
    """Computes the first n columns of the product of Householder matrices."""
    A = _to_numpy(input).astype(np.float64)
    tau_np = _to_numpy(tau).astype(np.float64)
    m, n = A.shape[-2], A.shape[-1]
    k = tau_np.shape[-1]
    Q = np.eye(m, dtype=np.float64)
    for i in range(k):
        v = np.zeros(m, dtype=np.float64)
        v[i] = 1.0
        v[i + 1:] = A[i + 1:, i]
        Q = Q - tau_np[i] * np.outer(Q @ v, v)
    Q = Q[:, :n]
    return _from_numpy(np.ascontiguousarray(Q.astype(to_numpy_dtype(input.dtype))), input.dtype, input.device)

def linalg_lstsq(a, b, rcond=None, driver=None):
    """Least-squares solution to a linear matrix equation."""
    from collections import namedtuple
    a_np = _to_numpy(a).astype(np.float64)
    b_np = _to_numpy(b).astype(np.float64)
    if rcond is None:
        rcond = -1 if np.lib.NumpyVersion(np.__version__) < '2.0.0' else None
    solution, residuals, rank, singular_values = np.linalg.lstsq(a_np, b_np, rcond=rcond)
    LstsqResult = namedtuple("LstsqResult", ["solution", "residuals", "rank", "singular_values"])
    return LstsqResult(
        _from_numpy(np.ascontiguousarray(solution.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device),
        _from_numpy(np.ascontiguousarray(np.atleast_1d(residuals).astype(to_numpy_dtype(a.dtype))), a.dtype, a.device),
        rank,
        _from_numpy(np.ascontiguousarray(singular_values.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device),
    )

def linalg_lu(a, pivot=True):
    """LU decomposition with partial pivoting."""
    from collections import namedtuple
    from scipy import linalg as scipy_linalg
    arr = _to_numpy(a).astype(np.float64)
    P_mat, L, U = scipy_linalg.lu(arr)
    LUResult = namedtuple("LUResult", ["P", "L", "U"])
    dt = to_numpy_dtype(a.dtype)
    return LUResult(
        _from_numpy(np.ascontiguousarray(P_mat.astype(dt)), a.dtype, a.device),
        _from_numpy(np.ascontiguousarray(L.astype(dt)), a.dtype, a.device),
        _from_numpy(np.ascontiguousarray(U.astype(dt)), a.dtype, a.device),
    )

def linalg_lu_factor(a, pivot=True):
    """Compact LU factorization."""
    from collections import namedtuple
    from scipy import linalg as scipy_linalg
    arr = _to_numpy(a).astype(np.float64)
    lu, piv = scipy_linalg.lu_factor(arr)
    LUFactorResult = namedtuple("LUFactorResult", ["LU", "pivots"])
    return LUFactorResult(
        _from_numpy(np.ascontiguousarray(lu.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device),
        _from_numpy(np.ascontiguousarray(piv.astype(np.int32)), int64_dtype, a.device),
    )

def linalg_lu_solve(LU, pivots, B, left=True, adjoint=False):
    """Solve using LU factorization."""
    from scipy import linalg as scipy_linalg
    lu_np = _to_numpy(LU).astype(np.float64)
    piv_np = _to_numpy(pivots).astype(np.int32)
    b_np = _to_numpy(B).astype(np.float64)
    if not left:
        b_np = b_np.T
    trans = 1 if adjoint else 0
    out = scipy_linalg.lu_solve((lu_np, piv_np), b_np, trans=trans)
    if not left:
        out = out.T
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(B.dtype))), B.dtype, B.device)

def linalg_matrix_exp(a):
    """Matrix exponential."""
    from scipy import linalg as scipy_linalg
    arr = _to_numpy(a).astype(np.float64)
    out = scipy_linalg.expm(arr)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)

def linalg_matrix_rank(a, atol=None, rtol=None, hermitian=False):
    """Numerical rank of a matrix."""
    arr = _to_numpy(a).astype(np.float64)
    if atol is not None or rtol is not None:
        s = np.linalg.svd(arr, compute_uv=False)
        tol = 0.0
        if atol is not None:
            tol = max(tol, atol)
        if rtol is not None:
            tol = max(tol, rtol * s[..., 0])
        rank = np.sum(s > tol, axis=-1)
    else:
        rank = np.linalg.matrix_rank(arr)
    return _from_numpy(np.ascontiguousarray(np.atleast_1d(rank).astype(np.int64)), int64_dtype, a.device)

def linalg_pinv(a, atol=None, rtol=None, hermitian=False):
    """Moore-Penrose pseudoinverse."""
    arr = _to_numpy(a).astype(np.float64)
    if rtol is not None:
        rcond = rtol
    elif atol is not None:
        s_max = np.linalg.svd(arr, compute_uv=False)[..., 0]
        rcond = atol / s_max
    else:
        rcond = 1e-15
    out = np.linalg.pinv(arr, rcond=rcond)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)

def linalg_solve(a, b, left=True):
    """Solve a square system of linear equations."""
    a_np = _to_numpy(a).astype(np.float64)
    b_np = _to_numpy(b).astype(np.float64)
    if not left:
        # X @ A = B => A^T @ X^T = B^T
        out = np.linalg.solve(a_np.T, b_np.T).T
    else:
        out = np.linalg.solve(a_np, b_np)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)

def linalg_solve_triangular(a, b, upper, left=True, unitriangular=False):
    """Solve a triangular system."""
    from scipy import linalg as scipy_linalg
    a_np = _to_numpy(a).astype(np.float64)
    b_np = _to_numpy(b).astype(np.float64)
    if not left:
        a_np = a_np.T
        b_np = b_np.T
        upper = not upper
    out = scipy_linalg.solve_triangular(a_np, b_np, lower=not upper, unit_diagonal=unitriangular)
    if not left:
        out = out.T
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)

def linalg_svd(a, full_matrices=True):
    """Singular value decomposition."""
    arr = _to_numpy(a).astype(np.float64)
    U, S, Vh = np.linalg.svd(arr, full_matrices=full_matrices)
    dt = to_numpy_dtype(a.dtype)
    return (
        _from_numpy(np.ascontiguousarray(U.astype(dt)), a.dtype, a.device),
        _from_numpy(np.ascontiguousarray(S.astype(dt)), a.dtype, a.device),
        _from_numpy(np.ascontiguousarray(Vh.astype(dt)), a.dtype, a.device),
    )

def linalg_svdvals(a):
    """Singular values of a matrix."""
    arr = _to_numpy(a).astype(np.float64)
    out = np.linalg.svd(arr, compute_uv=False)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)

def linalg_tensorinv(a, ind=2):
    """Tensor inverse (generalization of matrix inverse)."""
    arr = _to_numpy(a).astype(np.float64)
    out = np.linalg.tensorinv(arr, ind=ind)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)

def linalg_tensorsolve(a, b, dims=None):
    """Solve a tensor equation."""
    a_np = _to_numpy(a).astype(np.float64)
    b_np = _to_numpy(b).astype(np.float64)
    axes = None if dims is None else tuple(dims)
    out = np.linalg.tensorsolve(a_np, b_np, axes=axes)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)

def matrix_power(a, n):
    """Matrix raised to the integer power n."""
    arr = _to_numpy(a)
    if arr.ndim < 2:
        raise RuntimeError(
            f"matrix_power: input must be at least 2-D, got {arr.ndim}-D"
        )
    if arr.shape[-2] != arr.shape[-1]:
        raise RuntimeError(
            f"matrix_power: input must be a square matrix, got shape {arr.shape}"
        )
    out = np.linalg.matrix_power(arr, n)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)

def det(a):
    """Determinant of a square matrix (or batch of square matrices)."""
    arr = _to_numpy(a)
    if arr.ndim < 2:
        raise RuntimeError(f"det: input must be at least 2-D, got {arr.ndim}-D")
    if arr.shape[-2] != arr.shape[-1]:
        raise RuntimeError(
            f"det: input must be a square matrix, got shape {arr.shape}"
        )
    out = np.linalg.det(arr.astype(np.float64))
    return _from_numpy(np.ascontiguousarray(out).astype(to_numpy_dtype(a.dtype)), a.dtype, a.device)

def trace(a):
    """Sum of diagonal elements (2D only)."""
    arr_ndim = len(a.shape)
    if arr_ndim != 2:
        raise RuntimeError(
            f"trace: expected a matrix (2-D tensor), but got {arr_ndim}-D tensor"
        )
    # GPU composite: sum(diagonal(a))
    if _can_use_gpu(a):
        from .shape import diagonal
        from .reduce import sum_
        return sum_(diagonal(a))
    arr = _to_numpy(a)
    out = np.trace(arr)
    return _from_numpy(np.array(out, dtype=arr.dtype), a.dtype, a.device)

def dist(a, b, p=2):
    """p-norm distance between two tensors (flattened)."""
    # GPU composite: linalg_vector_norm(flatten(a) - flatten(b), p)
    if _can_use_gpu(a) and _can_use_gpu(b):
        from .shape import flatten
        return linalg_vector_norm(sub(flatten(a), flatten(b)), ord=p)
    a_np = _to_numpy(a)
    b_np = _to_numpy(b)
    diff = a_np.ravel() - b_np.ravel()
    out = np.linalg.norm(diff, ord=p)
    return _from_numpy(np.array(out, dtype=to_numpy_dtype(a.dtype)), a.dtype, a.device)

def cdist(x1, x2, p=2.0):
    """Batched pairwise distance between two sets of vectors."""
    # GPU composite via broadcasting
    if _can_use_gpu(x1) and _can_use_gpu(x2):
        from .reduce import sum_, amax
        ndim = len(x1.shape)
        if ndim == 2:
            # (M, D) → (M, 1, D) - (1, N, D) → (M, N, D)
            diff = sub(x1.unsqueeze(1), x2.unsqueeze(0))
        elif ndim == 3:
            # (B, M, D) → (B, M, 1, D) - (B, 1, N, D) → (B, M, N, D)
            diff = sub(x1.unsqueeze(2), x2.unsqueeze(1))
        else:
            diff = None
        if diff is not None:
            if p == 2.0:
                return gpu_sqrt(sum_(mul(diff, diff), dim=-1))
            if p == 1.0:
                return sum_(gpu_abs(diff), dim=-1)
            if p == float('inf'):
                return amax(gpu_abs(diff), dim=-1)
            # General p
            return gpu_pow(sum_(gpu_pow(gpu_abs(diff), p), dim=-1), 1.0 / p)
    a = _to_numpy(x1).astype(np.float64)
    b = _to_numpy(x2).astype(np.float64)
    if a.ndim == 2:
        a = a[np.newaxis]
        b = b[np.newaxis]
        squeeze = True
    else:
        squeeze = False
    batch = a.shape[0]
    results = []
    for i in range(batch):
        # Manual pairwise distance: (M, 1, D) - (1, N, D) -> (M, N, D)
        diff = a[i][:, np.newaxis, :] - b[i][np.newaxis, :, :]
        if p == 2.0:
            d = np.sqrt(np.sum(diff ** 2, axis=-1))
        elif p == 1.0:
            d = np.sum(np.abs(diff), axis=-1)
        elif p == float('inf'):
            d = np.max(np.abs(diff), axis=-1)
        elif p == 0.0:
            d = np.sum(diff != 0, axis=-1).astype(np.float64)
        else:
            d = np.sum(np.abs(diff) ** p, axis=-1) ** (1.0 / p)
        results.append(d)
    out = np.stack(results)
    if squeeze:
        out = out[0]
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(x1.dtype))), x1.dtype, x1.device)

