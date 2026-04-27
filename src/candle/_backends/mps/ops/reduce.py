import math
import ctypes
import struct
import numpy as np

from ._helpers import (  # pylint: disable=no-name-in-module
    _can_use_gpu, _empty_like, _unsupported_dtype,
    _metal_buf, _kernel_suffix, _scalar_fmt, _itemsize,
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
from .math import sqrt, mul, sub, exp, log, add, div, isnan, pow as _pow
from .math import abs as _abs
from .comparison import ne, logical_not
from .elementwise import where
from .shape import expand


def sum_(a, dim=None, keepdim=False, dtype=None):
    if dtype is not None:
        raise NotImplementedError("sum dtype not supported yet")
    if isinstance(dim, list):
        dim = tuple(dim)
    if isinstance(dim, tuple) and len(dim) == 0:
        dim = None

    ndim = len(a.shape)

    def _check_dim_range(d):
        if d < -ndim or d >= ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], but got {d})"
            )

    if isinstance(dim, int):
        _check_dim_range(dim)
    elif isinstance(dim, tuple):
        for d in dim:
            _check_dim_range(d)

    # GPU path: full-tensor reduction (dim=None)
    if dim is None and _can_use_gpu(a) and a.is_contiguous():
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        out_buf = _alloc_output_buf(1, a.dtype)
        d.dispatch_reduction(f"sum_partial_{sfx}", f"sum_final_{sfx}",
                             _metal_buf(a), out_buf, a.numel())
        out_shape = (1,) * ndim if keepdim else ()
        out_stride = (1,) * ndim if keepdim else ()
        return _from_metal_buffer(out_buf, out_shape, out_stride, a.dtype, a.device)

    # GPU path: axis reduction (dim specified)
    if dim is not None and _can_use_gpu(a) and a.is_contiguous():
        if isinstance(dim, int):
            dim_tuple = (dim,)
        else:
            dim_tuple = dim
        # For multi-dim, reduce sequentially
        result = a
        for d in sorted([x % ndim for x in dim_tuple], reverse=True):
            result = _gpu_reduce_single_dim(result, d, "sum", keepdim)
        return result

    # Non-contiguous: make contiguous and retry
    if _can_use_gpu(a):
        return sum_(a.contiguous(), dim=dim, keepdim=keepdim, dtype=dtype)
    _unsupported_dtype("sum", a)

def mean_(a, dim=None, keepdim=False):
    # GPU path: full-tensor mean (dim=None)
    if dim is None and _can_use_gpu(a) and a.is_contiguous():
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        sum_buf = _alloc_output_buf(1, a.dtype)
        d.dispatch_reduction(f"sum_partial_{sfx}", f"sum_final_{sfx}",
                             _metal_buf(a), sum_buf, a.numel())
        out_buf = _alloc_output_buf(1, a.dtype)
        n = float(a.numel())
        d.dispatch_binary_scalar(f"div_scalar_{sfx}", sum_buf, n, out_buf, 1)
        ndim = len(a.shape)
        out_shape = (1,) * ndim if keepdim else ()
        out_stride = (1,) * ndim if keepdim else ()
        return _from_metal_buffer(out_buf, out_shape, out_stride, a.dtype, a.device)

    # GPU path: axis reduction (dim specified)
    if dim is not None and _can_use_gpu(a) and a.is_contiguous():
        if isinstance(dim, int):
            dim_tuple = (dim,)
        else:
            dim_tuple = dim if isinstance(dim, tuple) else tuple(dim)
        ndim = len(a.shape)
        result = a
        for d in sorted([x % ndim for x in dim_tuple], reverse=True):
            result = _gpu_reduce_single_dim(result, d, "mean", keepdim)
        return result

    # Non-contiguous: make contiguous and retry
    if _can_use_gpu(a):
        return mean_(a.contiguous(), dim=dim, keepdim=keepdim)
    _unsupported_dtype("mean", a)

def std_(a, dim=None, keepdim=False, unbiased=True):
    if not a.dtype.is_floating_point and not a.dtype.is_complex:
        raise RuntimeError("std and var only support floating point and complex dtypes")
    # GPU composite: sqrt(var)
    if _can_use_gpu(a) and a.is_contiguous() and a.dtype.is_floating_point:
        v = var_(a, dim=dim, unbiased=unbiased, keepdim=keepdim)
        return sqrt(v)
    # Non-contiguous: make contiguous and retry
    if _can_use_gpu(a) and a.dtype.is_floating_point:
        return std_(a.contiguous(), dim=dim, keepdim=keepdim, unbiased=unbiased)
    _unsupported_dtype("std", a)

def var_(a, dim=None, unbiased=True, keepdim=False):
    # GPU composite using E[X^2] - E[X]^2 (avoids broadcast sub)
    if _can_use_gpu(a) and a.is_contiguous() and a.dtype.is_floating_point:
        sq = mul(a, a)
        mean_sq = mean_(sq, dim=dim, keepdim=keepdim)
        mean_val = mean_(a, dim=dim, keepdim=keepdim)
        mean_val_sq = mul(mean_val, mean_val)
        var_val = sub(mean_sq, mean_val_sq)
        if unbiased:
            if dim is None:
                n = a.numel()
            else:
                if isinstance(dim, int):
                    dims = (dim,)
                else:
                    dims = tuple(dim)
                n = 1
                ndim = len(a.shape)
                for d in dims:
                    n *= a.shape[d % ndim]
            if n > 1:
                correction = float(n) / float(n - 1)
                var_val = mul(var_val, correction)
        return var_val

    # Non-contiguous: make contiguous and retry
    if _can_use_gpu(a) and a.dtype.is_floating_point:
        return var_(a.contiguous(), dim=dim, unbiased=unbiased, keepdim=keepdim)
    _unsupported_dtype("var", a)

def var_mean(a, dim=None, unbiased=True, keepdim=False):
    return var_(a, dim=dim, unbiased=unbiased, keepdim=keepdim), mean_(a, dim=dim, keepdim=keepdim)

def norm_(a, p=2, dim=None, keepdim=False):
    """Compute the p-norm of a tensor (GPU composite)."""
    from ...._dtype import float32 as f32
    out_dtype = a.dtype if a.dtype.is_floating_point else f32
    if _can_use_gpu(a) and a.dtype.is_floating_point:
        a_c = a.contiguous() if not a.is_contiguous() else a
        if p == 2:
            sq = mul(a_c, a_c)
            s = sum_(sq, dim=dim, keepdim=keepdim)
            return sqrt(s)
        elif p == 1:
            ab = _abs(a_c)
            return sum_(ab, dim=dim, keepdim=keepdim)
        elif p == float('inf'):
            ab = _abs(a_c)
            return amax(ab, dim=dim, keepdim=keepdim)
        elif p == float('-inf'):
            ab = _abs(a_c)
            return amin(ab, dim=dim, keepdim=keepdim)
        else:
            ab = _abs(a_c)
            raised = _pow(ab, float(p))
            s = sum_(raised, dim=dim, keepdim=keepdim)
            return _pow(s, 1.0 / float(p))
    _unsupported_dtype("norm", a)

def prod_(a, dim=None, keepdim=False):
    ndim = len(a.shape)

    # GPU path: full-tensor reduction (dim=None)
    if dim is None and _can_use_gpu(a) and a.is_contiguous():
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        out_buf = _alloc_output_buf(1, a.dtype)
        d.dispatch_reduction(f"prod_partial_{sfx}", f"prod_final_{sfx}",
                             _metal_buf(a), out_buf, a.numel())
        out_shape = (1,) * ndim if keepdim else ()
        out_stride = (1,) * ndim if keepdim else ()
        return _from_metal_buffer(out_buf, out_shape, out_stride, a.dtype, a.device)

    # GPU path: axis reduction (dim specified)
    if dim is not None and _can_use_gpu(a) and a.is_contiguous():
        if isinstance(dim, int):
            return _gpu_reduce_single_dim(a, dim, "prod", keepdim)
        dim_tuple = dim if isinstance(dim, tuple) else tuple(dim)
        result = a
        for d in sorted([x % ndim for x in dim_tuple], reverse=True):
            result = _gpu_reduce_single_dim(result, d, "prod", keepdim)
        return result

    # Non-contiguous: make contiguous and retry
    if _can_use_gpu(a):
        return prod_(a.contiguous(), dim=dim, keepdim=keepdim)
    _unsupported_dtype("prod", a)

def all_(a, dim=None, keepdim=False):
    if _can_use_gpu(a) and a.is_contiguous():
        # GPU path: axis reduction (dim specified)
        if dim is not None:
            ndim = len(a.shape)
            if isinstance(dim, int):
                return _gpu_reduce_single_dim(a, dim, "all", keepdim)
            dim_tuple = dim if isinstance(dim, tuple) else tuple(dim)
            result = a
            for d in sorted([x % ndim for x in dim_tuple], reverse=True):
                result = _gpu_reduce_single_dim(result, d, "all", keepdim)
            return result
        # GPU path: full-tensor reduction (dim=None)
        # Convert to bool if not already
        if a.dtype != bool_dtype:
            a = ne(a, 0)
        d = _get_dispatcher()
        out_buf = _alloc_output_buf(1, bool_dtype)
        d.dispatch_reduction("all_partial_u8", "all_final_u8",
                             _metal_buf(a), out_buf, a.numel())
        ndim = len(a.shape)
        out_shape = (1,) * ndim if keepdim else ()
        out_stride = (1,) * ndim if keepdim else ()
        return _from_metal_buffer(out_buf, out_shape, out_stride, bool_dtype, a.device)
    # Non-contiguous: make contiguous and retry
    if _can_use_gpu(a):
        return all_(a.contiguous(), dim=dim, keepdim=keepdim)
    _unsupported_dtype("all", a)

def any_(a, dim=None, keepdim=False):
    if _can_use_gpu(a) and a.is_contiguous():
        # GPU path: axis reduction (dim specified)
        if dim is not None:
            ndim = len(a.shape)
            if isinstance(dim, int):
                return _gpu_reduce_single_dim(a, dim, "any", keepdim)
            dim_tuple = dim if isinstance(dim, tuple) else tuple(dim)
            result = a
            for d in sorted([x % ndim for x in dim_tuple], reverse=True):
                result = _gpu_reduce_single_dim(result, d, "any", keepdim)
            return result
        # GPU path: full-tensor reduction (dim=None)
        # Convert to bool if not already
        if a.dtype != bool_dtype:
            a = ne(a, 0)
        d = _get_dispatcher()
        out_buf = _alloc_output_buf(1, bool_dtype)
        d.dispatch_reduction("any_partial_u8", "any_final_u8",
                             _metal_buf(a), out_buf, a.numel())
        ndim = len(a.shape)
        out_shape = (1,) * ndim if keepdim else ()
        out_stride = (1,) * ndim if keepdim else ()
        return _from_metal_buffer(out_buf, out_shape, out_stride, bool_dtype, a.device)
    # Non-contiguous: make contiguous and retry
    if _can_use_gpu(a):
        return any_(a.contiguous(), dim=dim, keepdim=keepdim)
    _unsupported_dtype("any", a)

def argmax(a, dim=None, keepdim=False):
    # GPU path: full-tensor argmax (dim=None)
    if dim is None and _can_use_gpu(a) and a.is_contiguous():
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        out_buf = _alloc_output_buf(1, int64_dtype)  # uint output
        d.dispatch_arg_reduction(f"argmax_partial_{sfx}", f"argmax_final_{sfx}",
                                 _metal_buf(a), out_buf, a.numel())
        from ..runtime import buffer_contents
        ptr = buffer_contents(out_buf)
        idx_val = struct.unpack("I", (ctypes.c_char * 4).from_address(ptr))[0]
        # Write int64 value into a Metal buffer (no numpy)
        i64_buf = _alloc_output_buf(1, int64_dtype)
        i64_ptr = buffer_contents(i64_buf)
        struct.pack_into("q", (ctypes.c_char * 8).from_address(i64_ptr), 0, int(idx_val))
        return _from_metal_buffer(i64_buf, (), (), int64_dtype, a.device)

    # GPU path: axis argmax (dim specified)
    if dim is not None and _can_use_gpu(a) and a.is_contiguous():
        return _gpu_reduce_single_dim(a, dim, "argmax", keepdim)

    # Non-contiguous: make contiguous and retry
    if _can_use_gpu(a):
        return argmax(a.contiguous(), dim=dim, keepdim=keepdim)
    _unsupported_dtype("argmax", a)

def argmin(a, dim=None, keepdim=False):
    # GPU path: full-tensor argmin (dim=None)
    if dim is None and _can_use_gpu(a) and a.is_contiguous():
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        out_buf = _alloc_output_buf(1, int64_dtype)
        d.dispatch_arg_reduction(f"argmin_partial_{sfx}", f"argmin_final_{sfx}",
                                 _metal_buf(a), out_buf, a.numel())
        from ..runtime import buffer_contents
        ptr = buffer_contents(out_buf)
        idx_val = struct.unpack("I", (ctypes.c_char * 4).from_address(ptr))[0]
        # Write int64 value into a Metal buffer (no numpy)
        i64_buf = _alloc_output_buf(1, int64_dtype)
        i64_ptr = buffer_contents(i64_buf)
        struct.pack_into("q", (ctypes.c_char * 8).from_address(i64_ptr), 0, int(idx_val))
        return _from_metal_buffer(i64_buf, (), (), int64_dtype, a.device)

    # GPU path: axis argmin (dim specified)
    if dim is not None and _can_use_gpu(a) and a.is_contiguous():
        return _gpu_reduce_single_dim(a, dim, "argmin", keepdim)

    # Non-contiguous: make contiguous and retry
    if _can_use_gpu(a):
        return argmin(a.contiguous(), dim=dim, keepdim=keepdim)
    _unsupported_dtype("argmin", a)

def count_nonzero(a, dim=None, keepdim=False):
    if (_can_use_gpu(a) and a.is_contiguous()
            and a.dtype in (float32_dtype, float16_dtype, int32_dtype, int64_dtype)):
        # composite: ne(a, 0) → where(mask, 1.0, 0.0) → sum → cast to int64
        mask = ne(a, 0)
        if a.dtype in (float32_dtype, float16_dtype):
            ones_t = add(mul(a, 0.0), 1.0)
        else:
            # int types: create float32 ones via GPU fill
            numel = a.numel()
            buf = _alloc_output_buf(numel, float32_dtype)
            d = _get_dispatcher()
            sfx = _kernel_suffix(float32_dtype)
            d.dispatch_fill(f"fill_{sfx}", buf, 1.0, numel,
                            _itemsize(float32_dtype))
            from ...._C import _compute_strides
            ones_t = _from_metal_buffer(buf, tuple(a.shape),
                                        _compute_strides(tuple(a.shape)),
                                        float32_dtype, a.device)
        count_f = where(mask, ones_t, 0.0)
        s = sum_(count_f, dim=dim, keepdim=keepdim)
        # NOTE: int64 cast via numpy — result is already reduced, negligible overhead
        s_np = _to_numpy(s).astype(np.int64)
        return _from_numpy(np.ascontiguousarray(s_np), int64_dtype, a.device)
    # Non-contiguous: make contiguous and retry
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype, int32_dtype, int64_dtype):
        return count_nonzero(a.contiguous(), dim=dim, keepdim=keepdim)
    _unsupported_dtype("count_nonzero", a)

def cumsum(a, dim=0):
    # GPU path: float32/float16, contiguous
    if (_can_use_gpu(a) and a.is_contiguous()
            and a.dtype in (float32_dtype, float16_dtype)):
        ndim = len(a.shape)
        if dim < 0:
            dim = ndim + dim
        outer_size = 1
        for i in range(dim):
            outer_size *= a.shape[i]
        dim_size = a.shape[dim]
        inner_size = 1
        for i in range(dim + 1, ndim):
            inner_size *= a.shape[i]
        sfx = _kernel_suffix(a.dtype)
        d = _get_dispatcher()
        numel = outer_size * dim_size * inner_size
        out_buf = _alloc_output_buf(numel, a.dtype)
        d.dispatch_cumsum(f"cumsum_{sfx}", _metal_buf(a), out_buf,
                          outer_size, dim_size, inner_size)
        return _from_metal_buffer(out_buf, tuple(a.shape),
                                  tuple(a.stride()), a.dtype, a.device)
    # Non-contiguous: make contiguous and retry
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        return cumsum(a.contiguous(), dim=dim)
    _unsupported_dtype("cumsum", a)

def cumprod(a, dim=0):
    # GPU path: float32/float16, contiguous (same dispatch as cumsum)
    if (_can_use_gpu(a) and a.is_contiguous()
            and a.dtype in (float32_dtype, float16_dtype)):
        ndim = len(a.shape)
        if dim < 0:
            dim = ndim + dim
        outer_size = 1
        for i in range(dim):
            outer_size *= a.shape[i]
        dim_size = a.shape[dim]
        inner_size = 1
        for i in range(dim + 1, ndim):
            inner_size *= a.shape[i]
        sfx = _kernel_suffix(a.dtype)
        d = _get_dispatcher()
        numel = outer_size * dim_size * inner_size
        out_buf = _alloc_output_buf(numel, a.dtype)
        d.dispatch_cumsum(f"cumprod_{sfx}", _metal_buf(a), out_buf,
                          outer_size, dim_size, inner_size)
        return _from_metal_buffer(out_buf, tuple(a.shape),
                                  tuple(a.stride()), a.dtype, a.device)
    # Non-contiguous: make contiguous and retry
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        return cumprod(a.contiguous(), dim=dim)
    _unsupported_dtype("cumprod", a)

def _cumextreme_gpu(a, dim, mode):
    """GPU cummax/cummin helper. mode is 'cummax' or 'cummin'."""
    ndim = len(a.shape)
    if dim < 0:
        dim = ndim + dim
    outer_size = 1
    for i in range(dim):
        outer_size *= a.shape[i]
    dim_size = a.shape[dim]
    inner_size = 1
    for i in range(dim + 1, ndim):
        inner_size *= a.shape[i]
    sfx = _kernel_suffix(a.dtype)
    d = _get_dispatcher()
    numel = outer_size * dim_size * inner_size
    values_buf = _alloc_output_buf(numel, a.dtype)
    indices_buf = _alloc_output_buf(numel, int32_dtype)
    dispatch = d.dispatch_cummax if mode == "cummax" else d.dispatch_cummin
    dispatch(f"{mode}_{sfx}", _metal_buf(a), values_buf, indices_buf,
             outer_size, dim_size, inner_size)
    from ...._C import _compute_strides
    out_shape = tuple(a.shape)
    out_stride = _compute_strides(out_shape)
    values = _from_metal_buffer(values_buf, out_shape, out_stride, a.dtype, a.device)
    # Convert int32 indices to int64
    idx_np = np.frombuffer(
        _metal_buf_to_bytes(indices_buf, numel * 4),
        dtype=np.int32).astype(np.int64).reshape(a.shape)
    indices = _from_numpy(idx_np, int64_dtype, a.device)
    return values, indices


def cummax(a, dim=0):
    if (_can_use_gpu(a) and a.is_contiguous()
            and a.dtype in (float32_dtype, float16_dtype)):
        return _cumextreme_gpu(a, dim, "cummax")
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        return cummax(a.contiguous(), dim=dim)
    _unsupported_dtype("cummax", a)

def cummin(a, dim):
    """Cumulative minimum along a dimension, returns (values, indices) namedtuple."""
    if (_can_use_gpu(a) and a.is_contiguous()
            and a.dtype in (float32_dtype, float16_dtype)):
        from collections import namedtuple
        CumminResult = namedtuple("cummin", ["values", "indices"])
        vals, idxs = _cumextreme_gpu(a, dim, "cummin")
        return CumminResult(vals, idxs)
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        return cummin(a.contiguous(), dim=dim)
    _unsupported_dtype("cummin", a)


# ---------------------------------------------------------------------------
# Top-level gap-fill ops (Category C2)
# ---------------------------------------------------------------------------

def _sort_gpu(a, dim, descending):
    """GPU sort helper returning (values_buf, indices_buf, shape, strides)."""
    ndim = len(a.shape)
    if dim < 0:
        dim = ndim + dim
    outer_size = 1
    for i in range(dim):
        outer_size *= a.shape[i]
    dim_size = a.shape[dim]
    inner_size = 1
    for i in range(dim + 1, ndim):
        inner_size *= a.shape[i]
    sfx = _kernel_suffix(a.dtype)
    d = _get_dispatcher()
    numel = outer_size * dim_size * inner_size
    values_buf = _alloc_output_buf(numel, a.dtype)
    indices_buf = _alloc_output_buf(numel, int32_dtype)
    d.dispatch_sort(f"sort_{sfx}", _metal_buf(a), values_buf, indices_buf,
                    outer_size, dim_size, inner_size, descending)
    return values_buf, indices_buf, numel

def argsort(a, dim=-1, descending=False, stable=False):
    if (_can_use_gpu(a) and a.is_contiguous()
            and a.dtype in (float32_dtype, float16_dtype)):
        _, indices_buf, numel = _sort_gpu(a, dim, descending)
        # Convert int32 indices to int64
        idx_np = np.frombuffer(
            _metal_buf_to_bytes(indices_buf, numel * 4),
            dtype=np.int32).astype(np.int64).reshape(a.shape)
        return _from_numpy(idx_np, int64_dtype, a.device)
    # Non-contiguous: make contiguous and retry
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        return argsort(a.contiguous(), dim=dim, descending=descending, stable=stable)
    _unsupported_dtype("argsort", a)

def sort(a, dim=-1, descending=False, stable=False):
    if (_can_use_gpu(a) and a.is_contiguous()
            and a.dtype in (float32_dtype, float16_dtype)):
        values_buf, indices_buf, numel = _sort_gpu(a, dim, descending)
        from ...._C import _compute_strides
        out_shape = tuple(a.shape)
        out_stride = _compute_strides(out_shape)
        values = _from_metal_buffer(values_buf, out_shape, out_stride,
                                    a.dtype, a.device)
        # Convert int32 indices to int64
        idx_np = np.frombuffer(
            _metal_buf_to_bytes(indices_buf, numel * 4),
            dtype=np.int32).astype(np.int64).reshape(a.shape)
        indices = _from_numpy(idx_np, int64_dtype, a.device)
        return (values, indices)
    # Non-contiguous: make contiguous and retry
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        return sort(a.contiguous(), dim=dim, descending=descending, stable=stable)
    _unsupported_dtype("sort", a)

def topk(a, k, dim=-1, largest=True, sorted=True):
    if (_can_use_gpu(a) and a.is_contiguous()
            and a.dtype in (float32_dtype, float16_dtype)):
        values, indices = sort(a, dim=dim, descending=largest)
        # Slice first k along dim
        ndim = len(a.shape)
        if dim < 0:
            dim = ndim + dim
        slices = [slice(None)] * ndim
        slices[dim] = slice(0, k)
        return (values[tuple(slices)], indices[tuple(slices)])
    # Non-contiguous: make contiguous and retry
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        return topk(a.contiguous(), k=k, dim=dim, largest=largest, sorted=sorted)
    _unsupported_dtype("topk", a)

def min_(a, b):
    if _can_use_gpu(a) and isinstance(b, Tensor) and _can_use_gpu(b):
        return _dispatch_binary_gpu(a, b, "minimum")
    if _can_use_gpu(a) and not isinstance(b, Tensor):
        return _dispatch_binary_gpu(a, float(b), "minimum")
    _unsupported_dtype("min", a)

def max_(a, b):
    if _can_use_gpu(a) and isinstance(b, Tensor) and _can_use_gpu(b):
        return _dispatch_binary_gpu(a, b, "maximum")
    if _can_use_gpu(a) and not isinstance(b, Tensor):
        return _dispatch_binary_gpu(a, float(b), "maximum")
    _unsupported_dtype("max", a)

def amin(a, dim=None, keepdim=False):
    if dim is not None and _can_use_gpu(a) and a.is_contiguous():
        if isinstance(dim, int):
            return _gpu_reduce_single_dim(a, dim, "min", keepdim)
        ndim = len(a.shape)
        result = a
        for d in sorted([x % ndim for x in dim], reverse=True):
            result = _gpu_reduce_single_dim(result, d, "min", keepdim)
        return result
    # dim=None full reduction: use GPU reduce
    if dim is None and _can_use_gpu(a) and a.is_contiguous():
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        out_buf = _alloc_output_buf(1, a.dtype)
        d.dispatch_reduction(f"min_partial_{sfx}", f"min_final_{sfx}",
                             _metal_buf(a), out_buf, a.numel())
        ndim = len(a.shape)
        out_shape = (1,) * ndim if keepdim else ()
        out_stride = (1,) * ndim if keepdim else ()
        return _from_metal_buffer(out_buf, out_shape, out_stride, a.dtype, a.device)
    # Non-contiguous: make contiguous and retry
    if _can_use_gpu(a):
        return amin(a.contiguous(), dim=dim, keepdim=keepdim)
    _unsupported_dtype("amin", a)

def amax(a, dim=None, keepdim=False):
    if dim is not None and _can_use_gpu(a) and a.is_contiguous():
        if isinstance(dim, int):
            return _gpu_reduce_single_dim(a, dim, "max", keepdim)
        ndim = len(a.shape)
        result = a
        for d in sorted([x % ndim for x in dim], reverse=True):
            result = _gpu_reduce_single_dim(result, d, "max", keepdim)
        return result
    # dim=None full reduction: use GPU reduce
    if dim is None and _can_use_gpu(a) and a.is_contiguous():
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        out_buf = _alloc_output_buf(1, a.dtype)
        d.dispatch_reduction(f"max_partial_{sfx}", f"max_final_{sfx}",
                             _metal_buf(a), out_buf, a.numel())
        ndim = len(a.shape)
        out_shape = (1,) * ndim if keepdim else ()
        out_stride = (1,) * ndim if keepdim else ()
        return _from_metal_buffer(out_buf, out_shape, out_stride, a.dtype, a.device)
    # Non-contiguous: make contiguous and retry
    if _can_use_gpu(a):
        return amax(a.contiguous(), dim=dim, keepdim=keepdim)
    _unsupported_dtype("amax", a)

def fmin(a, b):
    # fmin ignores NaN (returns the non-NaN value) — use GPU minimum composite
    if _can_use_gpu(a) and isinstance(b, Tensor) and _can_use_gpu(b):
        return _dispatch_binary_gpu(a, b, "minimum")
    if _can_use_gpu(a) and not isinstance(b, Tensor):
        return _dispatch_binary_gpu(a, float(b), "minimum")
    _unsupported_dtype("fmin", a)

def fmax(a, b):
    # fmax ignores NaN (returns the non-NaN value) — use GPU maximum composite
    if _can_use_gpu(a) and isinstance(b, Tensor) and _can_use_gpu(b):
        return _dispatch_binary_gpu(a, b, "maximum")
    if _can_use_gpu(a) and not isinstance(b, Tensor):
        return _dispatch_binary_gpu(a, float(b), "maximum")
    _unsupported_dtype("fmax", a)

def maximum(a, b):
    if _can_use_gpu(a):
        return _dispatch_binary_gpu(a, b, "maximum")
    _unsupported_dtype("maximum", a)

def minimum(a, b):
    if _can_use_gpu(a):
        return _dispatch_binary_gpu(a, b, "minimum")
    _unsupported_dtype("minimum", a)

def logsumexp(a, dim, keepdim=False):
    """Numerically stable logsumexp: log(sum(exp(x), dim))."""
    # GPU composite: amax → sub → exp → sum → log → add
    if _can_use_gpu(a) and a.is_contiguous() and a.dtype.is_floating_point:
        max_val = amax(a, dim=dim, keepdim=True)
        # expand max_val to match a's shape for broadcast subtraction
        max_expanded = expand(max_val, tuple(a.shape))
        shifted = sub(a, max_expanded)
        exp_shifted = exp(shifted)
        sum_exp = sum_(exp_shifted, dim=dim, keepdim=keepdim)
        log_sum = log(sum_exp)
        if keepdim:
            result = add(log_sum, max_val)
        else:
            ndim = len(a.shape)
            d = dim % ndim if isinstance(dim, int) else dim
            from ...common.view import squeeze as _squeeze
            max_squeezed = _squeeze(max_val, d)
            result = add(log_sum, max_squeezed)
        return result

    # Non-contiguous: make contiguous and retry
    if _can_use_gpu(a) and a.dtype.is_floating_point:
        return logsumexp(a.contiguous(), dim=dim, keepdim=keepdim)
    _unsupported_dtype("logsumexp", a)

def renorm(a, p, dim, maxnorm):
    """Renormalize tensor: each sub-tensor along dim has norm <= maxnorm."""
    # GPU composite: compute norm along all axes except dim, then scale
    if _can_use_gpu(a) and a.dtype.is_floating_point:
        a_c = a.contiguous() if not a.is_contiguous() else a
        n = norm_(a_c, p=p, dim=dim, keepdim=True)
        # scale = where(norm > maxnorm, maxnorm / (norm + eps), 1.0)
        eps_t = _dispatch_binary_gpu(n, 1e-7, "add")
        from .math import reciprocal
        ratio = mul(reciprocal(eps_t), float(maxnorm))
        from .comparison import gt
        cond = gt(n, float(maxnorm))
        # Use where with scalar y=1.0 (no numpy ones tensor needed)
        scale = where(cond, ratio, 1.0)
        return mul(a_c, scale)
    _unsupported_dtype("renorm", a)

def nansum(a, dim=None, keepdim=False):
    """Sum ignoring NaN values."""
    if (_can_use_gpu(a) and a.is_contiguous()
            and a.dtype in (float32_dtype, float16_dtype)):
        # composite: where(isnan(a), 0, a) → sum
        mask = isnan(a)
        not_nan = logical_not(mask)
        cleaned = where(not_nan, a, 0.0)
        return sum_(cleaned, dim=dim, keepdim=keepdim)
    # Non-contiguous: make contiguous and retry
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        return nansum(a.contiguous(), dim=dim, keepdim=keepdim)
    _unsupported_dtype("nansum", a)

def nanmean(a, dim=None, keepdim=False):
    """Mean ignoring NaN values."""
    if (_can_use_gpu(a) and a.is_contiguous()
            and a.dtype in (float32_dtype, float16_dtype)):
        mask = isnan(a)
        not_nan = logical_not(mask)
        cleaned = where(not_nan, a, 0.0)
        s = sum_(cleaned, dim=dim, keepdim=keepdim)
        # Count non-NaN via GPU: cast not_nan bool to float, then sum
        not_nan_f = where(not_nan, 1.0, 0.0)
        cnt = sum_(not_nan_f, dim=dim, keepdim=keepdim)
        return div(s, cnt)
    # Non-contiguous: make contiguous and retry
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        return nanmean(a.contiguous(), dim=dim, keepdim=keepdim)
    _unsupported_dtype("nanmean", a)

def aminmax(a, dim=None, keepdim=False):
    """Returns the min and max of a tensor."""
    from collections import namedtuple
    # GPU composite: just call amin and amax
    mn_t = amin(a, dim=dim, keepdim=keepdim)
    mx_t = amax(a, dim=dim, keepdim=keepdim)
    AminmaxResult = namedtuple("aminmax", ["min", "max"])
    return AminmaxResult(mn_t, mx_t)

def quantile(a, q, dim=None, keepdim=False):
    """Compute the q-th quantile of the input tensor."""
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        from ...common.view import reshape, squeeze
        # Extract scalar q value
        if isinstance(q, Tensor):
            q_val = _scalar_value(q)
        else:
            q_val = float(q)
        if dim is None:
            flat = reshape(a.contiguous(), (-1,))
            sorted_vals, _ = sort(flat, dim=0)
            n = flat.shape[0]
            idx_f = q_val * (n - 1)
            lo = int(idx_f)
            hi = min(lo + 1, n - 1)
            frac = idx_f - lo
            v_lo = sorted_vals[(slice(lo, lo + 1),)]
            v_hi = sorted_vals[(slice(hi, hi + 1),)]
            result = add(v_lo, mul(sub(v_hi, v_lo), frac))
            return squeeze(result, 0)
        else:
            ndim = len(a.shape)
            if dim < 0:
                dim = dim + ndim
            sorted_vals, _ = sort(a.contiguous(), dim=dim)
            n = a.shape[dim]
            idx_f = q_val * (n - 1)
            lo = int(idx_f)
            hi = min(lo + 1, n - 1)
            frac = idx_f - lo
            slices_lo = [slice(None)] * ndim
            slices_lo[dim] = slice(lo, lo + 1)
            slices_hi = [slice(None)] * ndim
            slices_hi[dim] = slice(hi, hi + 1)
            v_lo = sorted_vals[tuple(slices_lo)]
            v_hi = sorted_vals[tuple(slices_hi)]
            result = add(v_lo, mul(sub(v_hi, v_lo), frac))
            if not keepdim:
                result = squeeze(result, dim)
            return result
    _unsupported_dtype("quantile", a)

def nanquantile(a, q, dim=None, keepdim=False):
    """Compute the q-th quantile ignoring NaN values."""
    # GPU composite for dim specified: NaN→+inf, sort, count, interpolate
    if (dim is not None and _can_use_gpu(a)
            and a.dtype in (float32_dtype, float16_dtype)):
        from ...common.view import squeeze as _squeeze
        a_c = a.contiguous() if not a.is_contiguous() else a
        ndim = len(a.shape)
        if dim < 0:
            dim = dim + ndim
        # Extract scalar q value
        if isinstance(q, Tensor):
            q_val = float(_scalar_value(q))
        else:
            q_val = float(q)
        # Replace NaN with +inf so they sort to the end
        nan_mask = isnan(a_c)
        not_nan_mask = logical_not(nan_mask)
        cleaned = where(not_nan_mask, a_c, float('inf'))
        sorted_vals, _ = sort(cleaned, dim=dim)
        # Count non-NaN per slice: create ones tensor via GPU, then mask+sum
        ones = add(mul(a_c, 0.0), 1.0)
        not_nan_f = where(not_nan_mask, ones, 0.0)
        count = sum_(not_nan_f, dim=dim, keepdim=True)
        # Quantile index = q * (count - 1), linear interpolation
        count_np = _to_numpy(count)
        idx_f_np = q_val * (count_np - 1.0)
        lo_np = np.floor(idx_f_np).astype(np.intp)
        hi_np = np.minimum(lo_np + 1, (count_np - 1).astype(np.intp))
        frac_np = idx_f_np - lo_np.astype(np.float64)
        sorted_np = _to_numpy(sorted_vals)
        v_lo = np.take_along_axis(sorted_np, lo_np, axis=dim)
        v_hi = np.take_along_axis(sorted_np, hi_np, axis=dim)
        result_np = v_lo + frac_np * (v_hi - v_lo)
        if not keepdim:
            result_np = np.squeeze(result_np, axis=dim)
        return _from_numpy(np.ascontiguousarray(result_np.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)
    # Fallback for dim=None or non-GPU
    arr = _to_numpy(a).astype(np.float64)
    q_val = _to_numpy(q) if hasattr(q, '_numpy_view') else np.asarray(q, dtype=np.float64)
    if dim is None:
        out = np.nanquantile(arr, q_val)
    else:
        out = np.nanquantile(arr, q_val, axis=dim, keepdims=keepdim)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)

def nanmedian(a, dim=None, keepdim=False):
    """Median ignoring NaN values. Returns (values, indices) when dim is given."""
    # GPU composite for dim specified: NaN→+inf, sort, count non-NaN, pick middle
    if (dim is not None and _can_use_gpu(a)
            and a.dtype in (float32_dtype, float16_dtype)):
        from ...common.view import squeeze as _squeeze
        a_c = a.contiguous() if not a.is_contiguous() else a
        ndim = len(a.shape)
        if dim < 0:
            dim = dim + ndim
        # Replace NaN with +inf so they sort to the end
        nan_mask = isnan(a_c)
        not_nan_mask = logical_not(nan_mask)
        cleaned = where(not_nan_mask, a_c, float('inf'))
        sorted_vals, sorted_idx = sort(cleaned, dim=dim)
        # Count non-NaN per slice: create ones tensor via GPU, then mask+sum
        ones = add(mul(a_c, 0.0), 1.0)
        not_nan_f = where(not_nan_mask, ones, 0.0)
        count = sum_(not_nan_f, dim=dim, keepdim=True)
        # Median index = (count - 1) // 2; use floor(mul(sub(count, 1), 0.5))
        from .math import floor
        med_idx_f = floor(mul(sub(count, 1.0), 0.5))
        # Convert to int for gather — small scalar, use numpy for index extraction
        med_idx_np = _to_numpy(med_idx_f).astype(np.intp)
        sorted_vals_np = _to_numpy(sorted_vals)
        sorted_idx_np = _to_numpy(sorted_idx)
        values_np = np.take_along_axis(sorted_vals_np, med_idx_np, axis=dim)
        indices_np = np.take_along_axis(sorted_idx_np, med_idx_np, axis=dim)
        if not keepdim:
            values_np = np.squeeze(values_np, axis=dim)
            indices_np = np.squeeze(indices_np, axis=dim)
        from collections import namedtuple
        NanmedianResult = namedtuple("nanmedian", ["values", "indices"])
        return NanmedianResult(
            _from_numpy(np.ascontiguousarray(values_np.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device),
            _from_numpy(np.ascontiguousarray(indices_np.astype(np.int64)), int64_dtype, a.device),
        )
    # Fallback for dim=None or non-GPU
    arr = _to_numpy(a).astype(np.float64)
    if dim is None:
        out = np.nanmedian(arr)
        return _from_numpy(np.array(out, dtype=to_numpy_dtype(a.dtype)), a.dtype, a.device)
    values = np.nanmedian(arr, axis=dim, keepdims=keepdim)
    n = arr.shape[dim]
    sorted_arr = np.sort(arr, axis=dim)
    not_nan = ~np.isnan(arr)
    count = np.sum(not_nan, axis=dim, keepdims=True)
    med_idx_sorted = (count - 1) // 2
    sorted_indices = np.argsort(arr, axis=dim)
    indices = np.take_along_axis(sorted_indices, med_idx_sorted.astype(np.intp), axis=dim)
    if not keepdim:
        indices = np.squeeze(indices, axis=dim)
    from collections import namedtuple
    NanmedianResult = namedtuple("nanmedian", ["values", "indices"])
    return NanmedianResult(
        _from_numpy(np.ascontiguousarray(values.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device),
        _from_numpy(np.ascontiguousarray(indices.astype(np.int64)), int64_dtype, a.device),
    )

def median(a, dim=None, keepdim=False):
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        from ...common.view import reshape, squeeze
        if dim is None:
            flat = reshape(a.contiguous(), (-1,))
            sorted_vals, sorted_idx = sort(flat, dim=0)
            n = flat.shape[0]
            mid = n // 2
            val = sorted_vals[(slice(mid, mid + 1),)]
            return squeeze(val, 0)
        ndim = len(a.shape)
        if dim < 0:
            dim = dim + ndim
        sorted_vals, sorted_idx = sort(a.contiguous(), dim=dim)
        n = a.shape[dim]
        mid = n // 2
        slices = [slice(None)] * ndim
        slices[dim] = slice(mid, mid + 1)
        values = sorted_vals[tuple(slices)]
        indices = sorted_idx[tuple(slices)]
        if not keepdim:
            values = squeeze(values, dim)
            indices = squeeze(indices, dim)
        return (values, indices)
    _unsupported_dtype("median", a)


# ---------------------------------------------------------------------------
# Group 7: New math ops for Tensor API alignment
# ---------------------------------------------------------------------------

def kthvalue(a, k, dim=-1, keepdim=False):
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        from ...common.view import squeeze
        ndim = len(a.shape)
        if dim < 0:
            dim = dim + ndim
        sorted_vals, sorted_idx = sort(a.contiguous(), dim=dim)
        slices = [slice(None)] * ndim
        slices[dim] = slice(k - 1, k)
        values = sorted_vals[tuple(slices)]
        indices = sorted_idx[tuple(slices)]
        if not keepdim:
            values = squeeze(values, dim)
            indices = squeeze(indices, dim)
        return (values, indices)
    _unsupported_dtype("kthvalue", a)

def unique(a, sorted=True, return_inverse=False, return_counts=False, dim=None):
    arr = _to_numpy(a)
    if dim is None:
        flat = arr.flatten()
        result = np.unique(flat, return_inverse=return_inverse, return_counts=return_counts)
    else:
        result = np.unique(arr, return_inverse=return_inverse, return_counts=return_counts, axis=dim)
    if isinstance(result, tuple):
        out = []
        for i, r in enumerate(result):
            r_cont = np.ascontiguousarray(r)
            if i == 0:
                out.append(_from_numpy(r_cont, a.dtype, a.device))
            else:
                out.append(_from_numpy(r_cont.astype(np.int64), int64_dtype, a.device))
        return tuple(out)
    return _from_numpy(np.ascontiguousarray(result), a.dtype, a.device)

def unique_consecutive(a, return_inverse=False, return_counts=False, dim=None):
    arr = _to_numpy(a)
    if dim is None:
        flat = arr.flatten()
        if flat.size == 0:
            mask = np.array([], dtype=bool)
        else:
            mask = np.concatenate([[True], flat[1:] != flat[:-1]])
        unique_vals = flat[mask]
        result = [_from_numpy(np.ascontiguousarray(unique_vals), a.dtype, a.device)]
        if return_inverse:
            inverse = np.cumsum(mask) - 1
            result.append(_from_numpy(inverse.astype(np.int64), int64_dtype, a.device))
        if return_counts:
            indices = np.concatenate([np.where(mask)[0], [len(flat)]])
            counts = np.diff(indices)
            result.append(_from_numpy(counts.astype(np.int64), int64_dtype, a.device))
        return tuple(result) if len(result) > 1 else result[0]
    # dim-based
    ndim = arr.ndim
    d = dim if dim >= 0 else dim + ndim
    n = arr.shape[d]
    if n == 0:
        result = [_from_numpy(np.ascontiguousarray(arr), a.dtype, a.device)]
        if return_inverse:
            result.append(_from_numpy(np.array([], dtype=np.int64), int64_dtype, a.device))
        if return_counts:
            result.append(_from_numpy(np.array([], dtype=np.int64), int64_dtype, a.device))
        return tuple(result) if len(result) > 1 else result[0]
    mask = [True]
    for i in range(1, n):
        mask.append(not np.array_equal(
            np.take(arr, [i], axis=d), np.take(arr, [i - 1], axis=d)))
    mask = np.array(mask)
    keep = np.where(mask)[0]
    unique_arr = np.concatenate([np.take(arr, [k], axis=d) for k in keep], axis=d)
    result = [_from_numpy(np.ascontiguousarray(unique_arr), a.dtype, a.device)]
    if return_inverse:
        inverse = np.cumsum(mask) - 1
        result.append(_from_numpy(inverse.astype(np.int64), int64_dtype, a.device))
    if return_counts:
        indices = np.concatenate([keep, [n]])
        counts = np.diff(indices)
        result.append(_from_numpy(counts.astype(np.int64), int64_dtype, a.device))
    return tuple(result) if len(result) > 1 else result[0]

def searchsorted(sorted_seq, values, out_int32=False, right=False, side=None, sorter=None):
    seq_np = _to_numpy(sorted_seq)
    val_np = _to_numpy(values) if isinstance(values, Tensor) else np.array(values)
    side_str = side if side is not None else ('right' if right else 'left')
    if sorter is not None:
        sorter_np = _to_numpy(sorter).astype(np.int64)
        out = np.searchsorted(seq_np.flatten(), val_np.flatten(), side=side_str, sorter=sorter_np)
    else:
        if seq_np.ndim == 1:
            out = np.searchsorted(seq_np, val_np, side=side_str)
        else:
            out = np.zeros_like(val_np, dtype=np.int64)
            for i in range(seq_np.shape[0]):
                out[i] = np.searchsorted(seq_np[i], val_np[i], side=side_str)
    out_dtype_np = np.int32 if out_int32 else np.int64
    return _from_numpy(out.astype(out_dtype_np), int64_dtype, sorted_seq.device)

def argwhere(a):
    """Returns indices of non-zero elements as a 2D tensor (shape [N, ndim])."""
    arr = _to_numpy(a)
    out = np.argwhere(arr)
    return _from_numpy(out.astype(np.int64), int64_dtype, a.device)

try:
    from candle._C._mps_ops import (  # pylint: disable=import-error,no-name-in-module
        sum_, mean_, std_, var_, var_mean, norm_, prod_,
        all_, any_, argmax, argmin, count_nonzero,
        cumsum, cumprod, cummax, cummin,
        argsort, sort, topk,
        min_, max_, amin, amax, fmin, fmax, maximum, minimum,
        logsumexp, renorm, nansum, nanmean,
        aminmax, quantile, nanquantile, nanmedian, median, kthvalue,
        unique, unique_consecutive, searchsorted, argwhere,
    )
except ImportError:
    pass
