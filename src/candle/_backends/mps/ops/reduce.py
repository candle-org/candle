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
from .math import sqrt, mul, sub, exp, log, add, div, isnan
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

    return _from_numpy(_to_numpy(a).sum(axis=dim, keepdims=keepdim), a.dtype, a.device)

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

    return _from_numpy(_to_numpy(a).mean(axis=dim, keepdims=keepdim), a.dtype, a.device)

def std_(a, dim=None, keepdim=False, unbiased=True):
    if not a.dtype.is_floating_point and not a.dtype.is_complex:
        raise RuntimeError("std and var only support floating point and complex dtypes")
    # GPU composite: sqrt(var)
    if _can_use_gpu(a) and a.is_contiguous() and a.dtype.is_floating_point:
        v = var_(a, dim=dim, unbiased=unbiased, keepdim=keepdim)
        return sqrt(v)
    ddof = 1 if unbiased else 0
    out = np.std(_to_numpy(a), axis=dim, keepdims=keepdim, ddof=ddof)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)

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

    arr = _to_numpy(a)
    ddof = 1 if unbiased else 0
    if dim is not None:
        if isinstance(dim, (list, tuple)):
            dim = tuple(dim)
        out = np.var(arr, axis=dim, keepdims=keepdim, ddof=ddof)
    else:
        out = np.var(arr, ddof=ddof)
        if keepdim:
            out = np.full([1] * arr.ndim, out)
    return _from_numpy(np.ascontiguousarray(np.atleast_1d(out).astype(arr.dtype, copy=False)), a.dtype, a.device)

def norm_(a, p=2, dim=None, keepdim=False):
    arr = _to_numpy(a).astype(np.float64)
    if dim is not None:
        if isinstance(dim, (list, tuple)):
            dim = tuple(dim)
        out = np.linalg.norm(arr, ord=p, axis=dim, keepdims=keepdim)
    else:
        out = np.linalg.norm(arr.ravel(), ord=p)
        if keepdim:
            out = np.full([1] * arr.ndim, out)
    from ...._dtype import float32 as f32
    out_dtype = a.dtype if a.dtype.is_floating_point else f32
    return _from_numpy(np.ascontiguousarray(np.atleast_1d(out).astype(to_numpy_dtype(out_dtype), copy=False)), out_dtype, a.device)

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

    arr = _to_numpy(a)
    if dim is not None:
        out = np.prod(arr, axis=dim, keepdims=keepdim)
    else:
        out = np.prod(arr)
        if keepdim:
            out = np.full([1] * arr.ndim, out)
    return _from_numpy(np.ascontiguousarray(np.atleast_1d(out).astype(arr.dtype, copy=False)), a.dtype, a.device)

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
    return _from_numpy(np.all(_to_numpy(a), axis=dim, keepdims=keepdim), bool_dtype, a.device)

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
    return _from_numpy(np.any(_to_numpy(a), axis=dim, keepdims=keepdim), bool_dtype, a.device)

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
        out = np.array(int(idx_val), dtype=np.int64)
        return _from_numpy(out, int64_dtype, a.device)

    # GPU path: axis argmax (dim specified)
    if dim is not None and _can_use_gpu(a) and a.is_contiguous():
        return _gpu_reduce_single_dim(a, dim, "argmax", keepdim)

    arr = _to_numpy(a)
    if dim is None:
        out = np.array(np.argmax(arr), dtype=np.int64)
    else:
        out = np.argmax(arr, axis=dim)
        if keepdim:
            out = np.expand_dims(out, axis=dim)
        out = out.astype(np.int64)
    return _from_numpy(out, int64_dtype, a.device)

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
        out = np.array(int(idx_val), dtype=np.int64)
        return _from_numpy(out, int64_dtype, a.device)

    # GPU path: axis argmin (dim specified)
    if dim is not None and _can_use_gpu(a) and a.is_contiguous():
        return _gpu_reduce_single_dim(a, dim, "argmin", keepdim)

    arr = _to_numpy(a)
    if dim is None:
        out = np.array(np.argmin(arr), dtype=np.int64)
    else:
        out = np.argmin(arr, axis=dim)
        if keepdim:
            out = np.expand_dims(out, axis=dim)
        out = out.astype(np.int64)
    return _from_numpy(out, int64_dtype, a.device)

def count_nonzero(a, dim=None, keepdim=False):
    if (_can_use_gpu(a) and a.is_contiguous()
            and a.dtype in (float32_dtype, float16_dtype, int32_dtype, int64_dtype)):
        # composite: ne(a, 0) → sum as float → cast to int64
        mask = ne(a, 0)
        # Need float for sum — use where(mask, 1.0, 0.0) with a float tensor
        ones_np = np.ones(a.shape, dtype=np.float32)
        ones_t = _from_numpy(ones_np, float32_dtype, a.device)
        count_f = where(mask, ones_t, 0.0)
        s = sum_(count_f, dim=dim, keepdim=keepdim)
        # Convert to int64 via numpy (small result)
        s_np = _to_numpy(s).astype(np.int64)
        return _from_numpy(np.ascontiguousarray(s_np), int64_dtype, a.device)
    arr = _to_numpy(a)
    if dim is None:
        count = np.count_nonzero(arr)
        if keepdim:
            out = np.array(count, dtype=np.int64).reshape((1,) * arr.ndim)
        else:
            out = np.array(count, dtype=np.int64)
    else:
        out = np.count_nonzero(arr, axis=dim, keepdims=keepdim).astype(np.int64)
    return _from_numpy(out, int64_dtype, a.device)

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
    return _from_numpy(np.cumsum(_to_numpy(a), axis=dim), a.dtype, a.device)

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
    return _from_numpy(np.cumprod(_to_numpy(a), axis=dim), a.dtype, a.device)

def cummax(a, dim=0):
    arr = _to_numpy(a)
    if dim < 0:
        dim += arr.ndim
    moved = np.moveaxis(arr, dim, 0)
    values = np.empty_like(moved)
    indices = np.empty(moved.shape, dtype=np.int64)
    max_vals = moved[0].copy()
    values[0] = max_vals
    indices[0] = 0
    for i in range(1, moved.shape[0]):
        mask = moved[i] > max_vals
        max_vals = np.where(mask, moved[i], max_vals)
        values[i] = max_vals
        indices[i] = np.where(mask, i, indices[i - 1])
    values = np.ascontiguousarray(np.moveaxis(values, 0, dim))
    indices = np.ascontiguousarray(np.moveaxis(indices, 0, dim))
    return (
        _from_numpy(values, a.dtype, a.device),
        _from_numpy(indices, int64_dtype, a.device),
    )

def cummin(a, dim):
    """Cumulative minimum along a dimension, returns (values, indices) namedtuple."""
    arr = _to_numpy(a)
    ndim = arr.ndim
    if dim < 0:
        dim = dim + ndim

    values = np.minimum.accumulate(arr, axis=dim)

    # Compute indices: for each position i along dim, index is where min first occurred
    n = arr.shape[dim]
    indices = np.zeros_like(arr, dtype=np.int64)

    # Iterate over the dimension to compute the argmin up to each point
    idx_shape = list(arr.shape)
    idx_shape[dim] = 1
    running_min = np.take(arr, [0], axis=dim)
    running_idx = np.zeros(idx_shape, dtype=np.int64)

    slc_base = [slice(None)] * ndim
    for i in range(n):
        slc = slc_base[:]
        slc[dim] = slice(i, i + 1)
        current = arr[tuple(slc)]
        new_min_mask = current < running_min
        running_idx = np.where(new_min_mask, i, running_idx)
        running_min = np.minimum(running_min, current)
        indices_slc = slc_base[:]
        indices_slc[dim] = i
        indices[tuple(indices_slc)] = running_idx.squeeze(axis=dim)

    from collections import namedtuple
    CumminResult = namedtuple("cummin", ["values", "indices"])
    return CumminResult(
        _from_numpy(np.ascontiguousarray(values), a.dtype, a.device),
        _from_numpy(np.ascontiguousarray(indices), int64_dtype, a.device),
    )


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
    arr = _to_numpy(a)
    kind = "stable" if stable else "quicksort"
    if descending:
        idx = np.argsort(-arr, axis=dim, kind=kind)
    else:
        idx = np.argsort(arr, axis=dim, kind=kind)
    return _from_numpy(idx.astype(np.int64), int64_dtype, a.device)

def sort(a, dim=-1, descending=False, stable=False):
    if (_can_use_gpu(a) and a.is_contiguous()
            and a.dtype in (float32_dtype, float16_dtype)):
        values_buf, indices_buf, numel = _sort_gpu(a, dim, descending)
        from ...._tensor import _compute_strides
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
    arr = _to_numpy(a)
    kind = "stable" if stable else "quicksort"
    if descending:
        idx = np.argsort(-arr, axis=dim, kind=kind)
    else:
        idx = np.argsort(arr, axis=dim, kind=kind)
    values = np.take_along_axis(arr, idx, axis=dim)
    return (
        _from_numpy(values, a.dtype, a.device),
        _from_numpy(idx.astype(np.int64), int64_dtype, a.device),
    )

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
    arr = _to_numpy(a)
    if largest:
        idx = np.argsort(-arr, axis=dim)
    else:
        idx = np.argsort(arr, axis=dim)
    idx = np.take(idx, np.arange(k), axis=dim)
    values = np.take_along_axis(arr, idx, axis=dim)
    return (
        _from_numpy(values, a.dtype, a.device),
        _from_numpy(idx.astype(np.int64), int64_dtype, a.device),
    )

def min_(a, b):
    return _from_numpy(np.minimum(_to_numpy(a), _to_numpy(b)), a.dtype, a.device)

def max_(a, b):
    return _from_numpy(np.maximum(_to_numpy(a), _to_numpy(b)), a.dtype, a.device)

def amin(a, dim=None, keepdim=False):
    if dim is not None and _can_use_gpu(a) and a.is_contiguous():
        if isinstance(dim, int):
            return _gpu_reduce_single_dim(a, dim, "min", keepdim)
        ndim = len(a.shape)
        result = a
        for d in sorted([x % ndim for x in dim], reverse=True):
            result = _gpu_reduce_single_dim(result, d, "min", keepdim)
        return result
    arr = _to_numpy(a)
    out = np.amin(arr, axis=dim, keepdims=keepdim)
    return _from_numpy(out, a.dtype, a.device)

def amax(a, dim=None, keepdim=False):
    if dim is not None and _can_use_gpu(a) and a.is_contiguous():
        if isinstance(dim, int):
            return _gpu_reduce_single_dim(a, dim, "max", keepdim)
        ndim = len(a.shape)
        result = a
        for d in sorted([x % ndim for x in dim], reverse=True):
            result = _gpu_reduce_single_dim(result, d, "max", keepdim)
        return result
    arr = _to_numpy(a)
    out = np.amax(arr, axis=dim, keepdims=keepdim)
    return _from_numpy(out, a.dtype, a.device)

def fmin(a, b):
    return _from_numpy(np.fmin(_to_numpy(a), _to_numpy(b)), a.dtype, a.device)

def fmax(a, b):
    return _from_numpy(np.fmax(_to_numpy(a), _to_numpy(b)), a.dtype, a.device)

def maximum(a, b):
    if _can_use_gpu(a):
        return _dispatch_binary_gpu(a, b, "maximum")
    a_np = _to_numpy(a)
    b_np = _to_numpy(b) if isinstance(b, Tensor) else b
    return _from_numpy(np.maximum(a_np, b_np), a.dtype, a.device)

def minimum(a, b):
    if _can_use_gpu(a):
        return _dispatch_binary_gpu(a, b, "minimum")
    a_np = _to_numpy(a)
    b_np = _to_numpy(b) if isinstance(b, Tensor) else b
    return _from_numpy(np.minimum(a_np, b_np), a.dtype, a.device)

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

    arr = _to_numpy(a)
    max_val = np.max(arr, axis=dim, keepdims=True)
    exp_shifted = np.exp(arr - max_val)
    sum_exp = np.sum(exp_shifted, axis=dim, keepdims=keepdim)
    if keepdim:
        out = np.log(sum_exp) + max_val
    else:
        out = np.log(sum_exp) + np.squeeze(max_val, axis=dim)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)

def renorm(a, p, dim, maxnorm):
    """Renormalize tensor: each sub-tensor along dim has norm <= maxnorm."""
    arr = _to_numpy(a)
    # Compute the norm along all axes except dim
    norm = np.linalg.norm(arr, ord=p, axis=dim, keepdims=True)
    # Scale: if norm > maxnorm, scale down; otherwise keep unchanged
    scale = np.where(norm > maxnorm, maxnorm / (norm + 1e-7), 1.0)
    out = arr * scale
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)

def nansum(a, dim=None, keepdim=False):
    """Sum ignoring NaN values."""
    if (_can_use_gpu(a) and a.is_contiguous()
            and a.dtype in (float32_dtype, float16_dtype)):
        # composite: where(isnan(a), 0, a) → sum
        mask = isnan(a)
        not_nan = logical_not(mask)
        cleaned = where(not_nan, a, 0.0)
        return sum_(cleaned, dim=dim, keepdim=keepdim)
    arr = _to_numpy(a)
    if dim is None:
        out = np.nansum(arr)
        return _from_numpy(np.array(out, dtype=to_numpy_dtype(a.dtype)), a.dtype, a.device)
    else:
        out = np.nansum(arr, axis=dim, keepdims=keepdim)
        return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)

def nanmean(a, dim=None, keepdim=False):
    """Mean ignoring NaN values."""
    if (_can_use_gpu(a) and a.is_contiguous()
            and a.dtype in (float32_dtype, float16_dtype)):
        mask = isnan(a)
        not_nan = logical_not(mask)
        cleaned = where(not_nan, a, 0.0)
        s = sum_(cleaned, dim=dim, keepdim=keepdim)
        # Count non-NaN: use add_scalar(0*a, 1) to get float ones, then mask
        zeros = _dispatch_binary_gpu(a, 0.0, "mul")  # 0*a = 0 everywhere (incl NaN→0*NaN=NaN... no)
        # Simplest: fall back to numpy for count since composite is tricky
        arr = _to_numpy(a)
        cnt_val = np.sum(~np.isnan(arr), axis=dim, keepdims=keepdim).astype(arr.dtype)
        cnt = _from_numpy(np.ascontiguousarray(cnt_val), a.dtype, a.device)
        return div(s, cnt)
    arr = _to_numpy(a)
    if dim is None:
        out = np.nanmean(arr)
        return _from_numpy(np.array(out, dtype=to_numpy_dtype(a.dtype)), a.dtype, a.device)
    else:
        out = np.nanmean(arr, axis=dim, keepdims=keepdim)
        return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)

def aminmax(a, dim=None, keepdim=False):
    """Returns the min and max of a tensor."""
    from collections import namedtuple
    arr = _to_numpy(a)
    if dim is None:
        mn = np.min(arr)
        mx = np.max(arr)
        mn_t = _from_numpy(np.array(mn, dtype=to_numpy_dtype(a.dtype)), a.dtype, a.device)
        mx_t = _from_numpy(np.array(mx, dtype=to_numpy_dtype(a.dtype)), a.dtype, a.device)
    else:
        mn = np.min(arr, axis=dim, keepdims=keepdim)
        mx = np.max(arr, axis=dim, keepdims=keepdim)
        mn_t = _from_numpy(np.ascontiguousarray(mn), a.dtype, a.device)
        mx_t = _from_numpy(np.ascontiguousarray(mx), a.dtype, a.device)
    AminmaxResult = namedtuple("aminmax", ["min", "max"])
    return AminmaxResult(mn_t, mx_t)

def quantile(a, q, dim=None, keepdim=False):
    """Compute the q-th quantile of the input tensor."""
    arr = _to_numpy(a).astype(np.float64)
    q_val = _to_numpy(q) if hasattr(q, '_numpy_view') else np.asarray(q, dtype=np.float64)
    if dim is None:
        out = np.quantile(arr, q_val)
    else:
        out = np.quantile(arr, q_val, axis=dim, keepdims=keepdim)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)

def nanquantile(a, q, dim=None, keepdim=False):
    """Compute the q-th quantile ignoring NaN values."""
    arr = _to_numpy(a).astype(np.float64)
    q_val = _to_numpy(q) if hasattr(q, '_numpy_view') else np.asarray(q, dtype=np.float64)
    if dim is None:
        out = np.nanquantile(arr, q_val)
    else:
        out = np.nanquantile(arr, q_val, axis=dim, keepdims=keepdim)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)

def nanmedian(a, dim=None, keepdim=False):
    """Median ignoring NaN values. Returns (values, indices) when dim is given."""
    arr = _to_numpy(a).astype(np.float64)
    if dim is None:
        out = np.nanmedian(arr)
        return _from_numpy(np.array(out, dtype=to_numpy_dtype(a.dtype)), a.dtype, a.device)
    else:
        values = np.nanmedian(arr, axis=dim, keepdims=keepdim)
        # Compute indices: for each slice along dim, find index of the median value
        n = arr.shape[dim]
        sorted_arr = np.sort(arr, axis=dim)
        # Count non-nan along dim
        not_nan = ~np.isnan(arr)
        count = np.sum(not_nan, axis=dim, keepdims=True)
        # Median index in sorted order
        med_idx_sorted = (count - 1) // 2
        # For each position, find the index in the original array
        sorted_indices = np.argsort(arr, axis=dim)
        # Gather the median index from sorted_indices
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
    arr = _to_numpy(a)
    if dim is None:
        out = np.median(arr.flatten())
        return _from_numpy(np.array(out, dtype=arr.dtype), a.dtype, a.device)
    else:
        if dim < 0:
            dim = dim + arr.ndim
        sorted_idx = np.argsort(arr, axis=dim)
        n = arr.shape[dim]
        mid = n // 2
        med_idx = np.take(sorted_idx, [mid], axis=dim)
        values = np.take_along_axis(arr, med_idx, axis=dim)
        if not keepdim:
            values = np.squeeze(values, axis=dim)
            med_idx = np.squeeze(med_idx, axis=dim)
        return (
            _from_numpy(np.ascontiguousarray(values), a.dtype, a.device),
            _from_numpy(np.ascontiguousarray(med_idx.astype(np.int64)), int64_dtype, a.device),
        )


# ---------------------------------------------------------------------------
# Group 7: New math ops for Tensor API alignment
# ---------------------------------------------------------------------------

def kthvalue(a, k, dim=-1, keepdim=False):
    arr = _to_numpy(a)
    if dim < 0:
        dim = dim + arr.ndim
    sorted_idx = np.argsort(arr, axis=dim)
    kth_idx = np.take(sorted_idx, [k - 1], axis=dim)
    values = np.take_along_axis(arr, kth_idx, axis=dim)
    if not keepdim:
        values = np.squeeze(values, axis=dim)
        kth_idx = np.squeeze(kth_idx, axis=dim)
    return (
        _from_numpy(np.ascontiguousarray(values), a.dtype, a.device),
        _from_numpy(np.ascontiguousarray(kth_idx.astype(np.int64)), int64_dtype, a.device),
    )

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

