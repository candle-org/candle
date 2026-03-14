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
from .math import sub, add, mul, div
from .comparison import gt, lt, logical_not


def where(cond, x, y):
    # GPU path: both cond and x are GPU+contiguous
    if (isinstance(x, Tensor) and _can_use_gpu(x) and x.is_contiguous()
            and isinstance(cond, Tensor) and _can_use_gpu(cond) and cond.is_contiguous()):
        d = _get_dispatcher()
        sfx = _kernel_suffix(x.dtype)
        numel = x.numel()
        out_buf = _alloc_output_buf(numel, x.dtype)
        # Ensure condition is uint8 (uchar) for the shader
        if cond.dtype != bool_dtype:
            cond_u8 = _from_numpy(
                _to_numpy(cond).astype(np.bool_).astype(np.uint8),
                bool_dtype, cond.device)
        else:
            cond_u8 = cond
        cond_buf = _metal_buf(cond_u8)

        if isinstance(y, Tensor) and _can_use_gpu(y) and y.is_contiguous() and y.shape == x.shape:
            # Both tensors, same shape
            d.dispatch_where(f"where_{sfx}", cond_buf, _metal_buf(x),
                             _metal_buf(y), out_buf, numel)
        elif not isinstance(y, Tensor):
            # y is scalar
            scalar_val = float(y) if x.dtype in (float32_dtype, float16_dtype) else int(y)
            d.dispatch_where_scalar(f"where_scalar_y_{sfx}", cond_buf,
                                    _metal_buf(x), scalar_val, out_buf,
                                    numel, scalar_fmt=_scalar_fmt(x.dtype))
        else:
            # Fallback to numpy
            cond_arr = _to_numpy(cond)
            x_arr = _to_numpy(x)
            y_arr = _to_numpy(y)
            out = np.where(cond_arr, x_arr, y_arr)
            return _from_numpy(out, x.dtype, x.device)
        return _from_metal_buffer(out_buf, x.shape, x.stride, x.dtype, x.device)

    # numpy fallback
    cond_arr = _to_numpy(cond)
    x_arr = _to_numpy(x)
    if isinstance(y, Tensor):
        y_arr = _to_numpy(y)
    else:
        y_arr = y
    out = np.where(cond_arr, x_arr, y_arr)
    return _from_numpy(out, x.dtype, x.device)

def lerp(a, b, weight):
    if _can_use_gpu(a) and _can_use_gpu(b):
        # lerp(a, b, w) = a + w * (b - a)
        diff = sub(b, a)
        if isinstance(weight, Tensor):
            return add(a, mul(diff, weight))
        else:
            return add(a, mul(diff, weight))
    arr_a = _to_numpy(a)
    arr_b = _to_numpy(b)
    if isinstance(weight, Tensor):
        w = _to_numpy(weight)
    else:
        w = weight
    out = arr_a + w * (arr_b - arr_a)
    return _from_numpy(out, a.dtype, a.device)

def addcmul(a, b, c, value=1.0):
    if _can_use_gpu(a) and _can_use_gpu(b) and _can_use_gpu(c):
        return add(a, mul(mul(b, c), value))
    return _from_numpy(
        _to_numpy(a) + value * (_to_numpy(b) * _to_numpy(c)),
        a.dtype,
        a.device,
    )

def addcdiv(a, b, c, value=1.0):
    if _can_use_gpu(a) and _can_use_gpu(b) and _can_use_gpu(c):
        return add(a, mul(div(b, c), value))
    return _from_numpy(
        _to_numpy(a) + value * (_to_numpy(b) / _to_numpy(c)),
        a.dtype,
        a.device,
    )

def logaddexp(a, b):
    if isinstance(a, Tensor) and isinstance(b, Tensor) and _can_use_gpu(a) and _can_use_gpu(b):
        return _dispatch_binary_gpu(a, b, "logaddexp")
    return _from_numpy(np.logaddexp(_to_numpy(a), _to_numpy(b)), a.dtype, a.device)

def logaddexp2(a, b):
    if isinstance(a, Tensor) and isinstance(b, Tensor) and _can_use_gpu(a) and _can_use_gpu(b):
        return _dispatch_binary_gpu(a, b, "logaddexp2")
    return _from_numpy(np.logaddexp2(_to_numpy(a), _to_numpy(b)), a.dtype, a.device)

def hypot(a, b):
    if isinstance(a, Tensor) and isinstance(b, Tensor) and _can_use_gpu(a) and _can_use_gpu(b):
        return _dispatch_binary_gpu(a, b, "hypot")
    return _from_numpy(np.hypot(_to_numpy(a), _to_numpy(b)), a.dtype, a.device)

def remainder(a, b):
    if isinstance(a, Tensor) and _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, a.dtype)
        if isinstance(b, Tensor) and _can_use_gpu(b):
            d.dispatch_binary(f"remainder_{sfx}", _metal_buf(a), _metal_buf(b),
                              out_buf, numel)
        else:
            scalar = float(b) if not isinstance(b, Tensor) else float(_to_numpy(b).ravel()[0])
            d.dispatch_binary_scalar(f"remainder_scalar_{sfx}", _metal_buf(a),
                                     scalar, out_buf, numel,
                                     scalar_fmt=_scalar_fmt(a.dtype))
        return _from_metal_buffer(out_buf, a.shape, a.stride, a.dtype, a.device)
    a_np = _to_numpy(a) if isinstance(a, Tensor) else a
    b_np = _to_numpy(b) if isinstance(b, Tensor) else b
    ref = a if isinstance(a, Tensor) else b
    return _from_numpy(np.remainder(a_np, b_np), ref.dtype, ref.device)

def fmod(a, b):
    if _can_use_gpu(a) and isinstance(b, Tensor) and _can_use_gpu(b):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, a.dtype)
        d.dispatch_binary(f"fmod_{sfx}", _metal_buf(a), _metal_buf(b),
                          out_buf, numel)
        return _from_metal_buffer(out_buf, a.shape, a.stride, a.dtype, a.device)
    return _from_numpy(np.fmod(_to_numpy(a), _to_numpy(b)), a.dtype, a.device)

def heaviside(a, values):
    """Heaviside step function."""
    if (_can_use_gpu(a) and a.is_contiguous()
            and a.dtype in (float32_dtype, float16_dtype)
            and isinstance(values, Tensor) and _can_use_gpu(values)
            and values.is_contiguous() and a.shape == values.shape):
        # composite: where(a > 0, 1, where(a == 0, values, 0))
        # Use: where(neg_mask, 0, where(pos_mask, 1, values)) with scalar y
        pos_mask = gt(a, 0)
        # where(pos_mask, values, values) but replace true branch with 1.0:
        # where(~pos_mask, values, scalar=1.0) → need x=values, y=1.0
        neg_mask = lt(a, 0)
        # Step 1: start with values (used at a==0)
        # Step 2: where a>0, set to 1.0 → where(~pos_mask, values, 1.0) → x=values, y=1.0, cond=~pos_mask
        not_pos = logical_not(pos_mask)
        out = where(not_pos, values, 1.0)
        # Step 3: where a<0, set to 0.0 → where(~neg_mask, out, 0.0)
        not_neg = logical_not(neg_mask)
        return where(not_neg, out, 0.0)
    a_np = _to_numpy(a)
    v_np = _to_numpy(values)
    out = np.heaviside(a_np, v_np)
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)


# ---------------------------------------------------------------------------
# torch.linalg ops
# ---------------------------------------------------------------------------

def diff(a, n=1, dim=-1, prepend=None, append=None):
    """Compute the n-th discrete difference along the given dim."""
    arr = _to_numpy(a)
    if prepend is not None or append is not None:
        pieces = []
        if prepend is not None:
            pieces.append(_to_numpy(prepend))
        pieces.append(arr)
        if append is not None:
            pieces.append(_to_numpy(append))
        arr = np.concatenate(pieces, axis=dim)
    out = np.diff(arr, n=n, axis=dim)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)

def bincount(a, weights=None, minlength=0):
    """Count number of occurrences of each value in a 1-D int tensor."""
    arr = _to_numpy(a).astype(np.int64).ravel()
    w = _to_numpy(weights).ravel() if weights is not None else None
    out = np.bincount(arr, weights=w, minlength=minlength)
    out_dtype = a.dtype if weights is None else weights.dtype
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(out_dtype))), out_dtype, a.device)

def histc(a, bins=100, min=0, max=0):
    """Histogram with equal-width bins (1-D output count tensor)."""
    arr = _to_numpy(a).ravel().astype(np.float64)
    lo = float(min)
    hi = float(max)
    if lo == 0 and hi == 0:
        lo = float(np.min(arr))
        hi = float(np.max(arr))
    out, _ = np.histogram(arr, bins=bins, range=(lo, hi))
    return _from_numpy(np.ascontiguousarray(out.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device)

def histogram(a, bins, range=None, weight=None, density=False):
    """Histogram returning (hist, bin_edges)."""
    arr = _to_numpy(a).ravel().astype(np.float64)
    bins_val = _to_numpy(bins) if hasattr(bins, '_numpy_view') else bins
    w = _to_numpy(weight).ravel().astype(np.float64) if weight is not None else None
    hist, edges = np.histogram(arr, bins=bins_val, range=range, weights=w, density=density)
    return (
        _from_numpy(np.ascontiguousarray(hist.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device),
        _from_numpy(np.ascontiguousarray(edges.astype(to_numpy_dtype(a.dtype))), a.dtype, a.device),
    )

def bucketize(a, boundaries, out_int32=False, right=False):
    """Maps values to bucket indices using boundaries."""
    arr = _to_numpy(a)
    b = _to_numpy(boundaries).ravel()
    side = 'right' if not right else 'left'
    out = np.searchsorted(b, arr, side=side)
    out_np_dtype = np.int32 if out_int32 else np.int64
    out_dtype = int64_dtype
    return _from_numpy(np.ascontiguousarray(out.astype(out_np_dtype)), out_dtype, a.device)

def isin(elements, test_elements):
    """Tests if each element is in test_elements."""
    e = _to_numpy(elements)
    te = _to_numpy(test_elements)
    out = np.isin(e, te)
    return _from_numpy(np.ascontiguousarray(out), bool_dtype, elements.device)

def uniform(a):
    """Return tensor of same shape filled with Uniform(0,1) samples."""
    from ...._random import _get_cpu_rng
    rng = _get_cpu_rng()
    arr = rng.uniform(0.0, 1.0, _to_numpy(a).shape).astype(_to_numpy(a).dtype)
    return _from_numpy(arr, a.dtype, a.device)


# ---------------------------------------------------------------------------
# Upsample ops — CPU numpy implementations
# ---------------------------------------------------------------------------

