import math
import ctypes
import struct
import numpy as np

from ._helpers import (
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
            from .comparison import ne
            cond_u8 = ne(cond, 0)
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
            # y is a tensor but non-contiguous or different shape — make contiguous and retry
            y_c = y.contiguous() if not y.is_contiguous() else y
            if y_c.shape != x.shape:
                _unsupported_dtype("where (shape mismatch)", x)
            d.dispatch_where(f"where_{sfx}", cond_buf, _metal_buf(x),
                             _metal_buf(y_c), out_buf, numel)
        return _from_metal_buffer(out_buf, x.shape, x.stride, x.dtype, x.device)

    # Non-contiguous inputs: make contiguous and retry
    if isinstance(x, Tensor) and _can_use_gpu(x) and isinstance(cond, Tensor) and _can_use_gpu(cond):
        cond_c = cond.contiguous() if not cond.is_contiguous() else cond
        x_c = x.contiguous() if not x.is_contiguous() else x
        return where(cond_c, x_c, y)
    _unsupported_dtype("where", x)

def lerp(a, b, weight):
    if _can_use_gpu(a) and _can_use_gpu(b):
        # lerp(a, b, w) = a + w * (b - a)
        diff = sub(b, a)
        if isinstance(weight, Tensor):
            return add(a, mul(diff, weight))
        else:
            return add(a, mul(diff, weight))
    _unsupported_dtype("lerp", a)

def addcmul(a, b, c, value=1.0):
    if _can_use_gpu(a) and _can_use_gpu(b) and _can_use_gpu(c):
        return add(a, mul(mul(b, c), value))
    _unsupported_dtype("addcmul", a)

def addcdiv(a, b, c, value=1.0):
    if _can_use_gpu(a) and _can_use_gpu(b) and _can_use_gpu(c):
        return add(a, mul(div(b, c), value))
    _unsupported_dtype("addcdiv", a)

def logaddexp(a, b):
    if isinstance(a, Tensor) and isinstance(b, Tensor) and _can_use_gpu(a) and _can_use_gpu(b):
        return _dispatch_binary_gpu(a, b, "logaddexp")
    _unsupported_dtype("logaddexp", a if isinstance(a, Tensor) else b)

def logaddexp2(a, b):
    if isinstance(a, Tensor) and isinstance(b, Tensor) and _can_use_gpu(a) and _can_use_gpu(b):
        return _dispatch_binary_gpu(a, b, "logaddexp2")
    _unsupported_dtype("logaddexp2", a if isinstance(a, Tensor) else b)

def hypot(a, b):
    if isinstance(a, Tensor) and isinstance(b, Tensor) and _can_use_gpu(a) and _can_use_gpu(b):
        return _dispatch_binary_gpu(a, b, "hypot")
    _unsupported_dtype("hypot", a if isinstance(a, Tensor) else b)

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
            scalar = float(b) if not isinstance(b, Tensor) else float(b.item())
            d.dispatch_binary_scalar(f"remainder_scalar_{sfx}", _metal_buf(a),
                                     scalar, out_buf, numel,
                                     scalar_fmt=_scalar_fmt(a.dtype))
        return _from_metal_buffer(out_buf, a.shape, a.stride, a.dtype, a.device)
    _unsupported_dtype("remainder", a if isinstance(a, Tensor) else b)

def fmod(a, b):
    if _can_use_gpu(a) and isinstance(b, Tensor) and _can_use_gpu(b):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, a.dtype)
        d.dispatch_binary(f"fmod_{sfx}", _metal_buf(a), _metal_buf(b),
                          out_buf, numel)
        return _from_metal_buffer(out_buf, a.shape, a.stride, a.dtype, a.device)
    _unsupported_dtype("fmod", a)

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
    # Non-contiguous: make contiguous and retry
    if _can_use_gpu(a) and isinstance(values, Tensor) and _can_use_gpu(values):
        a_c = a.contiguous() if not a.is_contiguous() else a
        v_c = values.contiguous() if not values.is_contiguous() else values
        from .shape import expand
        if a_c.shape != v_c.shape:
            v_c = expand(v_c, tuple(a_c.shape))
        return heaviside(a_c, v_c)
    _unsupported_dtype("heaviside", a)


# ---------------------------------------------------------------------------
# torch.linalg ops
# ---------------------------------------------------------------------------

def diff(a, n=1, dim=-1, prepend=None, append=None):
    """Compute the n-th discrete difference along the given dim."""
    if _can_use_gpu(a):
        from .shape import narrow, cat
        ndim = len(a.shape)
        d = dim if dim >= 0 else dim + ndim
        # Handle prepend/append by concatenating on-device
        src = a
        if prepend is not None or append is not None:
            pieces = []
            if prepend is not None:
                pieces.append(prepend)
            pieces.append(src)
            if append is not None:
                pieces.append(append)
            src = cat(pieces, dim=d)
        # Iterate n times: diff = src[1:] - src[:-1] along dim
        for _ in range(n):
            size = src.shape[d]
            # .contiguous() on narrow views creates separate GPU buffers,
            # avoiding aliased _metal_buf issues with sub()
            head = narrow(src, d, 1, size - 1).contiguous()
            tail = narrow(src, d, 0, size - 1).contiguous()
            src = sub(head, tail)
        return src
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
    if _can_use_gpu(elements):
        from .comparison import eq, logical_or
        from ._helpers import _read_scalar
        # Iterate over test_elements, accumulate eq() with logical_or()
        te_flat = test_elements.contiguous().reshape((-1,))
        n_test = te_flat.numel()
        result = eq(elements, _read_scalar(te_flat[0]) if n_test > 0 else 0)
        for i in range(1, n_test):
            result = logical_or(result, eq(elements, _read_scalar(te_flat[i])))
        return result
    e = _to_numpy(elements)
    te = _to_numpy(test_elements)
    out = np.isin(e, te)
    return _from_numpy(np.ascontiguousarray(out), bool_dtype, elements.device)

def uniform(a):
    """Return tensor of same shape filled with Uniform(0,1) samples."""
    from ._helpers import _empty_like
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        from .random import uniform_
        out = _empty_like(a)
        uniform_(out, 0.0, 1.0)
        return out
    from ...._random import _get_cpu_rng
    rng = _get_cpu_rng()
    arr = rng.uniform(0.0, 1.0, _to_numpy(a).shape).astype(_to_numpy(a).dtype)
    return _from_numpy(arr, a.dtype, a.device)


# ---------------------------------------------------------------------------
# Upsample ops — CPU numpy implementations
# ---------------------------------------------------------------------------

