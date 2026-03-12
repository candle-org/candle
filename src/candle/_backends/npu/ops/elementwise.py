"""Miscellaneous element-wise operations for NPU."""
from ._helpers import (
    _unwrap_storage, _wrap_tensor, _unary_op, _binary_op,
    _broadcast_shape, _broadcast_shape_checked,
    _numel, _dtype_itemsize, _use_soc_fallback,
    _scalar_to_npu_tensor, _scalar_to_npu_tensor_no_add,
    _npu_broadcast_to, _npu_arange_1d, _npu_linear_index,
    _npu_add_scalar_, npu_index_put_impl,
    _normalize_reduction_dims, _reduce_out_shape,
    _cast_tensor_dtype, _normalize_tensor_sequence_args,
    _matmul_out_shape,
    _iter_indices, _broadcast_index, _batch_offset,
    bool_dtype, int32_dtype, int64_dtype, float_dtype,
    npu_typed_storage_from_ptr, reshape,
    aclnn, npu_runtime, npu_state, ops_soc,
)
from .comparison import eq, gt, logical_and, logical_or, lt, ne
from .math import add, div, mul, sqrt, sub
from .reduce import searchsorted
from .shape import contiguous, index_put_, index_select, masked_select, nonzero, split


def where(cond, x, y):
    if x.device.type != "npu":
        raise ValueError("NPU where expects NPU tensors")
    if isinstance(cond, (int, float)):
        cond = _scalar_to_npu_tensor(cond, x)
    if isinstance(y, (int, float)):
        y = _scalar_to_npu_tensor(y, x)
    if cond.device.type != "npu" or y.device.type != "npu":
        raise ValueError("NPU where expects NPU tensors")
    if x.dtype != y.dtype:
        raise ValueError("NPU where requires matching dtypes")
    if cond.dtype != bool_dtype:
        cond = ne(cond, _scalar_to_npu_tensor(0, cond))

    out_shape = _broadcast_shape(cond.shape, x.shape)
    out_shape = _broadcast_shape(out_shape, y.shape)
    if out_shape != x.shape:
        x = _npu_broadcast_to(x, out_shape)
    if out_shape != y.shape:
        y = _npu_broadcast_to(y, out_shape)
    if out_shape != cond.shape:
        cond = _npu_broadcast_to(cond, out_shape)

    if _use_soc_fallback("where"):
        out = contiguous(y)
        idx = nonzero(cond, as_tuple=True)
        if len(idx) == 0 or idx[0].numel() == 0:
            return out
        vals = masked_select(x, cond)
        return index_put_(out, idx, vals, accumulate=False)

    if not aclnn.s_where_symbols_ok():
        raise RuntimeError("aclnnSWhere symbols not available")

    runtime = npu_runtime.get_runtime((x.device.index or 0))
    stream = npu_state.current_stream((x.device.index or 0))
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_ptr = npu_runtime._alloc_device(_numel(out_shape) * _dtype_itemsize(x.dtype), runtime=runtime)
    aclnn.s_where(
        _unwrap_storage(cond).data_ptr(),
        _unwrap_storage(x).data_ptr(),
        _unwrap_storage(y).data_ptr(),
        out_ptr,
        cond.shape,
        cond.stride,
        cond.dtype,
        x.shape,
        x.stride,
        x.dtype,
        y.shape,
        y.stride,
        y.dtype,
        out_shape,
        out_stride,
        x.dtype,
        runtime,
        stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), x.dtype, device=x.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def lerp(a, b, weight):
    if _use_soc_fallback("lerp"):
        # Static small-op fallback on 310B to avoid aclnnLerp 561103.
        delta = sub(b, a)
        scaled = mul(delta, weight)
        return add(a, scaled)

    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    a_storage = _unwrap_storage(a)
    b_storage = _unwrap_storage(b)
    if hasattr(weight, "shape"):
        # Tensor weight path
        w_storage = _unwrap_storage(weight)
        out_shape = _broadcast_shape(_broadcast_shape(a.shape, b.shape), weight.shape)
        out_stride = npu_runtime._contiguous_stride(out_shape)
        out_size = _numel(out_shape) * _dtype_itemsize(a.dtype)
        out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
        aclnn.lerp_tensor(
            a_storage.data_ptr(), b_storage.data_ptr(), w_storage.data_ptr(), out_ptr,
            a.shape, a.stride, b.shape, b.stride,
            weight.shape, weight.stride, out_shape, out_stride,
            a.dtype, runtime, stream=stream.stream,
        )
    else:
        # Scalar weight path
        out_shape = _broadcast_shape(a.shape, b.shape)
        out_stride = npu_runtime._contiguous_stride(out_shape)
        out_size = _numel(out_shape) * _dtype_itemsize(a.dtype)
        out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
        aclnn.lerp_scalar(
            a_storage.data_ptr(), b_storage.data_ptr(), out_ptr,
            a.shape, a.stride, b.shape, b.stride,
            out_shape, out_stride, a.dtype, float(weight),
            runtime, stream=stream.stream,
        )
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def addcmul(a, b, c, value=1.0):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    a_storage = _unwrap_storage(a)
    b_storage = _unwrap_storage(b)
    c_storage = _unwrap_storage(c)
    out_shape = _broadcast_shape(_broadcast_shape(a.shape, b.shape), c.shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_size = _numel(out_shape) * _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    if hasattr(value, "shape"):
        value = float(_to_numpy(value))
    aclnn.addcmul(
        a_storage.data_ptr(), b_storage.data_ptr(), c_storage.data_ptr(), out_ptr,
        a.shape, a.stride, b.shape, b.stride,
        c.shape, c.stride, out_shape, out_stride,
        a.dtype, float(value), runtime, stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def addcdiv(a, b, c, value=1.0):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    a_storage = _unwrap_storage(a)
    b_storage = _unwrap_storage(b)
    c_storage = _unwrap_storage(c)
    out_shape = _broadcast_shape(_broadcast_shape(a.shape, b.shape), c.shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_size = _numel(out_shape) * _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    if hasattr(value, "shape"):
        value = float(_to_numpy(value))
    aclnn.addcdiv(
        a_storage.data_ptr(), b_storage.data_ptr(), c_storage.data_ptr(), out_ptr,
        a.shape, a.stride, b.shape, b.stride,
        c.shape, c.stride, out_shape, out_stride,
        a.dtype, float(value), runtime, stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def logaddexp(a, b):
    return _binary_op(a, b, aclnn.slogaddexp, "logaddexp")


def logaddexp2(a, b):
    return _binary_op(a, b, aclnn.slogaddexp2, "logaddexp2")


def hypot(a, b):
    return sqrt(add(mul(a, a), mul(b, b)))


def _remainder_310b_fallback(a, b):
    # torch-style remainder keeps the sign of divisor b.
    r = fmod(a, b)
    zero = _scalar_to_npu_tensor(0, r)
    nz = ne(r, zero)
    r_neg = lt(r, zero)
    b_neg = lt(b, zero)
    mismatch = ne(r_neg, b_neg)
    fix = logical_and(nz, mismatch)
    return where(fix, add(r, b), r)


def remainder(a, b):
    if isinstance(b, (int, float)):
        b = _scalar_to_npu_tensor(b, a)
    if _use_soc_fallback("remainder"):
        return _remainder_310b_fallback(a, b)
    return _binary_op(a, b, aclnn.sremainder, "remainder")


def fmod(a, b):
    if isinstance(b, (int, float)):
        b = _scalar_to_npu_tensor(b, a)
    return _binary_op(a, b, aclnn.sfmod, "fmod")


def clamp(a, min_val=None, max_val=None):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU clamp expects NPU tensors")
    if min_val is None and max_val is None:
        raise ValueError("clamp requires min or max")
    out_shape = a.shape
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_size = _numel(out_shape) * _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    storage = _unwrap_storage(a)
    if hasattr(min_val, "shape") and hasattr(max_val, "shape"):
        out_shape = _broadcast_shape(a.shape, min_val.shape)
        out_shape = _broadcast_shape(out_shape, max_val.shape)
        out_stride = npu_runtime._contiguous_stride(out_shape)
        out_size = _numel(out_shape) * _dtype_itemsize(a.dtype)
        out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
        aclnn.clamp_tensor(
            storage.data_ptr(),
            _unwrap_storage(min_val).data_ptr(),
            _unwrap_storage(max_val).data_ptr(),
            out_ptr,
            a.shape,
            a.stride,
            min_val.shape,
            min_val.stride,
            max_val.shape,
            max_val.stride,
            out_shape,
            out_stride,
            a.dtype,
            runtime,
            stream=stream.stream,
        )
    elif hasattr(min_val, "shape"):
        out_shape = _broadcast_shape(a.shape, min_val.shape)
        out_stride = npu_runtime._contiguous_stride(out_shape)
        out_size = _numel(out_shape) * _dtype_itemsize(a.dtype)
        out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
        aclnn.clamp_min_tensor(
            storage.data_ptr(),
            _unwrap_storage(min_val).data_ptr(),
            out_ptr,
            a.shape,
            a.stride,
            min_val.shape,
            min_val.stride,
            out_shape,
            out_stride,
            a.dtype,
            runtime,
            stream=stream.stream,
        )
        if max_val is not None:
            temp_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), a.dtype, device=a.device)
            temp_tensor = _wrap_tensor(temp_storage, out_shape, out_stride)
            return clamp_max(temp_tensor, max_val)
    elif hasattr(max_val, "shape"):
        out_shape = _broadcast_shape(a.shape, max_val.shape)
        out_stride = npu_runtime._contiguous_stride(out_shape)
        out_size = _numel(out_shape) * _dtype_itemsize(a.dtype)
        out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
        aclnn.clamp_max_tensor(
            storage.data_ptr(),
            _unwrap_storage(max_val).data_ptr(),
            out_ptr,
            a.shape,
            a.stride,
            max_val.shape,
            max_val.stride,
            out_shape,
            out_stride,
            a.dtype,
            runtime,
            stream=stream.stream,
        )
        if min_val is not None:
            temp_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), a.dtype, device=a.device)
            temp_tensor = _wrap_tensor(temp_storage, out_shape, out_stride)
            return clamp_min(temp_tensor, min_val)
    else:
        aclnn.clamp_scalar(
            storage.data_ptr(),
            out_ptr,
            a.shape,
            a.stride,
            a.dtype,
            min_val,
            max_val,
            runtime,
            stream=stream.stream,
        )
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def clamp_min(a, min_val):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU clamp_min expects NPU tensors")
    storage = _unwrap_storage(a)
    if hasattr(min_val, "shape"):
        out_shape = _broadcast_shape(a.shape, min_val.shape)
        out_stride = npu_runtime._contiguous_stride(out_shape)
        out_size = _numel(out_shape) * _dtype_itemsize(a.dtype)
        out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
        aclnn.clamp_min_tensor(
            storage.data_ptr(),
            _unwrap_storage(min_val).data_ptr(),
            out_ptr,
            a.shape,
            a.stride,
            min_val.shape,
            min_val.stride,
            out_shape,
            out_stride,
            a.dtype,
            runtime,
            stream=stream.stream,
        )
    else:
        out_shape = a.shape
        out_stride = npu_runtime._contiguous_stride(out_shape)
        out_size = _numel(out_shape) * _dtype_itemsize(a.dtype)
        out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
        aclnn.clamp_min_scalar(
            storage.data_ptr(),
            out_ptr,
            a.shape,
            a.stride,
            a.dtype,
            min_val,
            runtime,
            stream=stream.stream,
        )
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def clamp_max(a, max_val):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU clamp_max expects NPU tensors")
    storage = _unwrap_storage(a)
    if hasattr(max_val, "shape"):
        out_shape = _broadcast_shape(a.shape, max_val.shape)
        out_stride = npu_runtime._contiguous_stride(out_shape)
        out_size = _numel(out_shape) * _dtype_itemsize(a.dtype)
        out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
        aclnn.clamp_max_tensor(
            storage.data_ptr(),
            _unwrap_storage(max_val).data_ptr(),
            out_ptr,
            a.shape,
            a.stride,
            max_val.shape,
            max_val.stride,
            out_shape,
            out_stride,
            a.dtype,
            runtime,
            stream=stream.stream,
        )
    else:
        out_shape = a.shape
        out_stride = npu_runtime._contiguous_stride(out_shape)
        out_size = _numel(out_shape) * _dtype_itemsize(a.dtype)
        out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
        aclnn.clamp_max_scalar(
            storage.data_ptr(),
            out_ptr,
            a.shape,
            a.stride,
            a.dtype,
            max_val,
            runtime,
            stream=stream.stream,
        )
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def heaviside_op(a, values):
    """Heaviside step function."""
    zero = _scalar_to_npu_tensor(0, a)
    one = _scalar_to_npu_tensor(1, a)
    pos_mask = gt(a, zero)
    eq_mask = eq(a, zero)
    # result = where(a > 0, 1, where(a == 0, values, 0))
    inner_result = where(eq_mask, values, zero)
    return where(pos_mask, one, inner_result)


def uniform_op(a):
    """Return tensor of same shape filled with Uniform(0,1) samples."""
    from ...._dispatch.dispatcher import dispatch
    from ...._creation import rand
    return rand(a.shape, dtype=a.dtype, device=a.device)


def isreal_op(a):
    """Returns bool tensor: True for all elements if dtype is non-complex."""
    from ...._dispatch.dispatcher import dispatch
    dtype_name = str(a.dtype).split(".")[-1]
    is_complex = "complex" in dtype_name
    if is_complex:
        # For complex tensors, check imag == 0
        # Since we don't have complex support on NPU, just return all True
        return dispatch("ones", "npu", a.shape, dtype=bool_dtype, device=a.device)
    else:
        return dispatch("ones", "npu", a.shape, dtype=bool_dtype, device=a.device)


def isin_op(elements, test_elements):
    """Tests if each element is in test_elements."""
    from ...._dispatch.dispatcher import dispatch
    from ...common import view as view_backend
    te_flat = dispatch("flatten", "npu", test_elements)
    te_len = te_flat.shape[0]
    elem_shape = elements.shape
    elem_flat = dispatch("flatten", "npu", elements)
    n = elem_flat.shape[0]
    # Loop over test elements, use tile to replicate (expand has NPU view bugs)
    from ...._creation import arange as _arange
    idx = _arange(0, 1, dtype=int64_dtype, device=te_flat.device)
    te_val = index_select(te_flat, 0, idx)
    te_tiled = dispatch("tile", "npu", te_val, (n,))
    result = eq(elem_flat, te_tiled)
    for i in range(1, te_len):
        idx_i = _arange(i, i + 1, dtype=int64_dtype, device=te_flat.device)
        te_val_i = index_select(te_flat, 0, idx_i)
        te_tiled_i = dispatch("tile", "npu", te_val_i, (n,))
        result = logical_or(result, eq(elem_flat, te_tiled_i))
    return view_backend.reshape(result, elem_shape)


def bucketize_op(a, boundaries, out_int32=False, right=False):
    """Maps values to bucket indices using boundaries (wrapper around searchsorted)."""
    return searchsorted(boundaries, a, out_int32=out_int32, right=right)


def diff_op(a, n=1, dim=-1, prepend=None, append=None):
    """Compute n-th discrete difference along dim."""
    from ...._dispatch.dispatcher import dispatch
    t = a
    if prepend is not None or append is not None:
        pieces = []
        if prepend is not None:
            pieces.append(prepend)
        pieces.append(t)
        if append is not None:
            pieces.append(append)
        t = dispatch("cat", "npu", pieces, dim=dim)
    ndim = len(t.shape)
    if dim < 0:
        dim = dim + ndim
    for _ in range(n):
        length = t.shape[dim]
        # Use index_select to create fresh tensors (narrow views have NPU offset bugs)
        from ...._creation import arange as _arange
        idx_hi = _arange(1, length, dtype=int64_dtype, device=t.device)
        idx_lo = _arange(0, length - 1, dtype=int64_dtype, device=t.device)
        s1 = index_select(t, dim, idx_hi)
        s0 = index_select(t, dim, idx_lo)
        t = sub(s1, s0)
    return t


def bincount_op(a, weights=None, minlength=0):
    """Count occurrences of each value in a 1-D int tensor."""
    from ...._dispatch.dispatcher import dispatch
    from ...common import view as view_backend
    flat = dispatch("flatten", "npu", a)
    n = flat.shape[0]
    if n == 0:
        length = max(0, minlength)
        return dispatch("zeros", "npu", (length,), dtype=float_dtype if weights is not None else int64_dtype, device=a.device)
    max_val = dispatch("amax", "npu", flat)
    # We need max_val as a Python int — sync to get value
    max_val_c = contiguous(max_val)
    import numpy as _np
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    max_np = _np.zeros(1, dtype=_np.int64)
    npu_runtime._memcpy_d2h(max_np.ctypes.data, max_np.nbytes, _unwrap_storage(max_val_c).data_ptr(), runtime=runtime)
    length = max(int(max_np[0]) + 1, minlength)
    out_dtype = weights.dtype if weights is not None else int64_dtype
    out = dispatch("zeros", "npu", (length,), dtype=out_dtype, device=a.device)
    if weights is not None:
        w_flat = dispatch("flatten", "npu", weights)
    else:
        w_flat = dispatch("ones", "npu", (n,), dtype=out_dtype, device=a.device)
    # Use scatter_add to accumulate
    idx = _cast_tensor_dtype(flat, int64_dtype)
    idx = view_backend.reshape(idx, (n,))
    from ...._functional import scatter_add_ as _scatter_add
    _scatter_add(out, 0, idx, w_flat)
    return out


def bincount_aclnn(a, weights=None, minlength=0):
    import numpy as _np
    from ...._dispatch.dispatcher import dispatch
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))

    # Get max value to determine output size (need sync)
    flat = dispatch("flatten", "npu", a)
    n = flat.shape[0]
    if n == 0:
        length = max(0, minlength)
        out_dt = weights.dtype if weights is not None else int64_dtype
        return dispatch("zeros", "npu", (length,), dtype=out_dt, device=a.device)

    max_val = dispatch("amax", "npu", flat)
    max_val_c = contiguous(max_val)
    max_np = _np.zeros(1, dtype=_np.int64)
    npu_runtime._memcpy_d2h(max_np.ctypes.data, max_np.nbytes, _unwrap_storage(max_val_c).data_ptr(), runtime=runtime)
    length = max(int(max_np[0]) + 1, minlength)

    out_dt = weights.dtype if weights is not None else int64_dtype
    out_shape = (length,)
    out_stride = (1,)
    out_nbytes = length * _dtype_itemsize(out_dt)
    out_ptr = npu_runtime._alloc_device(max(out_nbytes, 4), runtime=runtime)
    out_storage = npu_typed_storage_from_ptr(out_ptr, length, out_dt, device=a.device)

    s = _unwrap_storage(flat)
    w_ptr = None
    w_shape = None
    w_stride = None
    w_dtype = None
    if weights is not None:
        w_flat = dispatch("flatten", "npu", weights)
        w_s = _unwrap_storage(w_flat)
        w_ptr = w_s.data_ptr()
        w_shape = w_flat.shape
        w_stride = w_flat.stride
        w_dtype = w_flat.dtype

    aclnn.bincount(
        s.data_ptr(), w_ptr, out_ptr,
        flat.shape, flat.stride, out_shape, out_stride,
        flat.dtype, out_dt, minlength,
        weights_shape=w_shape, weights_stride=w_stride, weights_dtype=w_dtype,
        runtime=runtime, stream=stream.stream,
    )
    return _wrap_tensor(out_storage, out_shape, out_stride)


def histc_op(a, bins=100, min=0, max=0):
    """Histogram with equal-width bins."""
    import builtins
    builtins_abs = builtins.abs
    from ...._dispatch.dispatcher import dispatch
    from ...common import view as view_backend
    flat = dispatch("flatten", "npu", a)
    lo = float(min)
    hi = float(max)
    if lo == 0 and hi == 0:
        lo_t = dispatch("amin", "npu", flat)
        hi_t = dispatch("amax", "npu", flat)
        # Sync to get values
        import numpy as _np
        runtime = npu_runtime.get_runtime((a.device.index or 0))
        lo_np = _np.zeros(1, dtype=_np.float64)
        hi_np = _np.zeros(1, dtype=_np.float64)
        npu_runtime._memcpy_d2h(
            lo_np.ctypes.data, 4,
            _unwrap_storage(contiguous(_cast_tensor_dtype(lo_t, float_dtype))).data_ptr(),
            runtime=runtime
        )
        npu_runtime._memcpy_d2h(
            hi_np.ctypes.data, 4,
            _unwrap_storage(contiguous(_cast_tensor_dtype(hi_t, float_dtype))).data_ptr(),
            runtime=runtime
        )
        import struct
        lo = struct.unpack('f', lo_np[:4].tobytes())[0]
        hi = struct.unpack('f', hi_np[:4].tobytes())[0]
    # Compute bin edges and use searchsorted + bincount approach
    bin_width = (hi - lo) / bins
    if bin_width == 0:
        bin_width = 1.0
    # Clamp values to [lo, hi], compute bin indices
    clamped = dispatch("clamp", "npu", _cast_tensor_dtype(flat, float_dtype), lo, hi - 1e-7 * builtins_abs(hi - lo))
    lo_tensor = _scalar_to_npu_tensor(lo, clamped)
    shifted = sub(clamped, lo_tensor)
    bw_tensor = _scalar_to_npu_tensor(bin_width, clamped)
    indices = dispatch("floor", "npu", div(shifted, bw_tensor))
    indices = _cast_tensor_dtype(dispatch("clamp", "npu", indices, 0, bins - 1), int64_dtype)
    # Use scatter_add to count
    out = dispatch("zeros", "npu", (bins,), dtype=a.dtype, device=a.device)
    ones_t = dispatch("ones", "npu", (flat.shape[0],), dtype=a.dtype, device=a.device)
    from ...._functional import scatter_add_ as _scatter_add
    _scatter_add(out, 0, indices, ones_t)
    return out


def histogram_op(a, bins, range=None, weight=None, density=False):
    """Histogram returning (hist, bin_edges)."""
    from ...._dispatch.dispatcher import dispatch
    flat = dispatch("flatten", "npu", a)
    if isinstance(bins, int):
        nbins = bins
        if range is not None:
            lo, hi = float(range[0]), float(range[1])
        else:
            lo_t = dispatch("amin", "npu", _cast_tensor_dtype(flat, float_dtype))
            hi_t = dispatch("amax", "npu", _cast_tensor_dtype(flat, float_dtype))
            import numpy as _np
            runtime = npu_runtime.get_runtime((a.device.index or 0))
            lo_np = _np.zeros(1, dtype=_np.float32)
            hi_np = _np.zeros(1, dtype=_np.float32)
            npu_runtime._memcpy_d2h(
                lo_np.ctypes.data, 4, _unwrap_storage(contiguous(lo_t)).data_ptr(), runtime=runtime
            )
            npu_runtime._memcpy_d2h(
                hi_np.ctypes.data, 4, _unwrap_storage(contiguous(hi_t)).data_ptr(), runtime=runtime
            )
            lo, hi = float(lo_np[0]), float(hi_np[0])
        import numpy as _np
        edges_np = _np.linspace(lo, hi, nbins + 1, dtype=_np.float32)
    else:
        # bins is a tensor of edges
        edges_flat = dispatch("flatten", "npu", bins)
        nbins = edges_flat.shape[0] - 1
        # For simplicity, sync edges to CPU
        import numpy as _np
        runtime = npu_runtime.get_runtime((a.device.index or 0))
        edges_np = _np.zeros(edges_flat.shape[0], dtype=_np.float32)
        npu_runtime._memcpy_d2h(
            edges_np.ctypes.data, edges_np.nbytes,
            _unwrap_storage(contiguous(_cast_tensor_dtype(edges_flat, float_dtype))).data_ptr(),
            runtime=runtime
        )
    # Compute bin indices via searchsorted
    from ...._creation import tensor as create_tensor
    edges_tensor = create_tensor(edges_np.tolist(), dtype=a.dtype, device=a.device)
    indices = searchsorted(edges_tensor, _cast_tensor_dtype(flat, a.dtype), right=False)
    # Clamp to valid range [1, nbins] then shift to [0, nbins-1]
    one_t = _scalar_to_npu_tensor(1, indices)
    nbins_t = _scalar_to_npu_tensor(nbins, indices)
    indices = dispatch("clamp", "npu", indices, one_t, nbins_t)
    indices = sub(indices, one_t)
    indices = _cast_tensor_dtype(indices, int64_dtype)
    # Accumulate
    if weight is not None:
        w_flat = dispatch("flatten", "npu", weight)
        hist = dispatch("zeros", "npu", (nbins,), dtype=weight.dtype, device=a.device)
    else:
        w_flat = dispatch("ones", "npu", (flat.shape[0],), dtype=a.dtype, device=a.device)
        hist = dispatch("zeros", "npu", (nbins,), dtype=a.dtype, device=a.device)
    from ...._functional import scatter_add_ as _scatter_add
    _scatter_add(hist, 0, indices, w_flat)
    edges_out = create_tensor(edges_np.tolist(), dtype=a.dtype, device=a.device)
    return hist, edges_out
