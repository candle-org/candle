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

def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    return np.allclose(
        _to_numpy(a),
        _to_numpy(b),
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
    )

def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    out = np.isclose(
        _to_numpy(a),
        _to_numpy(b),
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
    )
    return _from_numpy(out, bool_dtype, a.device)

def equal(a, b):
    return np.array_equal(_to_numpy(a), _to_numpy(b))

def eq(a, b):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, bool_dtype)
        if isinstance(b, Tensor) and _can_use_gpu(b) and a.shape == b.shape and a.is_contiguous() and b.is_contiguous():
            d.dispatch_comparison(f"eq_{sfx}", _metal_buf(a), _metal_buf(b),
                                  out_buf, numel)
        elif not isinstance(b, Tensor) or not _can_use_gpu(b):
            scalar = _scalar_value(float(b) if not isinstance(b, Tensor) else float(_to_numpy(b).ravel()[0]), a.dtype)
            if a.is_contiguous():
                d.dispatch_comparison_scalar(f"eq_scalar_{sfx}", _metal_buf(a),
                                             scalar, out_buf, numel,
                                             scalar_fmt=_scalar_fmt(a.dtype))
            else:
                return _from_numpy(np.equal(_to_numpy(a), scalar), bool_dtype, a.device)
        else:
            return _from_numpy(np.equal(_to_numpy(a), _to_numpy(b)), bool_dtype, a.device)
        from ...._tensor import _compute_strides
        return _from_metal_buffer(out_buf, a.shape, _compute_strides(a.shape), bool_dtype, a.device)
    b_np = _to_numpy(b) if isinstance(b, Tensor) else b
    return _from_numpy(np.equal(_to_numpy(a), b_np), bool_dtype, a.device)

def ne(a, b):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, bool_dtype)
        if isinstance(b, Tensor) and _can_use_gpu(b) and a.shape == b.shape and a.is_contiguous() and b.is_contiguous():
            d.dispatch_comparison(f"ne_{sfx}", _metal_buf(a), _metal_buf(b),
                                  out_buf, numel)
        elif not isinstance(b, Tensor) or not _can_use_gpu(b):
            scalar = _scalar_value(float(b) if not isinstance(b, Tensor) else float(_to_numpy(b).ravel()[0]), a.dtype)
            if a.is_contiguous():
                d.dispatch_comparison_scalar(f"ne_scalar_{sfx}", _metal_buf(a),
                                             scalar, out_buf, numel,
                                             scalar_fmt=_scalar_fmt(a.dtype))
            else:
                return _from_numpy(np.not_equal(_to_numpy(a), scalar), bool_dtype, a.device)
        else:
            return _from_numpy(np.not_equal(_to_numpy(a), _to_numpy(b)), bool_dtype, a.device)
        from ...._tensor import _compute_strides
        return _from_metal_buffer(out_buf, a.shape, _compute_strides(a.shape), bool_dtype, a.device)
    b_np = _to_numpy(b) if isinstance(b, Tensor) else b
    return _from_numpy(np.not_equal(_to_numpy(a), b_np), bool_dtype, a.device)

def lt(a, b):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, bool_dtype)
        if isinstance(b, Tensor) and _can_use_gpu(b) and a.shape == b.shape and a.is_contiguous() and b.is_contiguous():
            d.dispatch_comparison(f"lt_{sfx}", _metal_buf(a), _metal_buf(b),
                                  out_buf, numel)
        elif not isinstance(b, Tensor) or not _can_use_gpu(b):
            scalar = float(b) if not isinstance(b, Tensor) else float(_to_numpy(b).ravel()[0])
            if a.is_contiguous():
                d.dispatch_comparison_scalar(f"lt_scalar_{sfx}", _metal_buf(a),
                                             scalar, out_buf, numel,
                                             scalar_fmt=_scalar_fmt(a.dtype))
            else:
                return _from_numpy(np.less(_to_numpy(a), scalar), bool_dtype, a.device)
        else:
            return _from_numpy(np.less(_to_numpy(a), _to_numpy(b)), bool_dtype, a.device)
        from ...._tensor import _compute_strides
        return _from_metal_buffer(out_buf, a.shape, _compute_strides(a.shape), bool_dtype, a.device)
    b_np = _to_numpy(b) if isinstance(b, Tensor) else b
    return _from_numpy(np.less(_to_numpy(a), b_np), bool_dtype, a.device)

def le(a, b):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, bool_dtype)
        if isinstance(b, Tensor) and _can_use_gpu(b) and a.shape == b.shape and a.is_contiguous() and b.is_contiguous():
            d.dispatch_comparison(f"le_{sfx}", _metal_buf(a), _metal_buf(b),
                                  out_buf, numel)
        elif not isinstance(b, Tensor) or not _can_use_gpu(b):
            scalar = float(b) if not isinstance(b, Tensor) else float(_to_numpy(b).ravel()[0])
            if a.is_contiguous():
                d.dispatch_comparison_scalar(f"le_scalar_{sfx}", _metal_buf(a),
                                             scalar, out_buf, numel,
                                             scalar_fmt=_scalar_fmt(a.dtype))
            else:
                return _from_numpy(np.less_equal(_to_numpy(a), scalar), bool_dtype, a.device)
        else:
            return _from_numpy(np.less_equal(_to_numpy(a), _to_numpy(b)), bool_dtype, a.device)
        from ...._tensor import _compute_strides
        return _from_metal_buffer(out_buf, a.shape, _compute_strides(a.shape), bool_dtype, a.device)
    b_np = _to_numpy(b) if isinstance(b, Tensor) else b
    return _from_numpy(np.less_equal(_to_numpy(a), b_np), bool_dtype, a.device)

def gt(a, b):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, bool_dtype)
        if isinstance(b, Tensor) and _can_use_gpu(b) and a.shape == b.shape and a.is_contiguous() and b.is_contiguous():
            d.dispatch_comparison(f"gt_{sfx}", _metal_buf(a), _metal_buf(b),
                                  out_buf, numel)
        elif not isinstance(b, Tensor) or not _can_use_gpu(b):
            scalar = float(b) if not isinstance(b, Tensor) else float(_to_numpy(b).ravel()[0])
            if a.is_contiguous():
                d.dispatch_comparison_scalar(f"gt_scalar_{sfx}", _metal_buf(a),
                                             scalar, out_buf, numel,
                                             scalar_fmt=_scalar_fmt(a.dtype))
            else:
                return _from_numpy(np.greater(_to_numpy(a), scalar), bool_dtype, a.device)
        else:
            return _from_numpy(np.greater(_to_numpy(a), _to_numpy(b)), bool_dtype, a.device)
        from ...._tensor import _compute_strides
        return _from_metal_buffer(out_buf, a.shape, _compute_strides(a.shape), bool_dtype, a.device)
    b_np = _to_numpy(b) if isinstance(b, Tensor) else b
    return _from_numpy(np.greater(_to_numpy(a), b_np), bool_dtype, a.device)

def ge(a, b):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, bool_dtype)
        if isinstance(b, Tensor) and _can_use_gpu(b) and a.shape == b.shape and a.is_contiguous() and b.is_contiguous():
            d.dispatch_comparison(f"ge_{sfx}", _metal_buf(a), _metal_buf(b),
                                  out_buf, numel)
        elif not isinstance(b, Tensor) or not _can_use_gpu(b):
            scalar = float(b) if not isinstance(b, Tensor) else float(_to_numpy(b).ravel()[0])
            if a.is_contiguous():
                d.dispatch_comparison_scalar(f"ge_scalar_{sfx}", _metal_buf(a),
                                             scalar, out_buf, numel,
                                             scalar_fmt=_scalar_fmt(a.dtype))
            else:
                return _from_numpy(np.greater_equal(_to_numpy(a), scalar), bool_dtype, a.device)
        else:
            return _from_numpy(np.greater_equal(_to_numpy(a), _to_numpy(b)), bool_dtype, a.device)
        from ...._tensor import _compute_strides
        return _from_metal_buffer(out_buf, a.shape, _compute_strides(a.shape), bool_dtype, a.device)
    b_np = _to_numpy(b) if isinstance(b, Tensor) else b
    return _from_numpy(np.greater_equal(_to_numpy(a), b_np), bool_dtype, a.device)

def logical_and(a, b):
    if (_can_use_gpu(a) and a.is_contiguous()
            and isinstance(b, Tensor) and _can_use_gpu(b) and b.is_contiguous()):
        # ne(a,0) returns bool(uint8), ne(b,0) returns bool(uint8), mul gives AND
        a_bool = ne(a, 0)
        b_bool = ne(b, 0)
        return _dispatch_binary_gpu(a_bool, b_bool, "mul")
    a_np = _to_numpy(a)
    b_np = _to_numpy(b) if isinstance(b, Tensor) else b
    return _from_numpy(np.logical_and(a_np, b_np), bool_dtype, a.device)

def logical_or(a, b):
    if (_can_use_gpu(a) and a.is_contiguous()
            and isinstance(b, Tensor) and _can_use_gpu(b) and b.is_contiguous()):
        # ne(a,0) | ne(b,0) = ne(a_bool + b_bool, 0)
        a_bool = ne(a, 0)
        b_bool = ne(b, 0)
        sum_buf = _dispatch_binary_gpu(a_bool, b_bool, "add")
        return ne(sum_buf, 0)
    a_np = _to_numpy(a)
    b_np = _to_numpy(b) if isinstance(b, Tensor) else b
    return _from_numpy(np.logical_or(a_np, b_np), bool_dtype, a.device)

def logical_not(a):
    if _can_use_gpu(a) and a.is_contiguous():
        return eq(a, 0)
    return _from_numpy(np.logical_not(_to_numpy(a)), bool_dtype, a.device)

def logical_xor(a, b):
    if (_can_use_gpu(a) and a.is_contiguous()
            and isinstance(b, Tensor) and _can_use_gpu(b) and b.is_contiguous()):
        a_bool = ne(a, 0)
        b_bool = ne(b, 0)
        return ne(a_bool, b_bool)
    a_np = _to_numpy(a).astype(bool)
    b_np = (_to_numpy(b) if isinstance(b, Tensor) else np.array(b)).astype(bool)
    return _from_numpy(np.logical_xor(a_np, b_np), bool_dtype, a.device)


# ---------------------------------------------------------------------------
# Group 3: Bitwise ops
# ---------------------------------------------------------------------------

def bitwise_and(a, b):
    a_np = _to_numpy(a)
    b_np = _to_numpy(b) if isinstance(b, Tensor) else b
    return _from_numpy(np.bitwise_and(a_np, b_np), a.dtype, a.device)

def bitwise_or(a, b):
    a_np = _to_numpy(a)
    b_np = _to_numpy(b) if isinstance(b, Tensor) else b
    return _from_numpy(np.bitwise_or(a_np, b_np), a.dtype, a.device)

def bitwise_xor(a, b):
    a_np = _to_numpy(a)
    b_np = _to_numpy(b) if isinstance(b, Tensor) else b
    return _from_numpy(np.bitwise_xor(a_np, b_np), a.dtype, a.device)

def bitwise_not(a):
    return _from_numpy(np.bitwise_not(_to_numpy(a)), a.dtype, a.device)


# ---------------------------------------------------------------------------
# Group 4: Random in-place op
# ---------------------------------------------------------------------------

