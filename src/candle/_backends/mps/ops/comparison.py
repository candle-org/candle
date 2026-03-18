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
    _to_numpy, _from_numpy, _read_scalar,
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
    # GPU composite: all(isclose(a, b))
    close = isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
    # all_ returns a scalar tensor; extract Python bool
    from .reduce import all_
    return bool(all_(close).item())

def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    # GPU composite: |a - b| <= atol + rtol * |b|
    if _can_use_gpu(a) and isinstance(b, Tensor) and _can_use_gpu(b):
        from .math import abs as _abs, sub, mul, add
        diff = _abs(sub(a, b))
        tol = add(_dispatch_binary_gpu(_abs(b), float(rtol), "mul"), float(atol))
        close = le(diff, tol)
        if equal_nan:
            from .math import isnan
            both_nan = logical_and(isnan(a), isnan(b))
            close = logical_or(close, both_nan)
        return close
    _unsupported_dtype("isclose", a)

def equal(a, b):
    # GPU composite: all(eq(a, b))
    if _can_use_gpu(a) and isinstance(b, Tensor) and _can_use_gpu(b):
        if a.shape != b.shape:
            return False
        from .reduce import all_
        return bool(all_(eq(a, b)).item())
    _unsupported_dtype("equal", a)

def eq(a, b):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, bool_dtype)
        if isinstance(b, Tensor) and _can_use_gpu(b) and a.shape == b.shape and a.is_contiguous() and b.is_contiguous():
            d.dispatch_comparison(f"eq_{sfx}", _metal_buf(a), _metal_buf(b),
                                  out_buf, numel)
        elif not isinstance(b, Tensor) or not _can_use_gpu(b) or a.shape != b.shape:
            scalar = _scalar_value(float(b) if not isinstance(b, Tensor) else _read_scalar(b), a.dtype)
            if a.is_contiguous():
                d.dispatch_comparison_scalar(f"eq_scalar_{sfx}", _metal_buf(a),
                                             scalar, out_buf, numel,
                                             scalar_fmt=_scalar_fmt(a.dtype))
            else:
                return eq(a.contiguous(), b)
        else:
            return eq(a.contiguous(), b.contiguous() if isinstance(b, Tensor) else b)
        from ...._tensor import _compute_strides
        return _from_metal_buffer(out_buf, a.shape, _compute_strides(a.shape), bool_dtype, a.device)
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("eq", a)

def ne(a, b):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, bool_dtype)
        if isinstance(b, Tensor) and _can_use_gpu(b) and a.shape == b.shape and a.is_contiguous() and b.is_contiguous():
            d.dispatch_comparison(f"ne_{sfx}", _metal_buf(a), _metal_buf(b),
                                  out_buf, numel)
        elif not isinstance(b, Tensor) or not _can_use_gpu(b) or a.shape != b.shape:
            scalar = _scalar_value(float(b) if not isinstance(b, Tensor) else _read_scalar(b), a.dtype)
            if a.is_contiguous():
                d.dispatch_comparison_scalar(f"ne_scalar_{sfx}", _metal_buf(a),
                                             scalar, out_buf, numel,
                                             scalar_fmt=_scalar_fmt(a.dtype))
            else:
                return ne(a.contiguous(), b)
        else:
            return ne(a.contiguous(), b.contiguous() if isinstance(b, Tensor) else b)
        from ...._tensor import _compute_strides
        return _from_metal_buffer(out_buf, a.shape, _compute_strides(a.shape), bool_dtype, a.device)
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("ne", a)

def lt(a, b):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, bool_dtype)
        if isinstance(b, Tensor) and _can_use_gpu(b) and a.shape == b.shape and a.is_contiguous() and b.is_contiguous():
            d.dispatch_comparison(f"lt_{sfx}", _metal_buf(a), _metal_buf(b),
                                  out_buf, numel)
        elif not isinstance(b, Tensor) or not _can_use_gpu(b) or a.shape != b.shape:
            scalar = float(b) if not isinstance(b, Tensor) else _read_scalar(b)
            if a.is_contiguous():
                d.dispatch_comparison_scalar(f"lt_scalar_{sfx}", _metal_buf(a),
                                             scalar, out_buf, numel,
                                             scalar_fmt=_scalar_fmt(a.dtype))
            else:
                return lt(a.contiguous(), b)
        else:
            return lt(a.contiguous(), b.contiguous() if isinstance(b, Tensor) else b)
        from ...._tensor import _compute_strides
        return _from_metal_buffer(out_buf, a.shape, _compute_strides(a.shape), bool_dtype, a.device)
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("lt", a)

def le(a, b):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, bool_dtype)
        if isinstance(b, Tensor) and _can_use_gpu(b) and a.shape == b.shape and a.is_contiguous() and b.is_contiguous():
            d.dispatch_comparison(f"le_{sfx}", _metal_buf(a), _metal_buf(b),
                                  out_buf, numel)
        elif not isinstance(b, Tensor) or not _can_use_gpu(b) or a.shape != b.shape:
            scalar = float(b) if not isinstance(b, Tensor) else _read_scalar(b)
            if a.is_contiguous():
                d.dispatch_comparison_scalar(f"le_scalar_{sfx}", _metal_buf(a),
                                             scalar, out_buf, numel,
                                             scalar_fmt=_scalar_fmt(a.dtype))
            else:
                return le(a.contiguous(), b)
        else:
            return le(a.contiguous(), b.contiguous() if isinstance(b, Tensor) else b)
        from ...._tensor import _compute_strides
        return _from_metal_buffer(out_buf, a.shape, _compute_strides(a.shape), bool_dtype, a.device)
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("le", a)

def gt(a, b):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, bool_dtype)
        if isinstance(b, Tensor) and _can_use_gpu(b) and a.shape == b.shape and a.is_contiguous() and b.is_contiguous():
            d.dispatch_comparison(f"gt_{sfx}", _metal_buf(a), _metal_buf(b),
                                  out_buf, numel)
        elif not isinstance(b, Tensor) or not _can_use_gpu(b) or a.shape != b.shape:
            scalar = float(b) if not isinstance(b, Tensor) else _read_scalar(b)
            if a.is_contiguous():
                d.dispatch_comparison_scalar(f"gt_scalar_{sfx}", _metal_buf(a),
                                             scalar, out_buf, numel,
                                             scalar_fmt=_scalar_fmt(a.dtype))
            else:
                return gt(a.contiguous(), b)
        else:
            return gt(a.contiguous(), b.contiguous() if isinstance(b, Tensor) else b)
        from ...._tensor import _compute_strides
        return _from_metal_buffer(out_buf, a.shape, _compute_strides(a.shape), bool_dtype, a.device)
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("gt", a)

def ge(a, b):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, bool_dtype)
        if isinstance(b, Tensor) and _can_use_gpu(b) and a.shape == b.shape and a.is_contiguous() and b.is_contiguous():
            d.dispatch_comparison(f"ge_{sfx}", _metal_buf(a), _metal_buf(b),
                                  out_buf, numel)
        elif not isinstance(b, Tensor) or not _can_use_gpu(b) or a.shape != b.shape:
            scalar = float(b) if not isinstance(b, Tensor) else _read_scalar(b)
            if a.is_contiguous():
                d.dispatch_comparison_scalar(f"ge_scalar_{sfx}", _metal_buf(a),
                                             scalar, out_buf, numel,
                                             scalar_fmt=_scalar_fmt(a.dtype))
            else:
                return ge(a.contiguous(), b)
        else:
            return ge(a.contiguous(), b.contiguous() if isinstance(b, Tensor) else b)
        from ...._tensor import _compute_strides
        return _from_metal_buffer(out_buf, a.shape, _compute_strides(a.shape), bool_dtype, a.device)
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("ge", a)

def logical_and(a, b):
    if (_can_use_gpu(a)
            and isinstance(b, Tensor) and _can_use_gpu(b)):
        a_bool = ne(a, 0)
        b_bool = ne(b, 0)
        return _dispatch_binary_gpu(a_bool, b_bool, "mul")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("logical_and", a)

def logical_or(a, b):
    if (_can_use_gpu(a)
            and isinstance(b, Tensor) and _can_use_gpu(b)):
        a_bool = ne(a, 0)
        b_bool = ne(b, 0)
        sum_buf = _dispatch_binary_gpu(a_bool, b_bool, "add")
        return ne(sum_buf, 0)
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("logical_or", a)

def logical_not(a):
    if _can_use_gpu(a):
        return eq(a, 0)
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("logical_not", a)

def logical_xor(a, b):
    if (_can_use_gpu(a)
            and isinstance(b, Tensor) and _can_use_gpu(b)):
        a_bool = ne(a, 0)
        b_bool = ne(b, 0)
        return ne(a_bool, b_bool)
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("logical_xor", a)


# ---------------------------------------------------------------------------
# Group 3: Bitwise ops
# ---------------------------------------------------------------------------

def bitwise_and(a, b):
    if _can_use_gpu(a):
        return _dispatch_binary_gpu(a, b, "bitwise_and")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("bitwise_and", a)

def bitwise_or(a, b):
    if _can_use_gpu(a):
        return _dispatch_binary_gpu(a, b, "bitwise_or")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("bitwise_or", a)

def bitwise_xor(a, b):
    if _can_use_gpu(a):
        return _dispatch_binary_gpu(a, b, "bitwise_xor")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("bitwise_xor", a)

def bitwise_not(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "bitwise_not")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("bitwise_not", a)

def bitwise_left_shift(a, b):
    if _can_use_gpu(a):
        return _dispatch_binary_gpu(a, b, "bitwise_left_shift")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("bitwise_left_shift", a)

def bitwise_right_shift(a, b):
    if _can_use_gpu(a):
        return _dispatch_binary_gpu(a, b, "bitwise_right_shift")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("bitwise_right_shift", a)


# ---------------------------------------------------------------------------
# Group 4: Random in-place op
# ---------------------------------------------------------------------------

