# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython op entry points for MPS math and activation ops."""

import numpy as np
import ctypes
import struct

from candle._backends.mps.ops.shape import expand

from candle._backends.mps.ops._helpers import (
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


# ---------------------------------------------------------------------------
# math.py functions
# ---------------------------------------------------------------------------

cpdef object add(object a, object b):
    if isinstance(b, Tensor) and _can_use_gpu(b):
        if not _can_use_gpu(a) or a.numel() < b.numel():
            a, b = b, a
    if _can_use_gpu(a):
        return _dispatch_binary_gpu(a, b, "add")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("add", a)


cpdef object mul(object a, object b):
    if isinstance(b, Tensor) and _can_use_gpu(b):
        if not _can_use_gpu(a) or a.numel() < b.numel():
            a, b = b, a
    if _can_use_gpu(a):
        return _dispatch_binary_gpu(a, b, "mul")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("mul", a)


cpdef object div(object a, object b):
    if _can_use_gpu(a):
        return _dispatch_binary_gpu(a, b, "div")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("div", a)


cpdef object true_divide(object a, object b):
    return div(a, b)


cpdef object sub(object a, object b):
    if _can_use_gpu(a):
        return _dispatch_binary_gpu(a, b, "sub")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("sub", a)


cpdef object abs(object a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "abs")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("abs", a)


cpdef object neg(object a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "neg")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("neg", a)


cpdef object sqrt(object a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "sqrt")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("sqrt", a)


cpdef object exp(object a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "exp")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("exp", a)


cpdef object log(object a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "log")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("log", a)


cpdef object sin(object a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "sin")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("sin", a)


cpdef object cos(object a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "cos")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("cos", a)


cpdef object tan(object a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "tan")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("tan", a)


cpdef object tanh(object a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "tanh")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("tanh", a)


cpdef object sigmoid(object a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "sigmoid")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("sigmoid", a)


cpdef object floor(object a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "floor")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("floor", a)


cpdef object ceil(object a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "ceil")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("ceil", a)


cpdef object round(object a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "round")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("round", a)


cpdef object trunc(object a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "trunc")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("trunc", a)


cpdef object frac(object a):
    if _can_use_gpu(a):
        return sub(a, _dispatch_unary_gpu(a, "trunc"))
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("frac", a)


cpdef object pow(object a, object b):
    if isinstance(a, Tensor) and _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, a.dtype)
        if isinstance(b, Tensor) and _can_use_gpu(b):
            d.dispatch_binary(f"pow_{sfx}", _metal_buf(a), _metal_buf(b),
                              out_buf, numel)
        else:
            scalar = float(b) if not isinstance(b, Tensor) else float(b.item())
            d.dispatch_binary_scalar(f"pow_scalar_{sfx}", _metal_buf(a),
                                     scalar, out_buf, numel,
                                     scalar_fmt=_scalar_fmt(a.dtype))
        return _from_metal_buffer(out_buf, a.shape, a.stride, a.dtype, a.device)
    ref = a if isinstance(a, Tensor) else b
    if ref.numel() == 0:
        return _empty_like(ref)
    _unsupported_dtype("pow", ref)


cpdef object log2(object a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "log2")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("log2", a)


cpdef object log10(object a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "log10")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("log10", a)


cpdef object exp2(object a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "exp2")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("exp2", a)


cpdef object rsqrt(object a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "rsqrt")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("rsqrt", a)


cpdef object sign(object a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "sign")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("sign", a)


cpdef object square(object a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "square")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("square", a)


cpdef object signbit(object a):
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        return _dispatch_unary_predicate_gpu(a, "signbit")
    if _can_use_gpu(a) and a.dtype in (int32_dtype, int64_dtype):
        from candle._backends.mps.ops.comparison import lt
        return lt(a, 0)
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("signbit", a)


cpdef object isnan(object a):
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        return _dispatch_unary_predicate_gpu(a, "isnan")
    if _can_use_gpu(a) and a.dtype in (int32_dtype, int64_dtype, bool_dtype):
        from candle._backends.mps.ops.comparison import ne
        return ne(a, a)
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("isnan", a)


cpdef object isinf(object a):
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        return _dispatch_unary_predicate_gpu(a, "isinf")
    if _can_use_gpu(a) and a.dtype in (int32_dtype, int64_dtype, bool_dtype):
        from candle._backends.mps.ops.comparison import ne
        return ne(a, a)
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("isinf", a)


cpdef object isfinite(object a):
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        return _dispatch_unary_predicate_gpu(a, "isfinite")
    if _can_use_gpu(a) and a.dtype in (int32_dtype, int64_dtype, bool_dtype):
        from candle._backends.mps.ops.comparison import eq
        return eq(a, a)
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("isfinite", a)


cpdef object isneginf(object a):
    """Returns a bool tensor indicating negative infinity."""
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        return _dispatch_unary_predicate_gpu(a, "isneginf")
    if _can_use_gpu(a) and a.dtype in (int32_dtype, int64_dtype, bool_dtype):
        from candle._backends.mps.ops.comparison import ne
        return ne(a, a)
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("isneginf", a)


cpdef object isposinf(object a):
    """Returns a bool tensor indicating positive infinity."""
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        return _dispatch_unary_predicate_gpu(a, "isposinf")
    if _can_use_gpu(a) and a.dtype in (int32_dtype, int64_dtype, bool_dtype):
        from candle._backends.mps.ops.comparison import ne
        return ne(a, a)
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("isposinf", a)


cpdef object isreal(object a):
    """Returns a bool tensor indicating real-valued elements."""
    if _can_use_gpu(a):
        from candle._backends.mps.ops.comparison import eq
        return eq(a, a)
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("isreal", a)


cpdef object sinh(object a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "sinh")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("sinh", a)


cpdef object cosh(object a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "cosh")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("cosh", a)


cpdef object asinh(object a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "asinh")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("asinh", a)


cpdef object acosh(object a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "acosh")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("acosh", a)


cpdef object atanh(object a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "atanh")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("atanh", a)


cpdef object erf(object a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "erf")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("erf", a)


cpdef object erfc(object a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "erfc")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("erfc", a)


cpdef object atan(object a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "atan")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("atan", a)


cpdef object atan2(object a, object b):
    if isinstance(a, Tensor) and isinstance(b, Tensor) and _can_use_gpu(a) and _can_use_gpu(b):
        return _dispatch_binary_gpu(a, b, "atan2")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("atan2", a)


cpdef object asin(object a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "asin")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("asin", a)


cpdef object acos(object a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "acos")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("acos", a)


cpdef object floor_divide(object a, object b):
    if _can_use_gpu(a):
        return _dispatch_binary_gpu(a, b, "floor_divide")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("floor_divide", a)


cpdef object log1p(object a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "log1p")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("log1p", a)


cpdef object expm1(object a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "expm1")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("expm1", a)


cpdef object reciprocal(object a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "reciprocal")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("reciprocal", a)


# ---------------------------------------------------------------------------
# activation.py functions
# ---------------------------------------------------------------------------

cpdef object relu(object a):
    if a.numel() == 0:
        return _empty_like(a)
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "relu")
    _unsupported_dtype("relu", a)


cpdef object gelu(object a):
    if a.numel() == 0:
        return _empty_like(a)
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "gelu")
    _unsupported_dtype("gelu", a)


cpdef object softplus(object a):
    if a.numel() == 0:
        return _empty_like(a)
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        exp_a = _dispatch_unary_gpu(a, "exp")
        sum_val = add(exp_a, 1.0)
        return _dispatch_unary_gpu(sum_val, "log")
    _unsupported_dtype("softplus", a)


cpdef object silu(object a):
    if a.numel() == 0:
        return _empty_like(a)
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "silu")
    _unsupported_dtype("silu", a)


cpdef object leaky_relu(object a, object negative_slope=0.01):
    if a.numel() == 0:
        return _empty_like(a)
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, a.dtype)
        scalar = float(negative_slope)
        if a.is_contiguous():
            d.dispatch_binary_scalar(f"leaky_relu_scalar_{sfx}", _metal_buf(a),
                                     scalar, out_buf, numel,
                                     scalar_fmt=_scalar_fmt(a.dtype))
        else:
            d.dispatch_binary_scalar_strided(
                f"leaky_relu_scalar_strided_{sfx}", _metal_buf(a), scalar,
                out_buf, numel, list(a.shape), list(a.stride),
                len(a.shape), scalar_fmt=_scalar_fmt(a.dtype))
        from candle._tensor import _compute_strides
        return _from_metal_buffer(out_buf, a.shape, _compute_strides(a.shape),
                                  a.dtype, a.device)
    _unsupported_dtype("leaky_relu", a)


cpdef object elu(object a, object alpha=1.0):
    if a.numel() == 0:
        return _empty_like(a)
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        relu_a = _dispatch_unary_gpu(a, "relu")
        exp_a = _dispatch_unary_gpu(a, "exp")
        elu_part = mul(sub(exp_a, 1.0), alpha)
        neg_part = clamp(elu_part, None, 0.0)
        return add(relu_a, neg_part)
    _unsupported_dtype("elu", a)


cpdef object mish(object a):
    if a.numel() == 0:
        return _empty_like(a)
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        sp = softplus(a)
        return mul(a, _dispatch_unary_gpu(sp, "tanh"))
    _unsupported_dtype("mish", a)


cpdef object prelu(object a, object weight):
    if a.numel() == 0:
        return _empty_like(a)
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        a_c = a.contiguous() if not a.is_contiguous() else a
        from candle._backends.mps.ops.comparison import gt
        mask = gt(a_c, 0)
        from candle._backends.mps.ops.shape import expand
        w_expanded = expand(weight, tuple(a_c.shape))
        from candle._backends.mps.ops.elementwise import where
        neg = mul(a_c, w_expanded)
        return where(mask, a_c, neg)
    _unsupported_dtype("prelu", a)


cpdef object clamp(object a, object min_val=None, object max_val=None):
    if a.numel() == 0:
        return _empty_like(a)
    if min_val is not None and max_val is not None:
        if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype, int32_dtype, int64_dtype):
            d = _get_dispatcher()
            sfx = _kernel_suffix(a.dtype)
            numel = a.numel()
            out_buf = _alloc_output_buf(numel, a.dtype)
            s_min = _scalar_value(min_val, a.dtype)
            s_max = _scalar_value(max_val, a.dtype)
            fmt = _scalar_fmt(a.dtype)
            if a.is_contiguous():
                d.dispatch_clamp(f"clamp_{sfx}", _metal_buf(a),
                                 s_min, s_max, out_buf, numel,
                                 scalar_fmt=fmt)
            else:
                d.dispatch_clamp_strided(
                    f"clamp_strided_{sfx}", _metal_buf(a),
                    s_min, s_max, out_buf, numel,
                    list(a.shape), list(a.stride), len(a.shape),
                    scalar_fmt=fmt)
            from candle._tensor import _compute_strides
            return _from_metal_buffer(out_buf, a.shape,
                                      _compute_strides(a.shape),
                                      a.dtype, a.device)
        _unsupported_dtype("clamp", a)
    if min_val is not None:
        return clamp_min(a, min_val)
    if max_val is not None:
        return clamp_max(a, max_val)
    return a


cpdef object clamp_min(object a, object min_val):
    if a.numel() == 0:
        return _empty_like(a)
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype, int32_dtype, int64_dtype):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, a.dtype)
        scalar = _scalar_value(min_val, a.dtype)
        if a.is_contiguous():
            d.dispatch_binary_scalar(f"clamp_min_scalar_{sfx}", _metal_buf(a),
                                     scalar, out_buf, numel,
                                     scalar_fmt=_scalar_fmt(a.dtype))
        else:
            d.dispatch_binary_scalar_strided(
                f"clamp_min_scalar_strided_{sfx}", _metal_buf(a), scalar,
                out_buf, numel, list(a.shape), list(a.stride),
                len(a.shape), scalar_fmt=_scalar_fmt(a.dtype))
        from candle._tensor import _compute_strides
        return _from_metal_buffer(out_buf, a.shape, _compute_strides(a.shape),
                                  a.dtype, a.device)
    _unsupported_dtype("clamp_min", a)


cpdef object clamp_max(object a, object max_val):
    if a.numel() == 0:
        return _empty_like(a)
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype, int32_dtype, int64_dtype):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, a.dtype)
        scalar = _scalar_value(max_val, a.dtype)
        if a.is_contiguous():
            d.dispatch_binary_scalar(f"clamp_max_scalar_{sfx}", _metal_buf(a),
                                     scalar, out_buf, numel,
                                     scalar_fmt=_scalar_fmt(a.dtype))
        else:
            d.dispatch_binary_scalar_strided(
                f"clamp_max_scalar_strided_{sfx}", _metal_buf(a), scalar,
                out_buf, numel, list(a.shape), list(a.stride),
                len(a.shape), scalar_fmt=_scalar_fmt(a.dtype))
        from candle._tensor import _compute_strides
        return _from_metal_buffer(out_buf, a.shape, _compute_strides(a.shape),
                                  a.dtype, a.device)
    _unsupported_dtype("clamp_max", a)


cpdef object relu6(object a):
    if a.numel() == 0:
        return _empty_like(a)
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype, int32_dtype, int64_dtype):
        return clamp(a, 0.0, 6.0)
    _unsupported_dtype("relu6", a)


cpdef object hardtanh(object a, object min_val=-1.0, object max_val=1.0):
    if a.numel() == 0:
        return _empty_like(a)
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype, int32_dtype, int64_dtype):
        return clamp(a, min_val, max_val)
    _unsupported_dtype("hardtanh", a)


cpdef object selu(object a):
    ALPHA = 1.6732632423543772
    SCALE = 1.0507009873554805
    if a.numel() == 0:
        return _empty_like(a)
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        exp_a = _dispatch_unary_gpu(a, "exp")
        elu_part = mul(sub(exp_a, 1.0), ALPHA)
        relu_a = _dispatch_unary_gpu(a, "relu")
        neg_part = clamp(elu_part, None, 0.0)
        result = add(relu_a, neg_part)
        return mul(result, SCALE)
    _unsupported_dtype("selu", a)


cpdef object celu(object a, object alpha=1.0):
    if a.numel() == 0:
        return _empty_like(a)
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        relu_a = _dispatch_unary_gpu(a, "relu")
        scaled = div(a, alpha)
        exp_scaled = _dispatch_unary_gpu(scaled, "exp")
        elu_part = mul(sub(exp_scaled, 1.0), alpha)
        neg_part = clamp(elu_part, None, 0.0)
        return add(relu_a, neg_part)
    _unsupported_dtype("celu", a)


cpdef object threshold(object a, object threshold_val, object value):
    if a.numel() == 0:
        return _empty_like(a)
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        from candle._backends.mps.ops.comparison import gt
        from candle._backends.mps.ops.elementwise import where
        a_c = a.contiguous() if not a.is_contiguous() else a
        mask = gt(a_c, threshold_val)
        fill = mul(add(mul(a_c, 0.0), 1.0), value)
        return where(mask, a_c, fill)
    _unsupported_dtype("threshold", a)


cpdef object hardshrink(object a, object lambd=0.5):
    if a.numel() == 0:
        return _empty_like(a)
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        from candle._backends.mps.ops.comparison import gt
        from candle._backends.mps.ops.elementwise import where
        a_c = a.contiguous() if not a.is_contiguous() else a
        abs_a = _dispatch_unary_gpu(a_c, "abs")
        mask = gt(abs_a, lambd)
        return where(mask, a_c, mul(a_c, 0.0))
    _unsupported_dtype("hardshrink", a)


cpdef object softshrink(object a, object lambd=0.5):
    if a.numel() == 0:
        return _empty_like(a)
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        a_c = a.contiguous() if not a.is_contiguous() else a
        s = _dispatch_unary_gpu(a_c, "sign")
        abs_a = _dispatch_unary_gpu(a_c, "abs")
        shifted = sub(abs_a, lambd)
        clamped = clamp(shifted, 0.0, None)
        return mul(s, clamped)
    _unsupported_dtype("softshrink", a)


cpdef object rrelu(object a, object lower=1.0/8, object upper=1.0/3, object training=False):
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        from candle._backends.mps.ops.random import uniform_, fill_
        from candle._backends.mps.ops.comparison import ge
        from candle._backends.mps.ops.elementwise import where
        a_c = a.contiguous() if not a.is_contiguous() else a
        slope = _empty_like(a_c)
        if training:
            uniform_(slope, lower, upper)
        else:
            fill_(slope, (lower + upper) / 2.0)
        mask = ge(a_c, 0)
        neg = mul(a_c, slope)
        result = where(mask, a_c, neg)
        result._rrelu_slope = slope
        return result
    arr = _to_numpy(a)
    if training:
        slope = np.random.uniform(lower, upper, size=arr.shape).astype(arr.dtype)
    else:
        slope = np.full_like(arr, (lower + upper) / 2.0)
    out = np.where(arr >= 0, arr, arr * slope)
    result = _from_numpy(out, a.dtype, a.device)
    result._rrelu_slope = _from_numpy(slope, a.dtype, a.device)
    return result


cpdef object hardswish(object a):
    if a.numel() == 0:
        return _empty_like(a)
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        shifted = add(a, 3.0)
        clamped = clamp(shifted, 0.0, 6.0)
        return div(mul(a, clamped), 6.0)
    _unsupported_dtype("hardswish", a)


cpdef object hardsigmoid(object a):
    if a.numel() == 0:
        return _empty_like(a)
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        shifted = add(a, 3.0)
        clamped = clamp(shifted, 0.0, 6.0)
        return div(clamped, 6.0)
    _unsupported_dtype("hardsigmoid", a)


cpdef object softsign(object a):
    if a.numel() == 0:
        return _empty_like(a)
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        abs_a = _dispatch_unary_gpu(a, "abs")
        denom = add(abs_a, 1.0)
        return div(a, denom)
    _unsupported_dtype("softsign", a)


cpdef object softmax(object a, object dim):
    ndim = len(a.shape)
    if a.numel() == 0:
        return _empty_like(a)
    actual_dim = dim if dim >= 0 else dim + ndim
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype) and ndim >= 1:
        if actual_dim == ndim - 1 and a.is_contiguous():
            d = _get_dispatcher()
            sfx = _kernel_suffix(a.dtype)
            numel = a.numel()
            cols = a.shape[ndim - 1]
            rows = numel // cols
            out_buf = _alloc_output_buf(numel, a.dtype)
            d.dispatch_softmax_2d(f"softmax_{sfx}", _metal_buf(a), out_buf,
                                  rows, cols)
            return _from_metal_buffer(out_buf, tuple(a.shape), tuple(a.stride), a.dtype, a.device)
        perm = list(range(ndim))
        last = ndim - 1
        perm[actual_dim], perm[last] = perm[last], perm[actual_dim]
        a_t = a.permute(*perm).contiguous()
        out_t = softmax(a_t, last)
        return out_t.permute(*perm).contiguous()
    _unsupported_dtype("softmax", a)


cpdef object log_softmax(object a, object dim):
    if a.numel() == 0:
        return _empty_like(a)
    ndim = len(a.shape)
    actual_dim = dim if dim >= 0 else dim + ndim
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype) and ndim >= 1:
        s = softmax(a, dim)
        return log(s)
    _unsupported_dtype("log_softmax", a)


cpdef object embedding(object weight, object indices, object padding_idx=None,
                       object scale_grad_by_freq=False, object sparse=False):
    if weight.numel() == 0:
        return _empty_like(weight)
    if (_can_use_gpu(weight) and weight.is_contiguous()
            and weight.dtype in (float32_dtype, float16_dtype)):
        from candle._backends.mps.ops.shape import _ensure_integer_indices
        idx_np = _ensure_integer_indices(_to_numpy(indices), "indices").astype(np.int64, copy=False)
        if idx_np.size and (idx_np.min() < 0 or idx_np.max() >= weight.shape[0]):
            raise IndexError("index out of range in self")
        flat_idx = idx_np.reshape(-1)
        flat_idx_i32 = flat_idx.astype(np.int32)
        idx_tensor = _from_numpy(flat_idx_i32, int32_dtype, weight.device)
        d = _get_dispatcher()
        sfx = _kernel_suffix(weight.dtype)
        vocab, dim = weight.shape[0], weight.shape[1]
        out_numel = len(flat_idx) * dim
        out_buf = _alloc_output_buf(out_numel, weight.dtype)
        d.dispatch_index_gather(f"index_select_{sfx}", _metal_buf(weight),
                                _metal_buf(idx_tensor), out_buf,
                                1, len(flat_idx), dim,
                                vocab, out_numel)
        out_shape = tuple(idx_np.shape) + (weight.shape[1],)
        s = 1
        out_stride = ()
        for d_ in reversed(out_shape):
            out_stride = (s,) + out_stride
            s *= d_
        return _from_metal_buffer(out_buf, out_shape, out_stride, weight.dtype, weight.device)
    _unsupported_dtype("embedding", weight)


cpdef object dropout(object a, object p=0.5, object training=True):
    if not training or p == 0:
        return a
    if p == 1.0:
        return mul(a, 0.0)
    if a.numel() == 0:
        return _empty_like(a)
    if _can_use_gpu(a) and a.is_contiguous() and a.dtype in (float32_dtype, float16_dtype):
        from candle.mps import _get_default_generator
        gen = _get_default_generator()
        numel = a.numel()
        increment = (numel + 3) // 4
        seed, offset = gen.philox_engine_inputs(increment)
        seed_lo, seed_hi = seed & 0xffffffff, (seed >> 32) & 0xffffffff
        sfx = _kernel_suffix(a.dtype)
        scale = 1.0 / (1.0 - p)
        out_buf = _alloc_output_buf(numel, a.dtype)
        _get_dispatcher().dispatch_philox_dropout(
            f"philox_dropout_{sfx}", _metal_buf(a), out_buf,
            float(p), float(scale), seed_lo, seed_hi, offset, numel)
        stride = tuple(a.stride())
        result = _from_metal_buffer(out_buf, tuple(a.shape), stride, a.dtype, a.device)
        result._backward_data = {
            'seed_lo': seed_lo,
            'seed_hi': seed_hi,
            'offset': offset,
            'p': float(p),
            'scale': float(scale),
        }
        return result
    _unsupported_dtype("dropout", a)


# ---------------------------------------------------------------------------
# comparison.py functions
# ---------------------------------------------------------------------------

cpdef object eq(object a, object b):
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
        from candle._tensor import _compute_strides
        return _from_metal_buffer(out_buf, a.shape, _compute_strides(a.shape), bool_dtype, a.device)
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("eq", a)


cpdef object ne(object a, object b):
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
        from candle._tensor import _compute_strides
        return _from_metal_buffer(out_buf, a.shape, _compute_strides(a.shape), bool_dtype, a.device)
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("ne", a)


cpdef object lt(object a, object b):
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
        from candle._tensor import _compute_strides
        return _from_metal_buffer(out_buf, a.shape, _compute_strides(a.shape), bool_dtype, a.device)
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("lt", a)


cpdef object le(object a, object b):
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
        from candle._tensor import _compute_strides
        return _from_metal_buffer(out_buf, a.shape, _compute_strides(a.shape), bool_dtype, a.device)
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("le", a)


cpdef object gt(object a, object b):
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
        from candle._tensor import _compute_strides
        return _from_metal_buffer(out_buf, a.shape, _compute_strides(a.shape), bool_dtype, a.device)
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("gt", a)


cpdef object ge(object a, object b):
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
        from candle._tensor import _compute_strides
        return _from_metal_buffer(out_buf, a.shape, _compute_strides(a.shape), bool_dtype, a.device)
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("ge", a)


cpdef object logical_and(object a, object b):
    if (_can_use_gpu(a)
            and isinstance(b, Tensor) and _can_use_gpu(b)):
        a_bool = ne(a, 0)
        b_bool = ne(b, 0)
        return _dispatch_binary_gpu(a_bool, b_bool, "mul")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("logical_and", a)


cpdef object logical_or(object a, object b):
    if (_can_use_gpu(a)
            and isinstance(b, Tensor) and _can_use_gpu(b)):
        a_bool = ne(a, 0)
        b_bool = ne(b, 0)
        sum_buf = _dispatch_binary_gpu(a_bool, b_bool, "add")
        return ne(sum_buf, 0)
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("logical_or", a)


cpdef object logical_not(object a):
    if _can_use_gpu(a):
        return eq(a, 0)
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("logical_not", a)


cpdef object logical_xor(object a, object b):
    if (_can_use_gpu(a)
            and isinstance(b, Tensor) and _can_use_gpu(b)):
        a_bool = ne(a, 0)
        b_bool = ne(b, 0)
        return ne(a_bool, b_bool)
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("logical_xor", a)


cpdef object bitwise_and(object a, object b):
    if _can_use_gpu(a):
        return _dispatch_binary_gpu(a, b, "bitwise_and")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("bitwise_and", a)


cpdef object bitwise_or(object a, object b):
    if _can_use_gpu(a):
        return _dispatch_binary_gpu(a, b, "bitwise_or")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("bitwise_or", a)


cpdef object bitwise_xor(object a, object b):
    if _can_use_gpu(a):
        return _dispatch_binary_gpu(a, b, "bitwise_xor")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("bitwise_xor", a)


cpdef object bitwise_not(object a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "bitwise_not")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("bitwise_not", a)


cpdef object bitwise_left_shift(object a, object b):
    if _can_use_gpu(a):
        return _dispatch_binary_gpu(a, b, "bitwise_left_shift")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("bitwise_left_shift", a)


cpdef object bitwise_right_shift(object a, object b):
    if _can_use_gpu(a):
        return _dispatch_binary_gpu(a, b, "bitwise_right_shift")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("bitwise_right_shift", a)


cpdef object allclose(object a, object b, object rtol=1e-05, object atol=1e-08, object equal_nan=False):
    close = isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
    from candle._backends.mps.ops.reduce import all_
    return bool(all_(close).item())


cpdef object isclose(object a, object b, object rtol=1e-05, object atol=1e-08, object equal_nan=False):
    if _can_use_gpu(a) and isinstance(b, Tensor) and _can_use_gpu(b):
        from candle._backends.mps.ops.math import abs as _abs, sub as _sub, mul as _mul, add as _add
        diff = _abs(_sub(a, b))
        tol = _add(_dispatch_binary_gpu(_abs(b), float(rtol), "mul"), float(atol))
        close = le(diff, tol)
        if equal_nan:
            from candle._backends.mps.ops.math import isnan
            both_nan = logical_and(isnan(a), isnan(b))
            close = logical_or(close, both_nan)
        return close
    _unsupported_dtype("isclose", a)


cpdef object equal(object a, object b):
    if _can_use_gpu(a) and isinstance(b, Tensor) and _can_use_gpu(b):
        if a.shape != b.shape:
            return False
        from candle._backends.mps.ops.reduce import all_
        return bool(all_(eq(a, b)).item())
    _unsupported_dtype("equal", a)


# ---------------------------------------------------------------------------
# random.py in-place functions
# ---------------------------------------------------------------------------

cpdef object add_(object a, object b):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        if isinstance(b, Tensor) and _can_use_gpu(b):
            d.dispatch_inplace_binary(f"add_inplace_{sfx}", _metal_buf(a),
                                      _metal_buf(b), numel)
        else:
            scalar = float(b) if not isinstance(b, Tensor) else float(_to_numpy(b).ravel()[0])
            d.dispatch_inplace_binary_scalar(f"add_inplace_scalar_{sfx}",
                                              _metal_buf(a), scalar, numel,
                                              scalar_fmt=_scalar_fmt(a.dtype))
        return a
    arr = _to_numpy(a)
    arr += _to_numpy(b) if isinstance(b, Tensor) else b
    return a


cpdef object mul_(object a, object b):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        if isinstance(b, Tensor) and _can_use_gpu(b):
            d.dispatch_inplace_binary(f"mul_inplace_{sfx}", _metal_buf(a),
                                      _metal_buf(b), numel)
        else:
            scalar = float(b) if not isinstance(b, Tensor) else float(_to_numpy(b).ravel()[0])
            d.dispatch_inplace_binary_scalar(f"mul_inplace_scalar_{sfx}",
                                              _metal_buf(a), scalar, numel,
                                              scalar_fmt=_scalar_fmt(a.dtype))
        return a
    arr = _to_numpy(a)
    arr *= _to_numpy(b) if isinstance(b, Tensor) else b
    return a


cpdef object sub_(object a, object b):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        if isinstance(b, Tensor) and _can_use_gpu(b):
            d.dispatch_inplace_binary(f"sub_inplace_{sfx}", _metal_buf(a),
                                      _metal_buf(b), numel)
        else:
            scalar = float(b) if not isinstance(b, Tensor) else float(_to_numpy(b).ravel()[0])
            d.dispatch_inplace_binary_scalar(f"sub_inplace_scalar_{sfx}",
                                              _metal_buf(a), scalar, numel,
                                              scalar_fmt=_scalar_fmt(a.dtype))
        return a
    arr = _to_numpy(a)
    arr -= _to_numpy(b) if isinstance(b, Tensor) else b
    return a


cpdef object div_(object a, object b):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        if isinstance(b, Tensor) and _can_use_gpu(b):
            d.dispatch_inplace_binary(f"div_inplace_{sfx}", _metal_buf(a),
                                      _metal_buf(b), numel)
        else:
            scalar = float(b) if not isinstance(b, Tensor) else float(_to_numpy(b).ravel()[0])
            d.dispatch_inplace_binary_scalar(f"div_inplace_scalar_{sfx}",
                                              _metal_buf(a), scalar, numel,
                                              scalar_fmt=_scalar_fmt(a.dtype))
        return a
    arr = _to_numpy(a)
    b_np = _to_numpy(b) if isinstance(b, Tensor) else b
    arr /= b_np
    return a


cpdef object relu_(object a):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        d.dispatch_inplace_unary(f"relu_inplace_{sfx}", _metal_buf(a), a.numel())
        return a
    arr = _to_numpy(a)
    np.maximum(arr, 0, out=arr)
    return a


cpdef object zero_(object a):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        d.dispatch_fill(f"fill_{sfx}", _metal_buf(a), 0.0, a.numel(),
                        scalar_fmt=_scalar_fmt(a.dtype))
        return a
    arr = _to_numpy(a)
    arr.fill(0)
    return a


cpdef object fill_(object a, object value):
    if _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        d.dispatch_fill(f"fill_{sfx}", _metal_buf(a), float(value), a.numel(),
                        scalar_fmt=_scalar_fmt(a.dtype))
        return a
    arr = _to_numpy(a)
    arr.fill(value)
    return a


cpdef object clamp_(object a, object min_val=None, object max_val=None):
    if _can_use_gpu(a):
        from candle._backends.mps.ops.activation import clamp as _clamp_fn
        clamped = _clamp_fn(a, min_val, max_val)
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        d.dispatch_unary(f"identity_{sfx}", _metal_buf(clamped),
                         _metal_buf(a), a.numel())
        return a
    arr = _to_numpy(a)
    np.clip(arr, min_val, max_val, out=arr)
    return a


cpdef object copy_(object a, object src):
    if _can_use_gpu(a) and _can_use_gpu(src):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = min(a.numel(), src.numel())
        d.dispatch_unary(f"identity_{sfx}", _metal_buf(src), _metal_buf(a), numel)
        return a
    arr = _to_numpy(a)
    src_arr = _to_numpy(src)
    np.copyto(arr, src_arr)
    return a


cpdef object uniform_(object a, object low=0.0, object high=1.0, object generator=None):
    if _can_use_gpu(a) and a.is_contiguous() and a.dtype in (float32_dtype, float16_dtype):
        from candle.mps import _get_default_generator
        gen = generator if (generator is not None and hasattr(generator, 'device') and generator.device.type == 'mps') else _get_default_generator()
        numel = a.numel()
        increment = (numel + 3) // 4
        seed, offset = gen.philox_engine_inputs(increment)
        seed_lo, seed_hi = seed & 0xffffffff, (seed >> 32) & 0xffffffff
        sfx = _kernel_suffix(a.dtype)
        fmt = _scalar_fmt(a.dtype)
        _get_dispatcher().dispatch_philox_fill(
            f"philox_uniform_{sfx}", _metal_buf(a),
            seed_lo, seed_hi, offset, low, high, numel, param_fmt=fmt)
        return a
    from candle._random import _get_cpu_rng
    rng = generator._rng if (generator is not None and hasattr(generator, '_rng') and generator._rng is not None) else _get_cpu_rng()
    arr = _to_numpy(a)
    arr[:] = rng.uniform(low, high, arr.shape).astype(arr.dtype)
    return a


cpdef object normal_(object a, object mean=0.0, object std=1.0, object generator=None):
    if _can_use_gpu(a) and a.is_contiguous() and a.dtype in (float32_dtype, float16_dtype):
        from candle.mps import _get_default_generator
        gen = generator if (generator is not None and hasattr(generator, 'device') and generator.device.type == 'mps') else _get_default_generator()
        numel = a.numel()
        increment = (numel + 3) // 4
        seed, offset = gen.philox_engine_inputs(increment)
        seed_lo, seed_hi = seed & 0xffffffff, (seed >> 32) & 0xffffffff
        sfx = _kernel_suffix(a.dtype)
        fmt = _scalar_fmt(a.dtype)
        _get_dispatcher().dispatch_philox_fill(
            f"philox_normal_{sfx}", _metal_buf(a),
            seed_lo, seed_hi, offset, mean, std, numel, param_fmt=fmt)
        return a
    from candle._random import _get_cpu_rng
    rng = generator._rng if (generator is not None and hasattr(generator, '_rng') and generator._rng is not None) else _get_cpu_rng()
    arr = _to_numpy(a)
    arr[:] = rng.normal(mean, std, arr.shape).astype(arr.dtype)
    return a


cpdef object bernoulli_(object a, object p=0.5, object generator=None):
    is_scalar_p = not hasattr(p, '_numpy_view') and not hasattr(p, 'numpy')
    if is_scalar_p and _can_use_gpu(a) and a.is_contiguous() and a.dtype in (float32_dtype, float16_dtype):
        from candle.mps import _get_default_generator
        gen = generator if (generator is not None and hasattr(generator, 'device') and generator.device.type == 'mps') else _get_default_generator()
        numel = a.numel()
        increment = (numel + 3) // 4
        seed, offset = gen.philox_engine_inputs(increment)
        seed_lo, seed_hi = seed & 0xffffffff, (seed >> 32) & 0xffffffff
        sfx = _kernel_suffix(a.dtype)
        _get_dispatcher().dispatch_philox_bernoulli(
            f"philox_bernoulli_{sfx}", _metal_buf(a), float(p),
            seed_lo, seed_hi, offset, numel)
        return a
    if (not is_scalar_p and _can_use_gpu(a) and a.is_contiguous()
            and a.dtype in (float32_dtype, float16_dtype)
            and isinstance(p, Tensor) and _can_use_gpu(p)):
        from candle._backends.mps.ops.elementwise import where
        uniform_(a, 0.0, 1.0, generator=generator)
        mask = lt(a, p)
        fill_(a, 1.0)
        result = where(mask, a, 0.0)
        copy_(a, result)
        return a
    from candle._random import _get_cpu_rng
    rng = generator._rng if (generator is not None and hasattr(generator, '_rng') and generator._rng is not None) else _get_cpu_rng()
    arr = _to_numpy(a)
    if hasattr(p, '_numpy_view'):
        probs = p._numpy_view()
    elif hasattr(p, 'numpy'):
        probs = p.numpy()
    else:
        probs = float(p)
    arr[...] = (rng.uniform(0, 1, arr.shape) < probs).astype(arr.dtype)
    return a


cpdef object exponential_(object a, object lambd=1.0, object generator=None):
    if _can_use_gpu(a) and a.is_contiguous() and a.dtype in (float32_dtype, float16_dtype):
        from candle._backends.mps.ops.math import neg as _neg, log as _log, div as _div
        uniform_(a, 0.0, 1.0, generator=generator)
        result = _div(_neg(_log(a)), lambd)
        copy_(a, result)
        return a
    from candle._random import _get_cpu_rng
    rng = generator._rng if (generator is not None and hasattr(generator, '_rng') and generator._rng is not None) else _get_cpu_rng()
    arr = _to_numpy(a)
    arr[...] = rng.exponential(1.0 / lambd, arr.shape).astype(arr.dtype)
    return a


cpdef object log_normal_(object a, object mean=1.0, object std=2.0, object generator=None):
    if _can_use_gpu(a) and a.is_contiguous() and a.dtype in (float32_dtype, float16_dtype):
        from candle._backends.mps.ops.math import mul as _mul, add as _add, exp as _exp
        normal_(a, 0.0, 1.0, generator=generator)
        result = _exp(_add(_mul(a, std), mean))
        copy_(a, result)
        return a
    from candle._random import _get_cpu_rng
    rng = generator._rng if (generator is not None and hasattr(generator, '_rng') and generator._rng is not None) else _get_cpu_rng()
    arr = _to_numpy(a)
    arr[...] = rng.lognormal(mean, std, arr.shape).astype(arr.dtype)
    return a


cpdef object cauchy_(object a, object median=0.0, object sigma=1.0, object generator=None):
    if _can_use_gpu(a) and a.is_contiguous() and a.dtype in (float32_dtype, float16_dtype):
        import math as _math
        from candle._backends.mps.ops.math import mul as _mul, add as _add, sub as _sub, tan as _tan
        uniform_(a, 0.0, 1.0, generator=generator)
        result = _add(median, _mul(sigma, _tan(_mul(_math.pi, _sub(a, 0.5)))))
        copy_(a, result)
        return a
    from candle._random import _get_cpu_rng
    rng = generator._rng if (generator is not None and hasattr(generator, '_rng') and generator._rng is not None) else _get_cpu_rng()
    arr = _to_numpy(a)
    arr[...] = (median + sigma * np.tan(np.pi * (rng.uniform(0, 1, arr.shape) - 0.5))).astype(arr.dtype)
    return a


cpdef object geometric_(object a, object p, object generator=None):
    if _can_use_gpu(a) and a.is_contiguous() and a.dtype in (float32_dtype, float16_dtype):
        import math as _math
        from candle._backends.mps.ops.math import sub as _sub, log as _log, div as _div, ceil as _ceil
        uniform_(a, 0.0, 1.0, generator=generator)
        result = _ceil(_div(_log(_sub(1.0, a)), _math.log(1.0 - p)))
        copy_(a, result)
        return a
    from candle._random import _get_cpu_rng
    rng = generator._rng if (generator is not None and hasattr(generator, '_rng') and generator._rng is not None) else _get_cpu_rng()
    arr = _to_numpy(a)
    arr[...] = rng.geometric(p, arr.shape).astype(arr.dtype)
    return a


cpdef object erfinv_(object a):
    if _can_use_gpu(a) and a.is_contiguous() and a.dtype in (float32_dtype, float16_dtype):
        from candle._backends.mps.ops.math import mul as _mul, add as _add, sub as _sub, div as _div, sqrt as _sqrt, log as _log, neg as _neg, abs as _abs
        from candle._backends.mps.ops.elementwise import where
        _sqrt2 = 1.4142135623730951
        p = _div(_add(a, 1.0), 2.0)
        q = _sub(p, 0.5)
        abs_q = _abs(q)
        central_mask = le(abs_q, 0.425)
        r2 = _mul(q, q)
        t = _add(_mul(r2, -25.44106049637), 41.39119773534)
        t = _add(_mul(t, r2), -18.61500062529)
        num = _add(_mul(t, r2), 2.50662823884)
        t = _add(_mul(r2, 3.13082909833), -21.06224101826)
        t = _add(_mul(t, r2), 23.08336743743)
        t = _add(_mul(t, r2), -8.47351093090)
        den = _add(_mul(t, r2), 1.0)
        central_val = _div(_mul(q, num), den)
        one_minus_p = _add(_neg(p), 1.0)
        p_lt_half = le(p, 0.5)
        pp = where(p_lt_half, p, one_minus_p)
        r = _sqrt(_mul(_log(pp), -2.0))
        t = _add(_mul(r, 0.010328), 0.802853)
        t_num = _add(_mul(t, r), 2.515517)
        t = _add(_mul(r, 0.001308), 0.189269)
        t = _add(_mul(t, r), 1.432788)
        t_den = _add(_mul(t, r), 1.0)
        tail_abs = _sub(r, _div(t_num, t_den))
        tail_val = where(p_lt_half, _neg(tail_abs), tail_abs)
        result = _div(where(central_mask, central_val, tail_val), _sqrt2)
        result = where(gt(a, -1.0), result, float('-inf'))
        result = where(lt(a, 1.0), result, float('inf'))
        copy_(a, result)
        return a
    arr = _to_numpy(a)
    arr[:] = _ndtr_inv((arr + 1.0) / 2.0) / np.sqrt(2.0)
    return a


def _ndtr_inv(p):
    """Inverse normal CDF (probit function) using rational approximation.
    Used to compute erfinv: erfinv(x) = ndtr_inv((x+1)/2) / sqrt(2)."""
    p = np.asarray(p, dtype=np.float64)
    result = np.zeros_like(p)

    q = p - 0.5
    mask_central = np.abs(q) <= 0.425
    if np.any(mask_central):
        r = q[mask_central]
        r2 = r * r
        a_coeff = np.array([
            2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637
        ])
        b_coeff = np.array([
            -8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833
        ])
        num = ((a_coeff[3] * r2 + a_coeff[2]) * r2 + a_coeff[1]) * r2 + a_coeff[0]
        den = (((b_coeff[3] * r2 + b_coeff[2]) * r2 + b_coeff[1]) * r2 + b_coeff[0]) * r2 + 1.0
        result[mask_central] = r * num / den

    mask_tail = ~mask_central & (p > 0) & (p < 1)
    if np.any(mask_tail):
        pp = np.where(p[mask_tail] < 0.5, p[mask_tail], 1.0 - p[mask_tail])
        r = np.sqrt(-2.0 * np.log(pp))
        c_coeff = np.array([
            2.515517, 0.802853, 0.010328
        ])
        d_coeff = np.array([
            1.432788, 0.189269, 0.001308
        ])
        num = (c_coeff[2] * r + c_coeff[1]) * r + c_coeff[0]
        den = ((d_coeff[2] * r + d_coeff[1]) * r + d_coeff[0]) * r + 1.0
        val = r - num / den
        val = np.where(p[mask_tail] < 0.5, -val, val)
        result[mask_tail] = val

    result[p <= 0] = -np.inf
    result[p >= 1] = np.inf
    return result


cpdef object randint_(object a, object low, object high=None, object generator=None):
    """In-place randint -- fills tensor a with random integers from [low, high)."""
    if high is None:
        low, high = 0, low
    from candle._random import _get_cpu_rng
    rng = generator._rng if (generator is not None and hasattr(generator, '_rng') and generator._rng is not None) else _get_cpu_rng()
    arr = _to_numpy(a)
    arr[...] = rng.randint(int(low), int(high), size=arr.shape)
    return a


cpdef object random_(object a, object from_=0, object to=None, object generator=None):
    """In-place random -- fills tensor with random values from [from_, to)."""
    from candle._random import _get_cpu_rng
    rng = generator._rng if (generator is not None and hasattr(generator, '_rng') and generator._rng is not None) else _get_cpu_rng()
    arr = _to_numpy(a)
    if to is None:
        if np.issubdtype(arr.dtype, np.floating):
            to = 2**24 if arr.dtype == np.float32 else 2**53
        else:
            to = int(np.iinfo(arr.dtype).max) + 1
    arr[...] = rng.randint(int(from_), int(to), size=arr.shape).astype(arr.dtype)
    return a

# ---------------------------------------------------------------------------
# elementwise.py functions
# ---------------------------------------------------------------------------

def _broadcast_contiguous(t, out_shape):
    """Broadcast tensor to out_shape and return a contiguous copy via unified memory."""
    arr = _to_numpy(t)
    arr = np.broadcast_to(arr, out_shape)
    arr = np.ascontiguousarray(arr)
    return _from_numpy(arr, t.dtype, t.device)


def where(cond, x, y):
    # Broadcast cond, x, y to the same shape if needed
    if isinstance(x, Tensor) and isinstance(y, Tensor) and isinstance(cond, Tensor):
        shapes = [cond.shape, x.shape, y.shape]
        if not (shapes[0] == shapes[1] == shapes[2]):
            max_ndim = max(len(s) for s in shapes)
            padded = [(1,) * (max_ndim - len(s)) + tuple(s) for s in shapes]
            out_shape = []
            for dims in zip(*padded):
                m = max(dims)
                for d in dims:
                    if d != 1 and d != m:
                        raise RuntimeError(
                            f"where: shape mismatch, cannot broadcast {shapes}")
                out_shape.append(m)
            out_shape = tuple(out_shape)
            if tuple(cond.shape) != out_shape:
                cond = _broadcast_contiguous(cond, out_shape)
            if tuple(x.shape) != out_shape:
                x = _broadcast_contiguous(x, out_shape)
            if tuple(y.shape) != out_shape:
                y = _broadcast_contiguous(y, out_shape)
    elif isinstance(x, Tensor) and not isinstance(y, Tensor) and isinstance(cond, Tensor):
        if tuple(cond.shape) != tuple(x.shape):
            max_ndim = max(len(cond.shape), len(x.shape))
            pc = (1,) * (max_ndim - len(cond.shape)) + tuple(cond.shape)
            px = (1,) * (max_ndim - len(x.shape)) + tuple(x.shape)
            out_shape = tuple(max(a, b) for a, b in zip(pc, px))
            if tuple(cond.shape) != out_shape:
                cond = _broadcast_contiguous(cond, out_shape)
            if tuple(x.shape) != out_shape:
                x = _broadcast_contiguous(x, out_shape)

    # GPU path: both cond and x are GPU+contiguous
    if (isinstance(x, Tensor) and _can_use_gpu(x) and x.is_contiguous()
            and isinstance(cond, Tensor) and _can_use_gpu(cond) and cond.is_contiguous()):
        d = _get_dispatcher()
        sfx = _kernel_suffix(x.dtype)
        numel = x.numel()
        out_buf = _alloc_output_buf(numel, x.dtype)
        # Ensure condition is uint8 (uchar) for the shader
        if cond.dtype != bool_dtype:
            from candle._backends.mps.ops.comparison import ne
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
        from candle._backends.mps.ops.shape import expand
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
        from candle._backends.mps.ops.shape import narrow, cat
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
        from candle._backends.mps.ops.comparison import eq, logical_or
        from candle._backends.mps.ops._helpers import _read_scalar
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
    from candle._backends.mps.ops._helpers import _empty_like
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        from candle._backends.mps.ops.random import uniform_
        out = _empty_like(a)
        uniform_(out, 0.0, 1.0)
        return out
    from candle._random import _get_cpu_rng
    rng = _get_cpu_rng()
    arr = rng.uniform(0.0, 1.0, _to_numpy(a).shape).astype(_to_numpy(a).dtype)
    return _from_numpy(arr, a.dtype, a.device)


# ---------------------------------------------------------------------------
# Upsample ops — CPU numpy implementations
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# reduce.py functions
# ---------------------------------------------------------------------------

_abs = abs
_pow = pow

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
    from candle._dtype import float32 as f32
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
        from candle._backends.mps.runtime import buffer_contents
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
        from candle._backends.mps.runtime import buffer_contents
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
            from candle._tensor import _compute_strides
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
    from candle._tensor import _compute_strides
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
        from candle._tensor import _compute_strides
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
            from candle._backends.common.view import squeeze as _squeeze
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
        from candle._backends.mps.ops.math import reciprocal
        ratio = mul(reciprocal(eps_t), float(maxnorm))
        from candle._backends.mps.ops.comparison import gt
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
        from candle._backends.common.view import reshape, squeeze
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
        from candle._backends.common.view import squeeze as _squeeze
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
        from candle._backends.common.view import squeeze as _squeeze
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
        from candle._backends.mps.ops.math import floor
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
        from candle._backends.common.view import reshape, squeeze
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
        from candle._backends.common.view import squeeze
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
