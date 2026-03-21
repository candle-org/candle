# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython op entry points for MPS math and activation ops."""

import numpy as np

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