"""Arithmetic and unary math operations for NPU."""

from ._helpers import (
    _unwrap_storage, _wrap_tensor, _unary_op, _binary_op,
    _cast_tensor_dtype, _broadcast_shape, _broadcast_shape_checked,
    _npu_broadcast_to,
    _scalar_to_npu_tensor, _scalar_to_npu_tensor_no_add,
    _numel, _dtype_itemsize, _use_soc_fallback,
    _nan_like,
    bool_dtype, int32_dtype, int64_dtype, float_dtype,
    npu_typed_storage_from_ptr,
    aclnn, npu_runtime, npu_state, ops_soc,
)


# ---------------------------------------------------------------------------
# Arithmetic (binary)
# ---------------------------------------------------------------------------

try:
    from candle._cython._npu_ops import (
        fast_abs as _fast_abs_impl,
        fast_acos as _fast_acos_impl,
        fast_acosh as _fast_acosh_impl,
        fast_add as _fast_add_impl,
        fast_asin as _fast_asin_impl,
        fast_asinh as _fast_asinh_impl,
        fast_atan as _fast_atan_impl,
        fast_atanh as _fast_atanh_impl,
        fast_ceil as _fast_ceil_impl,
        fast_cos as _fast_cos_impl,
        fast_cosh as _fast_cosh_impl,
        fast_erf as _fast_erf_impl,
        fast_erfc as _fast_erfc_impl,
        fast_exp as _fast_exp_impl,
        fast_exp2 as _fast_exp2_impl,
        fast_expm1 as _fast_expm1_impl,
        fast_floor as _fast_floor_impl,
        fast_isfinite as _fast_isfinite_impl,
        fast_isneginf as _fast_isneginf_impl,
        fast_isposinf as _fast_isposinf_impl,
        fast_log as _fast_log_impl,
        fast_log1p as _fast_log1p_impl,
        fast_log10 as _fast_log10_impl,
        fast_log2 as _fast_log2_impl,
        fast_mul as _fast_mul_impl,
        fast_neg as _fast_neg_impl,
        fast_round as _fast_round_impl,
        fast_rsqrt as _fast_rsqrt_impl,
        fast_sigmoid as _fast_sigmoid_impl,
        fast_sign as _fast_sign_impl,
        fast_signbit as _fast_signbit_impl,
        fast_sin as _fast_sin_impl,
        fast_sinh as _fast_sinh_impl,
        fast_sqrt as _fast_sqrt_impl,
        fast_square as _fast_square_impl,
        fast_tan as _fast_tan_impl,
        fast_tanh as _fast_tanh_impl,
        fast_trunc as _fast_trunc_impl,
    )  # pylint: disable=import-error,no-name-in-module
    _HAS_FAST_ABS = True
    _HAS_FAST_ACOS = True
    _HAS_FAST_ACOSH = True
    _HAS_FAST_ADD = True
    _HAS_FAST_ASIN = True
    _HAS_FAST_ASINH = True
    _HAS_FAST_ATAN = True
    _HAS_FAST_ATANH = True
    _HAS_FAST_CEIL = True
    _HAS_FAST_COS = True
    _HAS_FAST_COSH = True
    _HAS_FAST_ERF = True
    _HAS_FAST_ERFC = True
    _HAS_FAST_EXP = True
    _HAS_FAST_EXP2 = True
    _HAS_FAST_EXPM1 = True
    _HAS_FAST_FLOOR = True
    _HAS_FAST_ISFINITE = True
    _HAS_FAST_ISNEGINF = True
    _HAS_FAST_ISPOSINF = True
    _HAS_FAST_LOG = True
    _HAS_FAST_LOG1P = True
    _HAS_FAST_LOG10 = True
    _HAS_FAST_LOG2 = True
    _HAS_FAST_MUL = True
    _HAS_FAST_NEG = True
    _HAS_FAST_ROUND = True
    _HAS_FAST_RSQRT = True
    _HAS_FAST_SIGMOID = True
    _HAS_FAST_SIGN = True
    _HAS_FAST_SIGNBIT = True
    _HAS_FAST_SIN = True
    _HAS_FAST_SINH = True
    _HAS_FAST_SQRT = True
    _HAS_FAST_SQUARE = True
    _HAS_FAST_TAN = True
    _HAS_FAST_TANH = True
    _HAS_FAST_TRUNC = True
except ImportError:
    _fast_abs_impl = None  # type: ignore[assignment]
    _fast_acos_impl = None  # type: ignore[assignment]
    _fast_acosh_impl = None  # type: ignore[assignment]
    _fast_add_impl = None  # type: ignore[assignment]
    _fast_asin_impl = None  # type: ignore[assignment]
    _fast_asinh_impl = None  # type: ignore[assignment]
    _fast_atan_impl = None  # type: ignore[assignment]
    _fast_atanh_impl = None  # type: ignore[assignment]
    _fast_ceil_impl = None  # type: ignore[assignment]
    _fast_cos_impl = None  # type: ignore[assignment]
    _fast_cosh_impl = None  # type: ignore[assignment]
    _fast_erf_impl = None  # type: ignore[assignment]
    _fast_erfc_impl = None  # type: ignore[assignment]
    _fast_exp_impl = None  # type: ignore[assignment]
    _fast_exp2_impl = None  # type: ignore[assignment]
    _fast_expm1_impl = None  # type: ignore[assignment]
    _fast_floor_impl = None  # type: ignore[assignment]
    _fast_isfinite_impl = None  # type: ignore[assignment]
    _fast_isneginf_impl = None  # type: ignore[assignment]
    _fast_isposinf_impl = None  # type: ignore[assignment]
    _fast_log_impl = None  # type: ignore[assignment]
    _fast_log1p_impl = None  # type: ignore[assignment]
    _fast_log10_impl = None  # type: ignore[assignment]
    _fast_log2_impl = None  # type: ignore[assignment]
    _fast_mul_impl = None  # type: ignore[assignment]
    _fast_neg_impl = None  # type: ignore[assignment]
    _fast_round_impl = None  # type: ignore[assignment]
    _fast_rsqrt_impl = None  # type: ignore[assignment]
    _fast_sigmoid_impl = None  # type: ignore[assignment]
    _fast_sign_impl = None  # type: ignore[assignment]
    _fast_signbit_impl = None  # type: ignore[assignment]
    _fast_sin_impl = None  # type: ignore[assignment]
    _fast_sinh_impl = None  # type: ignore[assignment]
    _fast_sqrt_impl = None  # type: ignore[assignment]
    _fast_square_impl = None  # type: ignore[assignment]
    _fast_tan_impl = None  # type: ignore[assignment]
    _fast_tanh_impl = None  # type: ignore[assignment]
    _fast_trunc_impl = None  # type: ignore[assignment]
    _HAS_FAST_ABS = False
    _HAS_FAST_ACOS = False
    _HAS_FAST_ACOSH = False
    _HAS_FAST_ADD = False
    _HAS_FAST_ASIN = False
    _HAS_FAST_ASINH = False
    _HAS_FAST_ATAN = False
    _HAS_FAST_ATANH = False
    _HAS_FAST_CEIL = False
    _HAS_FAST_COS = False
    _HAS_FAST_COSH = False
    _HAS_FAST_ERF = False
    _HAS_FAST_ERFC = False
    _HAS_FAST_EXP = False
    _HAS_FAST_EXP2 = False
    _HAS_FAST_EXPM1 = False
    _HAS_FAST_FLOOR = False
    _HAS_FAST_ISFINITE = False
    _HAS_FAST_ISNEGINF = False
    _HAS_FAST_ISPOSINF = False
    _HAS_FAST_LOG = False
    _HAS_FAST_LOG1P = False
    _HAS_FAST_LOG10 = False
    _HAS_FAST_LOG2 = False
    _HAS_FAST_MUL = False
    _HAS_FAST_NEG = False
    _HAS_FAST_ROUND = False
    _HAS_FAST_RSQRT = False
    _HAS_FAST_SIGMOID = False
    _HAS_FAST_SIGN = False
    _HAS_FAST_SIGNBIT = False
    _HAS_FAST_SIN = False
    _HAS_FAST_SINH = False
    _HAS_FAST_SQRT = False
    _HAS_FAST_SQUARE = False
    _HAS_FAST_TAN = False
    _HAS_FAST_TANH = False
    _HAS_FAST_TRUNC = False


def add(a, b):
    if isinstance(b, (int, float)):
        b = _scalar_to_npu_tensor(b, a)
    if _HAS_FAST_ADD:
        return _fast_add_impl(a, b)
    return _binary_op(a, b, aclnn.add, "add")


def mul(a, b):
    if isinstance(b, (int, float)):
        b = _scalar_to_npu_tensor(b, a)
    if _HAS_FAST_MUL:
        return _fast_mul_impl(a, b)
    return _binary_op(a, b, aclnn.mul, "mul")


def sub(a, b):
    if isinstance(b, (int, float)):
        b = _scalar_to_npu_tensor(b, a)
    return _binary_op(a, b, aclnn.sub, "sub")


def div(a, b):
    if isinstance(b, (int, float)):
        b = _scalar_to_npu_tensor(b, a)
    return _binary_op(a, b, aclnn.div, "div")


# ---------------------------------------------------------------------------
# In-place arithmetic
# ---------------------------------------------------------------------------

def add_(a, b):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if isinstance(b, (int, float)):
        b = _scalar_to_npu_tensor(b, a)
    if a.device.type != "npu" or b.device.type != "npu":
        raise ValueError("NPU add_ expects NPU tensors")
    if a.dtype != b.dtype:
        raise ValueError("NPU add_ requires matching dtypes")
    out_shape = _broadcast_shape(a.shape, b.shape)
    if out_shape != a.shape:
        raise ValueError("NPU add_ requires broadcastable to self shape")
    a_storage = _unwrap_storage(a)
    b_storage = _unwrap_storage(b)
    aclnn.add(
        a_storage.data_ptr(),
        b_storage.data_ptr(),
        a_storage.data_ptr(),
        a.shape,
        a.stride,
        b.shape,
        b.stride,
        out_shape,
        a.stride,
        a.dtype,
        runtime,
        stream=stream.stream,
    )
    return a


def mul_(a, b):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if isinstance(b, (int, float)):
        b = _scalar_to_npu_tensor(b, a)
    if a.device.type != "npu" or b.device.type != "npu":
        raise ValueError("NPU mul_ expects NPU tensors")
    if a.dtype != b.dtype:
        raise ValueError("NPU mul_ requires matching dtypes")
    out_shape = _broadcast_shape(a.shape, b.shape)
    if out_shape != a.shape:
        raise ValueError("NPU mul_ requires broadcastable to self shape")
    a_storage = _unwrap_storage(a)
    b_storage = _unwrap_storage(b)
    aclnn.mul(
        a_storage.data_ptr(),
        b_storage.data_ptr(),
        a_storage.data_ptr(),
        a.shape,
        a.stride,
        b.shape,
        b.stride,
        out_shape,
        a.stride,
        a.dtype,
        runtime,
        stream=stream.stream,
    )
    return a


def sub_(a, b):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if isinstance(b, (int, float)):
        b = _scalar_to_npu_tensor(b, a)
    if a.device.type != "npu" or b.device.type != "npu":
        raise ValueError("NPU sub_ expects NPU tensors")
    a_storage = _unwrap_storage(a)
    b_storage = _unwrap_storage(b)
    aclnn.sub(
        a_storage.data_ptr(),
        b_storage.data_ptr(),
        a_storage.data_ptr(),
        a.shape,
        a.stride,
        b.shape,
        b.stride,
        a.shape,
        a.stride,
        a.dtype,
        runtime,
        stream=stream.stream,
    )
    return a


def div_(a, b):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if isinstance(b, (int, float)):
        b = _scalar_to_npu_tensor(b, a)
    if a.device.type != "npu" or b.device.type != "npu":
        raise ValueError("NPU div_ expects NPU tensors")
    a_storage = _unwrap_storage(a)
    b_storage = _unwrap_storage(b)
    aclnn.div(
        a_storage.data_ptr(),
        b_storage.data_ptr(),
        a_storage.data_ptr(),
        a.shape,
        a.stride,
        b.shape,
        b.stride,
        a.shape,
        a.stride,
        a.dtype,
        runtime,
        stream=stream.stream,
    )
    return a


# ---------------------------------------------------------------------------
# Unary math (simple)
# ---------------------------------------------------------------------------

def abs(a):
    if _HAS_FAST_ABS:
        return _fast_abs_impl(a)
    return _unary_op(a, aclnn.abs, "abs")


def neg(a):
    if _HAS_FAST_NEG:
        return _fast_neg_impl(a)
    return _unary_op(a, aclnn.neg, "neg")


def sign(a):
    if _HAS_FAST_SIGN:
        return _fast_sign_impl(a)
    return _unary_op(a, aclnn.sign, "sign")


def signbit(a):
    if _HAS_FAST_SIGNBIT:
        return _fast_signbit_impl(a)
    return _unary_op(a, aclnn.signbit, "signbit", out_dtype=bool_dtype)


def square(a):
    if aclnn.square_symbols_ok():
        try:
            if _HAS_FAST_SQUARE:
                return _fast_square_impl(a)
            return _unary_op(a, aclnn.square, "square")
        except RuntimeError:
            pass
    return mul(a, a)


# ---------------------------------------------------------------------------
# Float classification
# ---------------------------------------------------------------------------

def isfinite(a):
    if _HAS_FAST_ISFINITE:
        return _fast_isfinite_impl(a)
    return _unary_op(a, aclnn.isfinite, "isfinite", out_dtype=bool_dtype)


def isinf(a):
    """Check for infinity values.

    When fallback is active (910B): aclnnIsInf returns 161001 (unavailable),
    so we use composite: !isfinite(x) & isfinite(1/x).
    """
    # Lazy import to avoid circular dependency with comparison/logical ops
    from . import logical_and, logical_not

    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU isinf expects NPU tensors")
    out_shape = a.shape
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_size = _numel(out_shape) * _dtype_itemsize(bool_dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    storage = _unwrap_storage(a)
    if not a.dtype.is_floating_point:
        runtime.defer_free(out_ptr)
        return logical_not(isfinite(a))
    if _use_soc_fallback("isinf"):
        # Composite: !isfinite(x) & isfinite(1/x)
        if not (aclnn.logical_not_symbols_ok() and aclnn.logical_and_symbols_ok()):
            raise RuntimeError("aclnnIsInf unavailable and logical ops missing")
        runtime.defer_free(out_ptr)
        finite = isfinite(a)
        recip = pow(a, -1.0)
        recip_finite = isfinite(recip)
        return logical_and(logical_not(finite), recip_finite)
    # TODO: re-enable native aclnnIsInf when CANN fixes 161001
    aclnn.isinf(
        storage.data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        a.dtype,
        runtime,
        stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), bool_dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def isnan(a):
    # Lazy import to avoid circular dependency with comparison/logical ops
    from . import logical_and, logical_not

    if a.device.type != "npu":
        raise ValueError("NPU isnan expects NPU tensors")
    if not a.dtype.is_floating_point:
        return logical_not(isfinite(a))
    if not (aclnn.logical_not_symbols_ok() and aclnn.logical_and_symbols_ok()):
        raise RuntimeError("aclnn logical ops missing for isnan")
    finite = isfinite(a)
    recip = pow(a, -1.0)
    recip_finite = isfinite(recip)
    return logical_and(logical_not(finite), logical_not(recip_finite))


def isposinf(a):
    # Lazy import to avoid circular dependency with comparison/logical ops
    from . import logical_and, gt

    if aclnn.isposinf_symbols_ok():
        try:
            if _HAS_FAST_ISPOSINF:
                return _fast_isposinf_impl(a)
            return _unary_op(a, aclnn.isposinf, "isposinf", out_dtype=bool_dtype)
        except RuntimeError:
            pass
    return logical_and(isinf(a), gt(a, _scalar_to_npu_tensor(0, a)))


def isneginf(a):
    # Lazy import to avoid circular dependency with comparison/logical ops
    from . import logical_and, lt

    if aclnn.isneginf_symbols_ok():
        try:
            if _HAS_FAST_ISNEGINF:
                return _fast_isneginf_impl(a)
            return _unary_op(a, aclnn.isneginf, "isneginf", out_dtype=bool_dtype)
        except RuntimeError:
            pass
    return logical_and(isinf(a), lt(a, _scalar_to_npu_tensor(0, a)))


# ---------------------------------------------------------------------------
# Transcendental
# ---------------------------------------------------------------------------

def exp(a):
    if _HAS_FAST_EXP:
        return _fast_exp_impl(a)
    return _unary_op(a, aclnn.exp, "exp")


def log(a):
    if _HAS_FAST_LOG:
        return _fast_log_impl(a)
    return _unary_op(a, aclnn.log, "log")


def expm1(a):
    if _HAS_FAST_EXPM1:
        return _fast_expm1_impl(a)
    if not aclnn.expm1_symbols_ok():
        raise RuntimeError("aclnnExpm1 symbols not available")
    return _unary_op(a, aclnn.expm1, "expm1")


def log1p(a):
    if _HAS_FAST_LOG1P:
        return _fast_log1p_impl(a)
    if not aclnn.log1p_symbols_ok():
        raise RuntimeError("aclnnLog1p symbols not available")
    return _unary_op(a, aclnn.log1p, "log1p")


def sqrt(a):
    if _HAS_FAST_SQRT:
        return _fast_sqrt_impl(a)
    return _unary_op(a, aclnn.sqrt, "sqrt")


def rsqrt(a):
    if _HAS_FAST_RSQRT:
        return _fast_rsqrt_impl(a)
    return _unary_op(a, aclnn.rsqrt, "rsqrt")


def sin(a):
    if _HAS_FAST_SIN:
        return _fast_sin_impl(a)
    return _unary_op(a, aclnn.sin, "sin")


def cos(a):
    if _HAS_FAST_COS:
        return _fast_cos_impl(a)
    return _unary_op(a, aclnn.cos, "cos")


def tan(a):
    if _HAS_FAST_TAN:
        return _fast_tan_impl(a)
    return _unary_op(a, aclnn.tan, "tan")


def tanh(a):
    if _HAS_FAST_TANH:
        return _fast_tanh_impl(a)
    return _unary_op(a, aclnn.tanh, "tanh")


def sigmoid(a):
    if _HAS_FAST_SIGMOID:
        return _fast_sigmoid_impl(a)
    return _unary_op(a, aclnn.sigmoid, "sigmoid")


def sinh(a):
    if _HAS_FAST_SINH:
        return _fast_sinh_impl(a)
    return _unary_op(a, aclnn.sinh, "sinh")


def cosh(a):
    if _HAS_FAST_COSH:
        return _fast_cosh_impl(a)
    return _unary_op(a, aclnn.cosh, "cosh")


def erf(a):
    if _HAS_FAST_ERF:
        return _fast_erf_impl(a)
    return _unary_op(a, aclnn.erf, "erf")


def erfc(a):
    if _HAS_FAST_ERFC:
        return _fast_erfc_impl(a)
    return _unary_op(a, aclnn.erfc, "erfc")


def floor(a):
    if _HAS_FAST_FLOOR:
        return _fast_floor_impl(a)
    return _unary_op(a, aclnn.floor, "floor")


def ceil(a):
    if _HAS_FAST_CEIL:
        return _fast_ceil_impl(a)
    return _unary_op(a, aclnn.ceil, "ceil")


def round(a):
    if _HAS_FAST_ROUND:
        return _fast_round_impl(a)
    return _unary_op(a, aclnn.round, "round")


def trunc(a):
    try:
        if _HAS_FAST_TRUNC:
            return _fast_trunc_impl(a)
        return _unary_op(a, aclnn.trunc, "trunc")
    except RuntimeError as exc:
        if "561103" not in str(exc):
            raise
    if not a.dtype.is_floating_point:
        return a
    if not aclnn.sign_symbols_ok():
        raise RuntimeError("aclnnTrunc not available and aclnnSign unavailable")
    return mul(sign(a), floor(abs(a)))


def frac(a):
    if _use_soc_fallback("frac"):
        # TODO: re-enable native kernel when CANN fixes aclnnFrac on 910A (561000)
        out = trunc(a)
        return add(a, neg(out))
    try:
        return _unary_op(a, aclnn.frac, "frac")
    except RuntimeError as exc:
        if "561103" not in str(exc):
            raise
    out = trunc(a)
    return add(a, neg(out))


def log2(a):
    if _HAS_FAST_LOG2:
        return _fast_log2_impl(a)
    return _unary_op(a, aclnn.log2, "log2")


def log10(a):
    if _HAS_FAST_LOG10:
        return _fast_log10_impl(a)
    return _unary_op(a, aclnn.log10, "log10")


def exp2(a):
    if _HAS_FAST_EXP2:
        return _fast_exp2_impl(a)
    return _unary_op(a, aclnn.exp2, "exp2")

def asinh(a):
    if _HAS_FAST_ASINH:
        return _fast_asinh_impl(a)
    return _unary_op(a, aclnn.asinh, "asinh")


def acosh(a):
    if _HAS_FAST_ACOSH:
        return _fast_acosh_impl(a)
    return _unary_op(a, aclnn.acosh, "acosh")


def atanh(a):
    if _HAS_FAST_ATANH:
        return _fast_atanh_impl(a)
    return _unary_op(a, aclnn.atanh, "atanh")


def atan(a):
    if _HAS_FAST_ATAN:
        return _fast_atan_impl(a)
    return _unary_op(a, aclnn.atan, "atan")


def asin(a):
    if _HAS_FAST_ASIN:
        return _fast_asin_impl(a)
    return _unary_op(a, aclnn.asin, "asin")


def acos(a):
    if _HAS_FAST_ACOS:
        return _fast_acos_impl(a)
    return _unary_op(a, aclnn.acos, "acos")


# ---------------------------------------------------------------------------
# Binary math
# ---------------------------------------------------------------------------

def atan2(a, b):
    # Lazy import to avoid circular dependency with comparison/logical ops
    from . import eq, lt, ge, gt, logical_and, where

    if _use_soc_fallback("atan2"):
        z = div(a, b)
        out = atan(z)

        zero = _scalar_to_npu_tensor(0, out)
        pi = _scalar_to_npu_tensor(3.141592653589793, out)
        pi_half = _scalar_to_npu_tensor(1.5707963267948966, out)

        x_lt0 = lt(b, zero)
        x_eq0 = eq(b, zero)
        y_ge0 = ge(a, zero)
        y_gt0 = gt(a, zero)
        y_lt0 = lt(a, zero)
        y_eq0 = eq(a, zero)

        out = where(logical_and(x_lt0, y_ge0), add(out, pi), out)
        out = where(logical_and(x_lt0, y_lt0), sub(out, pi), out)
        out = where(logical_and(x_eq0, y_gt0), pi_half, out)
        out = where(logical_and(x_eq0, y_lt0), neg(pi_half), out)
        out = where(logical_and(x_eq0, y_eq0), zero, out)
        return out

    return _binary_op(a, b, aclnn.atan2, "atan2")


def _pow_tensor_scalar_op(a, exponent):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU pow expects NPU tensors")
    out_size = _numel(a.shape) * _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    storage = _unwrap_storage(a)
    aclnn.pow_tensor_scalar(
        storage.data_ptr(),
        exponent,
        out_ptr,
        a.shape,
        a.stride,
        a.dtype,
        runtime,
        stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(a.shape), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, a.shape, a.stride)


def reciprocal(a):
    return pow(a, -1.0)


def pow(a, b):
    if hasattr(b, "shape"):
        return _binary_op(a, b, aclnn.pow_tensor_tensor, "pow")
    return _pow_tensor_scalar_op(a, b)


def floor_divide(a, b):
    """Compute floor division using aclnnFloorDivide."""
    from ...._tensor import Tensor
    if not isinstance(b, Tensor):
        from ...._creation import tensor as _tensor
        b = _tensor(float(b), device=a.device)
    return _binary_op(a, b, aclnn.floor_divide, "floor_divide")
