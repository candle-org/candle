"""Arithmetic and unary math operations for NPU."""

from ._helpers import (
    _scalar_to_npu_tensor,
    _use_soc_fallback,
)


# ---------------------------------------------------------------------------
# Arithmetic (binary)
# ---------------------------------------------------------------------------

try:
    from candle._C._npu_ops import (
        fast_abs as _fast_abs_impl,
        fast_acos as _fast_acos_impl,
        fast_acosh as _fast_acosh_impl,
        fast_add as _fast_add_impl,
        fast_add_inplace as _fast_add_inplace_impl,
        fast_asin as _fast_asin_impl,
        fast_asinh as _fast_asinh_impl,
        fast_atan as _fast_atan_impl,
        fast_atan2 as _fast_atan2_impl,
        fast_atanh as _fast_atanh_impl,
        fast_ceil as _fast_ceil_impl,
        fast_ceil_inplace as _fast_ceil_inplace_impl,
        fast_cos as _fast_cos_impl,
        fast_cos_inplace as _fast_cos_inplace_impl,
        fast_cosh as _fast_cosh_impl,
        fast_erf as _fast_erf_impl,
        fast_div as _fast_div_impl,
        fast_div_inplace as _fast_div_inplace_impl,
        fast_erfc as _fast_erfc_impl,
        fast_exp as _fast_exp_impl,
        fast_exp_inplace as _fast_exp_inplace_impl,
        fast_exp2 as _fast_exp2_impl,
        fast_expm1 as _fast_expm1_impl,
        fast_floor as _fast_floor_impl,
        fast_floor_inplace as _fast_floor_inplace_impl,
        fast_frac as _fast_frac_impl,
        fast_isfinite as _fast_isfinite_impl,
        fast_isinf as _fast_isinf_impl,
        fast_isnan as _fast_isnan_impl,
        fast_isneginf as _fast_isneginf_impl,
        fast_isposinf as _fast_isposinf_impl,
        fast_log as _fast_log_impl,
        fast_log_inplace as _fast_log_inplace_impl,
        fast_log1p as _fast_log1p_impl,
        fast_log10 as _fast_log10_impl,
        fast_log2 as _fast_log2_impl,
        fast_floor_divide as _fast_floor_divide_impl,
        fast_mul as _fast_mul_impl,
        fast_mul_inplace as _fast_mul_inplace_impl,
        fast_neg as _fast_neg_impl,
        fast_neg_inplace as _fast_neg_inplace_impl,
        fast_pow as _fast_pow_impl,
        fast_pow_tensor_scalar as _fast_pow_tensor_scalar_impl,
        fast_reciprocal as _fast_reciprocal_impl,
        fast_round as _fast_round_impl,
        fast_rsqrt as _fast_rsqrt_impl,
        fast_sigmoid as _fast_sigmoid_impl,
        fast_sign as _fast_sign_impl,
        fast_signbit as _fast_signbit_impl,
        fast_sin as _fast_sin_impl,
        fast_sin_inplace as _fast_sin_inplace_impl,
        fast_sinh as _fast_sinh_impl,
        fast_sqrt as _fast_sqrt_impl,
        fast_square as _fast_square_impl,
        fast_sub as _fast_sub_impl,
        fast_sub_inplace as _fast_sub_inplace_impl,
        fast_tan as _fast_tan_impl,
        fast_tan_inplace as _fast_tan_inplace_impl,
        fast_tanh as _fast_tanh_impl,
        fast_trunc as _fast_trunc_impl,
    )  # pylint: disable=import-error,no-name-in-module
    _HAS_FAST_ADD = True
    _HAS_FAST_ABS = True
    _HAS_FAST_NEG = True
    _HAS_FAST_SIGN = True
    _HAS_FAST_SIGNBIT = True
    _HAS_FAST_ISFINITE = True
    _HAS_FAST_ISINF = True
    _HAS_FAST_ISNAN = True
    _HAS_FAST_ISPOSINF = True
    _HAS_FAST_ISNEGINF = True
    _HAS_FAST_SQUARE = True
    _HAS_FAST_EXP = True
    _HAS_FAST_EXPM1 = True
    _HAS_FAST_LOG = True
    _HAS_FAST_LOG1P = True
    _HAS_FAST_SQRT = True
    _HAS_FAST_RSQRT = True
    _HAS_FAST_SIN = True
    _HAS_FAST_COS = True
    _HAS_FAST_TAN = True
    _HAS_FAST_TANH = True
    _HAS_FAST_SIGMOID = True
    _HAS_FAST_SINH = True
    _HAS_FAST_COSH = True
    _HAS_FAST_ERF = True
    _HAS_FAST_ERFC = True
    _HAS_FAST_FLOOR = True
    _HAS_FAST_FRAC = True
    _HAS_FAST_RECIPROCAL = True
    _HAS_FAST_CEIL = True
    _HAS_FAST_ROUND = True
    _HAS_FAST_TRUNC = True
    _HAS_FAST_LOG2 = True
    _HAS_FAST_LOG10 = True
    _HAS_FAST_EXP2 = True
    _HAS_FAST_ASINH = True
    _HAS_FAST_ACOSH = True
    _HAS_FAST_ATANH = True
    _HAS_FAST_ATAN = True
    _HAS_FAST_ATAN2 = True
    _HAS_FAST_ASIN = True
    _HAS_FAST_ACOS = True
    _HAS_FAST_DIV = True
    _HAS_FAST_FLOOR_DIVIDE = True
    _HAS_FAST_MUL = True
    _HAS_FAST_POW = True
    _HAS_FAST_POW_TENSOR_SCALAR = True
    _HAS_FAST_SUB = True
    _HAS_FAST_ADD_INPLACE = True
    _HAS_FAST_SUB_INPLACE = True
    _HAS_FAST_MUL_INPLACE = True
    _HAS_FAST_DIV_INPLACE = True
except ImportError:
    _fast_add_impl = None  # type: ignore[assignment]
    _fast_abs_impl = None  # type: ignore[assignment]
    _fast_neg_impl = None  # type: ignore[assignment]
    _fast_pow_impl = None  # type: ignore[assignment]
    _fast_pow_tensor_scalar_impl = None  # type: ignore[assignment]
    _fast_sign_impl = None  # type: ignore[assignment]
    _fast_signbit_impl = None  # type: ignore[assignment]
    _fast_isfinite_impl = None  # type: ignore[assignment]
    _fast_isinf_impl = None  # type: ignore[assignment]
    _fast_isnan_impl = None  # type: ignore[assignment]
    _fast_isposinf_impl = None  # type: ignore[assignment]
    _fast_isneginf_impl = None  # type: ignore[assignment]
    _fast_square_impl = None  # type: ignore[assignment]
    _fast_sub_impl = None  # type: ignore[assignment]
    _fast_exp_impl = None  # type: ignore[assignment]
    _fast_div_impl = None  # type: ignore[assignment]
    _fast_expm1_impl = None  # type: ignore[assignment]
    _fast_floor_divide_impl = None  # type: ignore[assignment]
    _fast_log_impl = None  # type: ignore[assignment]
    _fast_log1p_impl = None  # type: ignore[assignment]
    _fast_sqrt_impl = None  # type: ignore[assignment]
    _fast_rsqrt_impl = None  # type: ignore[assignment]
    _fast_sin_impl = None  # type: ignore[assignment]
    _fast_cos_impl = None  # type: ignore[assignment]
    _fast_tan_impl = None  # type: ignore[assignment]
    _fast_tanh_impl = None  # type: ignore[assignment]
    _fast_sigmoid_impl = None  # type: ignore[assignment]
    _fast_sinh_impl = None  # type: ignore[assignment]
    _fast_cosh_impl = None  # type: ignore[assignment]
    _fast_erf_impl = None  # type: ignore[assignment]
    _fast_erfc_impl = None  # type: ignore[assignment]
    _fast_floor_impl = None  # type: ignore[assignment]
    _fast_frac_impl = None  # type: ignore[assignment]
    _fast_reciprocal_impl = None  # type: ignore[assignment]
    _fast_ceil_impl = None  # type: ignore[assignment]
    _fast_round_impl = None  # type: ignore[assignment]
    _fast_trunc_impl = None  # type: ignore[assignment]
    _fast_log2_impl = None  # type: ignore[assignment]
    _fast_log10_impl = None  # type: ignore[assignment]
    _fast_exp2_impl = None  # type: ignore[assignment]
    _fast_asinh_impl = None  # type: ignore[assignment]
    _fast_acosh_impl = None  # type: ignore[assignment]
    _fast_atanh_impl = None  # type: ignore[assignment]
    _fast_atan_impl = None  # type: ignore[assignment]
    _fast_atan2_impl = None  # type: ignore[assignment]
    _fast_asin_impl = None  # type: ignore[assignment]
    _fast_acos_impl = None  # type: ignore[assignment]
    _fast_mul_impl = None  # type: ignore[assignment]
    _HAS_FAST_ADD = False
    _HAS_FAST_ABS = False
    _HAS_FAST_NEG = False
    _HAS_FAST_SIGN = False
    _HAS_FAST_SIGNBIT = False
    _HAS_FAST_ISFINITE = False
    _HAS_FAST_ISINF = False
    _HAS_FAST_ISNAN = False
    _HAS_FAST_ISPOSINF = False
    _HAS_FAST_ISNEGINF = False
    _HAS_FAST_SQUARE = False
    _HAS_FAST_EXP = False
    _HAS_FAST_EXPM1 = False
    _HAS_FAST_LOG = False
    _HAS_FAST_LOG1P = False
    _HAS_FAST_SQRT = False
    _HAS_FAST_RSQRT = False
    _HAS_FAST_SIN = False
    _HAS_FAST_COS = False
    _HAS_FAST_TAN = False
    _HAS_FAST_TANH = False
    _HAS_FAST_SIGMOID = False
    _HAS_FAST_SINH = False
    _HAS_FAST_COSH = False
    _HAS_FAST_ERF = False
    _HAS_FAST_ERFC = False
    _HAS_FAST_FLOOR = False
    _HAS_FAST_FRAC = False
    _HAS_FAST_RECIPROCAL = False
    _HAS_FAST_CEIL = False
    _HAS_FAST_ROUND = False
    _HAS_FAST_TRUNC = False
    _HAS_FAST_LOG2 = False
    _HAS_FAST_LOG10 = False
    _HAS_FAST_EXP2 = False
    _HAS_FAST_ASINH = False
    _HAS_FAST_ACOSH = False
    _HAS_FAST_ATANH = False
    _HAS_FAST_ATAN = False
    _HAS_FAST_ATAN2 = False
    _HAS_FAST_ASIN = False
    _HAS_FAST_ACOS = False
    _HAS_FAST_DIV = False
    _HAS_FAST_FLOOR_DIVIDE = False
    _HAS_FAST_MUL = False
    _HAS_FAST_POW = False
    _HAS_FAST_POW_TENSOR_SCALAR = False
    _HAS_FAST_SUB = False
    _HAS_FAST_ADD_INPLACE = False
    _HAS_FAST_SUB_INPLACE = False
    _HAS_FAST_MUL_INPLACE = False
    _HAS_FAST_DIV_INPLACE = False
    _fast_add_inplace_impl = None  # type: ignore[assignment]
    _fast_sub_inplace_impl = None  # type: ignore[assignment]
    _fast_mul_inplace_impl = None  # type: ignore[assignment]
    _fast_div_inplace_impl = None  # type: ignore[assignment]
    _fast_neg_inplace_impl = None  # type: ignore[assignment]
    _fast_exp_inplace_impl = None  # type: ignore[assignment]
    _fast_log_inplace_impl = None  # type: ignore[assignment]
    _fast_tan_inplace_impl = None  # type: ignore[assignment]
    _fast_floor_inplace_impl = None  # type: ignore[assignment]
    _fast_ceil_inplace_impl = None  # type: ignore[assignment]
    _fast_sin_inplace_impl = None  # type: ignore[assignment]
    _fast_cos_inplace_impl = None  # type: ignore[assignment]


def add(a, b):
    if isinstance(b, (int, float)):
        b = _scalar_to_npu_tensor(b, a)
    if _HAS_FAST_ADD:
        return _fast_add_impl(a, b)
    raise RuntimeError("Cython NPU add implementation is unavailable")


def mul(a, b):
    if isinstance(b, (int, float)):
        b = _scalar_to_npu_tensor(b, a)
    if _HAS_FAST_MUL:
        return _fast_mul_impl(a, b)
    raise RuntimeError("Cython NPU mul implementation is unavailable")


def sub(a, b):
    if isinstance(b, (int, float)):
        b = _scalar_to_npu_tensor(b, a)
    if _HAS_FAST_SUB:
        return _fast_sub_impl(a, b)
    raise RuntimeError("Cython NPU sub implementation is unavailable")


def div(a, b):
    if isinstance(b, (int, float)):
        b = _scalar_to_npu_tensor(b, a)
    if _HAS_FAST_DIV:
        return _fast_div_impl(a, b)
    raise RuntimeError("Cython NPU div implementation is unavailable")


# ---------------------------------------------------------------------------
# In-place arithmetic
# ---------------------------------------------------------------------------

def add_(a, b):
    if isinstance(b, (int, float)):
        b = _scalar_to_npu_tensor(b, a)
    if _HAS_FAST_ADD_INPLACE:
        return _fast_add_inplace_impl(a, b)
    raise RuntimeError("Cython NPU add_ implementation is unavailable")


def mul_(a, b):
    if isinstance(b, (int, float)):
        b = _scalar_to_npu_tensor(b, a)
    if _HAS_FAST_MUL_INPLACE:
        return _fast_mul_inplace_impl(a, b)
    raise RuntimeError("Cython NPU mul_ implementation is unavailable")


def sub_(a, b):
    if isinstance(b, (int, float)):
        b = _scalar_to_npu_tensor(b, a)
    if _HAS_FAST_SUB_INPLACE:
        return _fast_sub_inplace_impl(a, b)
    raise RuntimeError("Cython NPU sub_ implementation is unavailable")


def div_(a, b):
    if isinstance(b, (int, float)):
        b = _scalar_to_npu_tensor(b, a)
    if _HAS_FAST_DIV_INPLACE:
        return _fast_div_inplace_impl(a, b)
    raise RuntimeError("Cython NPU div_ implementation is unavailable")


# In-place unary aliases (operator intake tranche 2). Same pattern as
# `erfinv_ = _fast_erfinv_inplace_impl` from PR #508 — direct Cython aliases
# avoid pure-Python call frames on the dispatch hot path.
neg_ = _fast_neg_inplace_impl
exp_ = _fast_exp_inplace_impl
log_ = _fast_log_inplace_impl
tan_ = _fast_tan_inplace_impl
floor_ = _fast_floor_inplace_impl
ceil_ = _fast_ceil_inplace_impl
sin_ = _fast_sin_inplace_impl
cos_ = _fast_cos_inplace_impl


# ---------------------------------------------------------------------------
# Unary math (simple)
# ---------------------------------------------------------------------------

def abs(a):
    if _HAS_FAST_ABS:
        return _fast_abs_impl(a)
    raise RuntimeError("Cython NPU abs implementation is unavailable")


def neg(a):
    if _HAS_FAST_NEG:
        return _fast_neg_impl(a)
    raise RuntimeError("Cython NPU neg implementation is unavailable")


def sign(a):
    if _HAS_FAST_SIGN:
        return _fast_sign_impl(a)
    raise RuntimeError("Cython NPU sign implementation is unavailable")


def signbit(a):
    if _HAS_FAST_SIGNBIT:
        return _fast_signbit_impl(a)
    raise RuntimeError("Cython NPU signbit implementation is unavailable")


def square(a):
    if _HAS_FAST_SQUARE:
        return _fast_square_impl(a)
    raise RuntimeError("Cython NPU square implementation is unavailable")


# ---------------------------------------------------------------------------
# Float classification
# ---------------------------------------------------------------------------

def isfinite(a):
    if _HAS_FAST_ISFINITE:
        return _fast_isfinite_impl(a)
    raise RuntimeError("Cython NPU isfinite implementation is unavailable")


def isinf(a):
    """Check for infinity values.

    When fallback is active (910B): aclnnIsInf returns 161001 (unavailable),
    so the Cython composite (!isfinite(x) & isfinite(1/x)) is used instead.
    """
    if _HAS_FAST_ISINF:
        return _fast_isinf_impl(a)
    raise RuntimeError("Cython NPU isinf implementation is unavailable")


def isnan(a):
    if _HAS_FAST_ISNAN and a.dtype.is_floating_point:
        return _fast_isnan_impl(a)
    if not a.dtype.is_floating_point:
        from . import logical_not
        return logical_not(isfinite(a))
    raise RuntimeError("Cython NPU isnan implementation is unavailable")


def isposinf(a):
    if _HAS_FAST_ISPOSINF:
        return _fast_isposinf_impl(a)
    raise RuntimeError("Cython NPU isposinf implementation is unavailable")


def isneginf(a):
    if _HAS_FAST_ISNEGINF:
        return _fast_isneginf_impl(a)
    raise RuntimeError("Cython NPU isneginf implementation is unavailable")


# ---------------------------------------------------------------------------
# Transcendental
# ---------------------------------------------------------------------------

def exp(a):
    if _HAS_FAST_EXP:
        return _fast_exp_impl(a)
    raise RuntimeError("Cython NPU exp implementation is unavailable")


def log(a):
    if _HAS_FAST_LOG:
        return _fast_log_impl(a)
    raise RuntimeError("Cython NPU log implementation is unavailable")


def expm1(a):
    if _HAS_FAST_EXPM1:
        return _fast_expm1_impl(a)
    raise RuntimeError("Cython NPU expm1 implementation is unavailable")


def log1p(a):
    if _HAS_FAST_LOG1P:
        return _fast_log1p_impl(a)
    raise RuntimeError("Cython NPU log1p implementation is unavailable")


def sqrt(a):
    if _HAS_FAST_SQRT:
        return _fast_sqrt_impl(a)
    raise RuntimeError("Cython NPU sqrt implementation is unavailable")


def rsqrt(a):
    if _HAS_FAST_RSQRT:
        return _fast_rsqrt_impl(a)
    raise RuntimeError("Cython NPU rsqrt implementation is unavailable")


def sin(a):
    if _HAS_FAST_SIN:
        return _fast_sin_impl(a)
    raise RuntimeError("Cython NPU sin implementation is unavailable")


def cos(a):
    if _HAS_FAST_COS:
        return _fast_cos_impl(a)
    raise RuntimeError("Cython NPU cos implementation is unavailable")


def tan(a):
    if _HAS_FAST_TAN:
        return _fast_tan_impl(a)
    raise RuntimeError("Cython NPU tan implementation is unavailable")


def tanh(a):
    if _HAS_FAST_TANH:
        return _fast_tanh_impl(a)
    raise RuntimeError("Cython NPU tanh implementation is unavailable")


def sigmoid(a):
    if _HAS_FAST_SIGMOID:
        return _fast_sigmoid_impl(a)
    raise RuntimeError("Cython NPU sigmoid implementation is unavailable")


def sinh(a):
    if _HAS_FAST_SINH:
        return _fast_sinh_impl(a)
    raise RuntimeError("Cython NPU sinh implementation is unavailable")


def cosh(a):
    if _HAS_FAST_COSH:
        return _fast_cosh_impl(a)
    raise RuntimeError("Cython NPU cosh implementation is unavailable")


def erf(a):
    if _HAS_FAST_ERF:
        return _fast_erf_impl(a)
    raise RuntimeError("Cython NPU erf implementation is unavailable")


def erfc(a):
    if _HAS_FAST_ERFC:
        return _fast_erfc_impl(a)
    raise RuntimeError("Cython NPU erfc implementation is unavailable")


def floor(a):
    if _HAS_FAST_FLOOR:
        return _fast_floor_impl(a)
    raise RuntimeError("Cython NPU floor implementation is unavailable")


def ceil(a):
    if _HAS_FAST_CEIL:
        return _fast_ceil_impl(a)
    raise RuntimeError("Cython NPU ceil implementation is unavailable")


def round(a):
    if _HAS_FAST_ROUND:
        return _fast_round_impl(a)
    raise RuntimeError("Cython NPU round implementation is unavailable")


def trunc(a):
    if _HAS_FAST_TRUNC:
        return _fast_trunc_impl(a)
    raise RuntimeError("Cython NPU trunc implementation is unavailable")


def frac(a):
    if _HAS_FAST_FRAC:
        return _fast_frac_impl(a)
    raise RuntimeError("Cython NPU frac implementation is unavailable")


def log2(a):
    if _HAS_FAST_LOG2:
        return _fast_log2_impl(a)
    raise RuntimeError("Cython NPU log2 implementation is unavailable")


def log10(a):
    if _HAS_FAST_LOG10:
        return _fast_log10_impl(a)
    raise RuntimeError("Cython NPU log10 implementation is unavailable")


def exp2(a):
    if _HAS_FAST_EXP2:
        return _fast_exp2_impl(a)
    raise RuntimeError("Cython NPU exp2 implementation is unavailable")


def asinh(a):
    if _HAS_FAST_ASINH:
        return _fast_asinh_impl(a)
    raise RuntimeError("Cython NPU asinh implementation is unavailable")


def acosh(a):
    if _HAS_FAST_ACOSH:
        return _fast_acosh_impl(a)
    raise RuntimeError("Cython NPU acosh implementation is unavailable")


def atanh(a):
    if _HAS_FAST_ATANH:
        return _fast_atanh_impl(a)
    raise RuntimeError("Cython NPU atanh implementation is unavailable")


def atan(a):
    if _HAS_FAST_ATAN:
        return _fast_atan_impl(a)
    raise RuntimeError("Cython NPU atan implementation is unavailable")


def asin(a):
    if _HAS_FAST_ASIN:
        return _fast_asin_impl(a)
    raise RuntimeError("Cython NPU asin implementation is unavailable")


def acos(a):
    if _HAS_FAST_ACOS:
        return _fast_acos_impl(a)
    raise RuntimeError("Cython NPU acos implementation is unavailable")


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

    if _HAS_FAST_ATAN2:
        return _fast_atan2_impl(a, b)
    raise RuntimeError("Cython NPU atan2 implementation is unavailable")


def reciprocal(a):
    if _HAS_FAST_RECIPROCAL:
        return _fast_reciprocal_impl(a)
    raise RuntimeError("Cython NPU reciprocal implementation is unavailable")


def pow(a, b):
    if hasattr(b, "shape"):
        if _HAS_FAST_POW:
            return _fast_pow_impl(a, b)
        raise RuntimeError("Cython NPU pow implementation is unavailable")
    if _HAS_FAST_POW_TENSOR_SCALAR:
        return _fast_pow_tensor_scalar_impl(a, b)
    raise RuntimeError("Cython NPU pow tensor-scalar implementation is unavailable")


def floor_divide(a, b):
    from ...._tensor import Tensor
    if not isinstance(b, Tensor):
        from ...._creation import tensor as _tensor
        b = _tensor(float(b), device=a.device)
    if _HAS_FAST_FLOOR_DIVIDE:
        return _fast_floor_divide_impl(a, b)
    raise RuntimeError("Cython NPU floor_divide implementation is unavailable")
