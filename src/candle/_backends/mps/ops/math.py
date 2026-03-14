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

def add(a, b):
    # For commutative add, ensure the larger tensor is 'a' so the GPU
    # dispatch allocates the correct output size and shape.
    if isinstance(b, Tensor) and _can_use_gpu(b):
        if not _can_use_gpu(a) or a.numel() < b.numel():
            a, b = b, a
    if _can_use_gpu(a):
        return _dispatch_binary_gpu(a, b, "add")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("add", a)

def mul(a, b):
    if isinstance(b, Tensor) and _can_use_gpu(b):
        if not _can_use_gpu(a) or a.numel() < b.numel():
            a, b = b, a
    if _can_use_gpu(a):
        return _dispatch_binary_gpu(a, b, "mul")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("mul", a)

def div(a, b):
    if _can_use_gpu(a):
        return _dispatch_binary_gpu(a, b, "div")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("div", a)

def true_divide(a, b):
    return div(a, b)

def sub(a, b):
    if _can_use_gpu(a):
        return _dispatch_binary_gpu(a, b, "sub")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("sub", a)

def abs(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "abs")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("abs", a)

def neg(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "neg")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("neg", a)

def exp(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "exp")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("exp", a)

def log(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "log")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("log", a)

def sqrt(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "sqrt")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("sqrt", a)

def sin(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "sin")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("sin", a)

def cos(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "cos")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("cos", a)

def tan(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "tan")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("tan", a)

def tanh(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "tanh")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("tanh", a)

def sigmoid(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "sigmoid")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("sigmoid", a)

def floor(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "floor")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("floor", a)

def ceil(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "ceil")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("ceil", a)

def round(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "round")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("round", a)

def trunc(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "trunc")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("trunc", a)

def frac(a):
    if _can_use_gpu(a):
        return sub(a, _dispatch_unary_gpu(a, "trunc"))
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("frac", a)

def pow(a, b):
    if isinstance(a, Tensor) and _can_use_gpu(a):
        d = _get_dispatcher()
        sfx = _kernel_suffix(a.dtype)
        numel = a.numel()
        out_buf = _alloc_output_buf(numel, a.dtype)
        if isinstance(b, Tensor) and _can_use_gpu(b):
            d.dispatch_binary(f"pow_{sfx}", _metal_buf(a), _metal_buf(b),
                              out_buf, numel)
        else:
            scalar = float(b) if not isinstance(b, Tensor) else float(_to_numpy(b).ravel()[0])
            d.dispatch_binary_scalar(f"pow_scalar_{sfx}", _metal_buf(a),
                                     scalar, out_buf, numel,
                                     scalar_fmt=_scalar_fmt(a.dtype))
        return _from_metal_buffer(out_buf, a.shape, a.stride, a.dtype, a.device)
    ref = a if isinstance(a, Tensor) else b
    if ref.numel() == 0:
        return _empty_like(ref)
    _unsupported_dtype("pow", ref)

def log2(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "log2")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("log2", a)

def log10(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "log10")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("log10", a)

def exp2(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "exp2")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("exp2", a)

def rsqrt(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "rsqrt")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("rsqrt", a)

def sign(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "sign")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("sign", a)

def square(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "square")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("square", a)

def signbit(a):
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        return _dispatch_unary_predicate_gpu(a, "signbit")
    # Integer types: signbit is always False for unsigned, check sign for signed
    if _can_use_gpu(a) and a.dtype in (int32_dtype, int64_dtype):
        from .comparison import lt
        from ...._tensor import _compute_strides
        zero_buf = _alloc_output_buf(a.numel(), a.dtype)
        return lt(a, 0)
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("signbit", a)

def isnan(a):
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        return _dispatch_unary_predicate_gpu(a, "isnan")
    # Integer/bool types: never NaN
    if _can_use_gpu(a) and a.dtype in (int32_dtype, int64_dtype, bool_dtype):
        out = np.zeros(a.shape, dtype=np.uint8)
        return _from_numpy(out, bool_dtype, a.device)
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("isnan", a)

def isinf(a):
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        return _dispatch_unary_predicate_gpu(a, "isinf")
    # Integer/bool types: never inf
    if _can_use_gpu(a) and a.dtype in (int32_dtype, int64_dtype, bool_dtype):
        out = np.zeros(a.shape, dtype=np.uint8)
        return _from_numpy(out, bool_dtype, a.device)
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("isinf", a)

def isfinite(a):
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        return _dispatch_unary_predicate_gpu(a, "isfinite")
    # Integer/bool types: always finite
    if _can_use_gpu(a) and a.dtype in (int32_dtype, int64_dtype, bool_dtype):
        out = np.ones(a.shape, dtype=np.uint8)
        return _from_numpy(out, bool_dtype, a.device)
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("isfinite", a)

def isneginf(a):
    """Returns a bool tensor indicating negative infinity."""
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        return _dispatch_unary_predicate_gpu(a, "isneginf")
    if _can_use_gpu(a) and a.dtype in (int32_dtype, int64_dtype, bool_dtype):
        out = np.zeros(a.shape, dtype=np.uint8)
        return _from_numpy(out, bool_dtype, a.device)
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("isneginf", a)

def isposinf(a):
    """Returns a bool tensor indicating positive infinity."""
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        return _dispatch_unary_predicate_gpu(a, "isposinf")
    if _can_use_gpu(a) and a.dtype in (int32_dtype, int64_dtype, bool_dtype):
        out = np.zeros(a.shape, dtype=np.uint8)
        return _from_numpy(out, bool_dtype, a.device)
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("isposinf", a)

def isreal(a):
    """Returns a bool tensor indicating real-valued elements."""
    # Candle has no complex dtype support; all tensors are real
    from ...._tensor import _compute_strides
    ones = np.ones(a.shape, dtype=np.uint8)
    return _from_numpy(ones, bool_dtype, a.device)

def sinh(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "sinh")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("sinh", a)

def cosh(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "cosh")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("cosh", a)

def asinh(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "asinh")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("asinh", a)

def acosh(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "acosh")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("acosh", a)

def atanh(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "atanh")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("atanh", a)

def erf(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "erf")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("erf", a)

def erfc(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "erfc")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("erfc", a)

def atan(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "atan")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("atan", a)

def atan2(a, b):
    if isinstance(a, Tensor) and isinstance(b, Tensor) and _can_use_gpu(a) and _can_use_gpu(b):
        return _dispatch_binary_gpu(a, b, "atan2")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("atan2", a)

def asin(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "asin")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("asin", a)

def acos(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "acos")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("acos", a)

def floor_divide(a, b):
    if _can_use_gpu(a):
        return _dispatch_binary_gpu(a, b, "floor_divide")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("floor_divide", a)

def log1p(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "log1p")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("log1p", a)

def expm1(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "expm1")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("expm1", a)

def reciprocal(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "reciprocal")
    if a.numel() == 0:
        return _empty_like(a)
    _unsupported_dtype("reciprocal", a)

