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

def add(a, b):
    # For commutative add, ensure the larger tensor is 'a' so the GPU
    # dispatch allocates the correct output size and shape.
    if isinstance(b, Tensor) and _can_use_gpu(b):
        if not _can_use_gpu(a) or a.numel() < b.numel():
            a, b = b, a
    if _can_use_gpu(a):
        return _dispatch_binary_gpu(a, b, "add")
    a_np = _to_numpy(a)
    b_np = _to_numpy(b) if isinstance(b, Tensor) else b
    return _from_numpy(a_np + b_np, a.dtype, a.device)

def mul(a, b):
    # For commutative mul, ensure the larger tensor is 'a' so the GPU
    # dispatch allocates the correct output size and shape.
    if isinstance(b, Tensor) and _can_use_gpu(b):
        if not _can_use_gpu(a) or a.numel() < b.numel():
            a, b = b, a
    if _can_use_gpu(a):
        return _dispatch_binary_gpu(a, b, "mul")
    a_np = _to_numpy(a)
    b_np = _to_numpy(b) if isinstance(b, Tensor) else b
    return _from_numpy(a_np * b_np, a.dtype, a.device)

def div(a, b):
    if _can_use_gpu(a):
        return _dispatch_binary_gpu(a, b, "div")
    a_np = _to_numpy(a)
    b_np = _to_numpy(b) if isinstance(b, Tensor) else b
    out = np.true_divide(a_np, b_np)
    return _from_numpy(out.astype(to_numpy_dtype(a.dtype), copy=False), a.dtype, a.device)

def true_divide(a, b):
    return div(a, b)

def sub(a, b):
    if _can_use_gpu(a):
        return _dispatch_binary_gpu(a, b, "sub")
    a_np = _to_numpy(a)
    b_np = _to_numpy(b) if isinstance(b, Tensor) else b
    return _from_numpy(a_np - b_np, a.dtype, a.device)

def abs(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "abs")
    return _from_numpy(np.abs(_to_numpy(a)), a.dtype, a.device)

def neg(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "neg")
    return _from_numpy(np.negative(_to_numpy(a)), a.dtype, a.device)

def exp(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "exp")
    return _from_numpy(np.exp(_to_numpy(a)), a.dtype, a.device)

def log(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "log")
    return _from_numpy(np.log(_to_numpy(a)), a.dtype, a.device)

def sqrt(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "sqrt")
    return _from_numpy(np.sqrt(_to_numpy(a)), a.dtype, a.device)

def sin(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "sin")
    return _from_numpy(np.sin(_to_numpy(a)), a.dtype, a.device)

def cos(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "cos")
    return _from_numpy(np.cos(_to_numpy(a)), a.dtype, a.device)

def tan(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "tan")
    return _from_numpy(np.tan(_to_numpy(a)), a.dtype, a.device)

def tanh(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "tanh")
    return _from_numpy(np.tanh(_to_numpy(a)), a.dtype, a.device)

def sigmoid(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "sigmoid")
    arr = _to_numpy(a)
    out = 1.0 / (1.0 + np.exp(-arr))
    return _from_numpy(out, a.dtype, a.device)

def floor(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "floor")
    return _from_numpy(np.floor(_to_numpy(a)), a.dtype, a.device)

def ceil(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "ceil")
    return _from_numpy(np.ceil(_to_numpy(a)), a.dtype, a.device)

def round(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "round")
    return _from_numpy(np.round(_to_numpy(a)), a.dtype, a.device)

def trunc(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "trunc")
    return _from_numpy(np.trunc(_to_numpy(a)), a.dtype, a.device)

def frac(a):
    if _can_use_gpu(a):
        return sub(a, _dispatch_unary_gpu(a, "trunc"))
    arr = _to_numpy(a)
    out = arr - np.trunc(arr)
    return _from_numpy(out, a.dtype, a.device)

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
    if isinstance(a, Tensor):
        arr_a = _to_numpy(a)
        ref = a
    else:
        arr_a = a
        ref = b
    if isinstance(b, Tensor):
        arr_b = _to_numpy(b)
    else:
        arr_b = b
    return _from_numpy(np.power(arr_a, arr_b), ref.dtype, ref.device)

def log2(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "log2")
    return _from_numpy(np.log2(_to_numpy(a)), a.dtype, a.device)

def log10(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "log10")
    return _from_numpy(np.log10(_to_numpy(a)), a.dtype, a.device)

def exp2(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "exp2")
    return _from_numpy(np.exp2(_to_numpy(a)), a.dtype, a.device)

def rsqrt(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "rsqrt")
    arr = _to_numpy(a)
    out = 1.0 / np.sqrt(arr)
    return _from_numpy(out, a.dtype, a.device)

def sign(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "sign")
    return _from_numpy(np.sign(_to_numpy(a)), a.dtype, a.device)

def square(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "square")
    arr = _to_numpy(a)
    out = np.square(arr)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)

def signbit(a):
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        return _dispatch_unary_predicate_gpu(a, "signbit")
    arr = np.signbit(_to_numpy(a))
    return _from_numpy(arr, bool_dtype, a.device)

def isnan(a):
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        return _dispatch_unary_predicate_gpu(a, "isnan")
    arr = np.isnan(_to_numpy(a))
    return _from_numpy(arr, bool_dtype, a.device)

def isinf(a):
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        return _dispatch_unary_predicate_gpu(a, "isinf")
    arr = np.isinf(_to_numpy(a))
    return _from_numpy(arr, bool_dtype, a.device)

def isfinite(a):
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        return _dispatch_unary_predicate_gpu(a, "isfinite")
    arr = np.isfinite(_to_numpy(a))
    return _from_numpy(arr, bool_dtype, a.device)

def isneginf(a):
    """Returns a bool tensor indicating negative infinity."""
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        return _dispatch_unary_predicate_gpu(a, "isneginf")
    arr = _to_numpy(a)
    out = np.isneginf(arr)
    return _from_numpy(np.ascontiguousarray(out), bool_dtype, a.device)

def isposinf(a):
    """Returns a bool tensor indicating positive infinity."""
    if _can_use_gpu(a) and a.dtype in (float32_dtype, float16_dtype):
        return _dispatch_unary_predicate_gpu(a, "isposinf")
    arr = _to_numpy(a)
    out = np.isposinf(arr)
    return _from_numpy(np.ascontiguousarray(out), bool_dtype, a.device)

def isreal(a):
    """Returns a bool tensor indicating real-valued elements."""
    arr = _to_numpy(a)
    out = np.isreal(arr)
    if out.ndim == 0:
        out = np.array(out)
    return _from_numpy(np.ascontiguousarray(out.astype(np.bool_)), bool_dtype, a.device)

def sinh(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "sinh")
    return _from_numpy(np.sinh(_to_numpy(a)), a.dtype, a.device)

def cosh(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "cosh")
    return _from_numpy(np.cosh(_to_numpy(a)), a.dtype, a.device)

def asinh(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "asinh")
    return _from_numpy(np.arcsinh(_to_numpy(a)), a.dtype, a.device)

def acosh(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "acosh")
    return _from_numpy(np.arccosh(_to_numpy(a)), a.dtype, a.device)

def atanh(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "atanh")
    return _from_numpy(np.arctanh(_to_numpy(a)), a.dtype, a.device)

def erf(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "erf")
    arr = _to_numpy(a)
    out = np.vectorize(math.erf)(arr)
    return _from_numpy(out, a.dtype, a.device)

def erfc(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "erfc")
    arr = _to_numpy(a)
    out = np.vectorize(math.erfc)(arr)
    return _from_numpy(out, a.dtype, a.device)

def atan(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "atan")
    return _from_numpy(np.arctan(_to_numpy(a)), a.dtype, a.device)

def atan2(a, b):
    if isinstance(a, Tensor) and isinstance(b, Tensor) and _can_use_gpu(a) and _can_use_gpu(b):
        return _dispatch_binary_gpu(a, b, "atan2")
    return _from_numpy(np.arctan2(_to_numpy(a), _to_numpy(b)), a.dtype, a.device)

def asin(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "asin")
    return _from_numpy(np.arcsin(_to_numpy(a)), a.dtype, a.device)

def acos(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "acos")
    return _from_numpy(np.arccos(_to_numpy(a)), a.dtype, a.device)

def floor_divide(a, b):
    if _can_use_gpu(a):
        return _dispatch_binary_gpu(a, b, "floor_divide")
    a_np = _to_numpy(a)
    b_np = _to_numpy(b) if isinstance(b, Tensor) else b
    out = np.floor_divide(a_np, b_np)
    return _from_numpy(out.astype(to_numpy_dtype(a.dtype), copy=False), a.dtype, a.device)

def log1p(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "log1p")
    return _from_numpy(np.log1p(_to_numpy(a)), a.dtype, a.device)

def expm1(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "expm1")
    return _from_numpy(np.expm1(_to_numpy(a)), a.dtype, a.device)

def reciprocal(a):
    if _can_use_gpu(a):
        return _dispatch_unary_gpu(a, "reciprocal")
    return _from_numpy(1.0 / _to_numpy(a), a.dtype, a.device)

