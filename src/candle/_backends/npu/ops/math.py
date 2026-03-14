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

def add(a, b):
    if isinstance(b, (int, float)):
        b = _scalar_to_npu_tensor(b, a)
    return _binary_op(a, b, aclnn.add, "add")


def mul(a, b):
    if isinstance(b, (int, float)):
        b = _scalar_to_npu_tensor(b, a)
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
    return _unary_op(a, aclnn.abs, "abs")


def neg(a):
    return _unary_op(a, aclnn.neg, "neg")


def sign(a):
    return _unary_op(a, aclnn.sign, "sign")


def signbit(a):
    return _unary_op(a, aclnn.signbit, "signbit", out_dtype=bool_dtype)


def square(a):
    if aclnn.square_symbols_ok():
        try:
            return _unary_op(a, aclnn.square, "square")
        except RuntimeError:
            pass
    return mul(a, a)


# ---------------------------------------------------------------------------
# Float classification
# ---------------------------------------------------------------------------

def isfinite(a):
    return _unary_op(a, aclnn.isfinite, "isfinite", out_dtype=bool_dtype)


def isinf(a):
    """Check for infinity values.

    When fallback is active (910B): aclnnIsInf returns 161001 (unavailable),
    so we use composite: !isfinite(x) & isfinite(1/x).
    """
    # Lazy import to avoid circular dependency with comparison/logical ops
    from . import logical_and, gt, lt

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
        aclnn.logical_not(
            _unwrap_storage(isfinite(a)).data_ptr(),
            out_ptr,
            out_shape,
            out_stride,
            bool_dtype,
            runtime,
            stream=stream.stream,
        )
        out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), bool_dtype, device=a.device)
        return _wrap_tensor(out_storage, out_shape, out_stride)
    if _use_soc_fallback("isinf"):
        # Composite: !isfinite(x) & isfinite(1/x)
        if not (aclnn.logical_not_symbols_ok() and aclnn.logical_and_symbols_ok()):
            raise RuntimeError("aclnnIsInf unavailable and logical ops missing")
        finite = isfinite(a)
        recip = pow(a, -1.0)
        recip_finite = isfinite(recip)
        tmp_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
        aclnn.logical_not(
            _unwrap_storage(finite).data_ptr(),
            tmp_ptr,
            out_shape,
            out_stride,
            bool_dtype,
            runtime,
            stream=stream.stream,
        )
        tmp_bool_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
        aclnn.logical_and(
            tmp_ptr,
            _unwrap_storage(recip_finite).data_ptr(),
            tmp_bool_ptr,
            out_shape,
            out_stride,
            out_shape,
            out_stride,
            out_shape,
            out_stride,
            bool_dtype,
            runtime,
            stream=stream.stream,
        )
        runtime.defer_free(tmp_ptr)
        runtime.defer_free(out_ptr)  # free the unused initial allocation
        out_ptr = tmp_bool_ptr
    else:
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
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU isnan expects NPU tensors")
    out_shape = a.shape
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_size = _numel(out_shape) * _dtype_itemsize(bool_dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    if not a.dtype.is_floating_point:
        aclnn.logical_not(
            _unwrap_storage(isfinite(a)).data_ptr(),
            out_ptr,
            out_shape,
            out_stride,
            bool_dtype,
            runtime,
            stream=stream.stream,
        )
        out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), bool_dtype, device=a.device)
        return _wrap_tensor(out_storage, out_shape, out_stride)
    if not (aclnn.logical_not_symbols_ok() and aclnn.logical_and_symbols_ok()):
        raise RuntimeError("aclnn logical ops missing for isnan")
    finite = isfinite(a)
    recip = pow(a, -1.0)
    recip_finite = isfinite(recip)
    tmp_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    aclnn.logical_not(
        _unwrap_storage(finite).data_ptr(),
        tmp_ptr,
        out_shape,
        out_stride,
        bool_dtype,
        runtime,
        stream=stream.stream,
    )
    aclnn.logical_not(
        _unwrap_storage(recip_finite).data_ptr(),
        out_ptr,
        out_shape,
        out_stride,
        bool_dtype,
        runtime,
        stream=stream.stream,
    )
    aclnn.logical_and(
        tmp_ptr,
        out_ptr,
        out_ptr,
        out_shape,
        out_stride,
        out_shape,
        out_stride,
        out_shape,
        out_stride,
        bool_dtype,
        runtime,
        stream=stream.stream,
    )
    runtime.defer_free(tmp_ptr)
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), bool_dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def isposinf(a):
    # Lazy import to avoid circular dependency with comparison/logical ops
    from . import logical_and, gt

    if aclnn.isposinf_symbols_ok():
        try:
            return _unary_op(a, aclnn.isposinf, "isposinf", out_dtype=bool_dtype)
        except RuntimeError:
            pass
    return logical_and(isinf(a), gt(a, _scalar_to_npu_tensor(0, a)))


def isneginf(a):
    # Lazy import to avoid circular dependency with comparison/logical ops
    from . import logical_and, lt

    if aclnn.isneginf_symbols_ok():
        try:
            return _unary_op(a, aclnn.isneginf, "isneginf", out_dtype=bool_dtype)
        except RuntimeError:
            pass
    return logical_and(isinf(a), lt(a, _scalar_to_npu_tensor(0, a)))


# ---------------------------------------------------------------------------
# Transcendental
# ---------------------------------------------------------------------------

def exp(a):
    return _unary_op(a, aclnn.exp, "exp")


def log(a):
    return _unary_op(a, aclnn.log, "log")


def expm1(a):
    if not aclnn.expm1_symbols_ok():
        raise RuntimeError("aclnnExpm1 symbols not available")
    return _unary_op(a, aclnn.expm1, "expm1")


def log1p(a):
    if not aclnn.log1p_symbols_ok():
        raise RuntimeError("aclnnLog1p symbols not available")
    return _unary_op(a, aclnn.log1p, "log1p")


def sqrt(a):
    return _unary_op(a, aclnn.sqrt, "sqrt")


def rsqrt(a):
    return _unary_op(a, aclnn.rsqrt, "rsqrt")


def sin(a):
    return _unary_op(a, aclnn.sin, "sin")


def cos(a):
    return _unary_op(a, aclnn.cos, "cos")


def tan(a):
    return _unary_op(a, aclnn.tan, "tan")


def tanh(a):
    return _unary_op(a, aclnn.tanh, "tanh")


def sigmoid(a):
    return _unary_op(a, aclnn.sigmoid, "sigmoid")


def sinh(a):
    return _unary_op(a, aclnn.sinh, "sinh")


def cosh(a):
    return _unary_op(a, aclnn.cosh, "cosh")


def erf(a):
    return _unary_op(a, aclnn.erf, "erf")


def erfc(a):
    return _unary_op(a, aclnn.erfc, "erfc")


def floor(a):
    return _unary_op(a, aclnn.floor, "floor")


def ceil(a):
    return _unary_op(a, aclnn.ceil, "ceil")


def round(a):
    return _unary_op(a, aclnn.round, "round")


def trunc(a):
    try:
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
    try:
        return _unary_op(a, aclnn.frac, "frac")
    except RuntimeError as exc:
        if "561103" not in str(exc):
            raise
    out = trunc(a)
    return add(a, neg(out))


def log2(a):
    return _unary_op(a, aclnn.log2, "log2")


def log10(a):
    return _unary_op(a, aclnn.log10, "log10")


def exp2(a):
    return _unary_op(a, aclnn.exp2, "exp2")

def asinh(a):
    return _unary_op(a, aclnn.asinh, "asinh")


def acosh(a):
    # Lazy import to avoid circular dependency with comparison/logical ops
    from . import lt, where

    out = _unary_op(a, aclnn.acosh, "acosh")
    if a.dtype.name == "float16":
        return out
    one = _scalar_to_npu_tensor(1, a)
    mask = lt(a, one)
    return where(mask, _nan_like(a), out)


def atanh(a):
    # Lazy import to avoid circular dependency with comparison/logical ops
    from . import ge, where

    out = _unary_op(a, aclnn.atanh, "atanh")
    if a.dtype.name == "float16":
        return out
    one = _scalar_to_npu_tensor(1, a)
    mask = ge(abs(a), one)
    return where(mask, _nan_like(a), out)


def atan(a):
    return _unary_op(a, aclnn.atan, "atan")


def asin(a):
    # Lazy import to avoid circular dependency with comparison/logical ops
    from . import gt, where

    out = _unary_op(a, aclnn.asin, "asin")
    if a.dtype.name == "float16":
        return out
    one = _scalar_to_npu_tensor(1, a)
    mask = gt(abs(a), one)
    return where(mask, _nan_like(a), out)


def acos(a):
    # Lazy import to avoid circular dependency with comparison/logical ops
    from . import gt, where

    out = _unary_op(a, aclnn.acos, "acos")
    if a.dtype.name == "float16":
        return out
    one = _scalar_to_npu_tensor(1, a)
    mask = gt(abs(a), one)
    return where(mask, _nan_like(a), out)


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

    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))

    out_shape = tuple(_broadcast_shape_checked(a.shape, b.shape, "floor_divide"))
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_numel = _numel(out_shape)
    itemsize = _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(max(out_numel, 1) * itemsize, runtime=runtime)

    aclnn.floor_divide(
        _unwrap_storage(a).data_ptr(), _unwrap_storage(b).data_ptr(), out_ptr,
        a.shape, a.stride, b.shape, b.stride,
        out_shape, out_stride, a.dtype,
        runtime, stream=stream.stream,
    )

    out_storage = npu_typed_storage_from_ptr(out_ptr, max(out_numel, 1), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)
