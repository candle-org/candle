"""Comparison, logical, and bitwise operations for NPU."""

try:
    from candle._C._npu_ops import (
        fast_logical_not as _fast_logical_not_impl,
        fast_bitwise_not as _fast_bitwise_not_impl,
        fast_isclose as _fast_isclose_impl,
        fast_eq as _fast_eq_impl,
        fast_ne as _fast_ne_impl,
        fast_le as _fast_le_impl,
        fast_lt as _fast_lt_impl,
        fast_gt as _fast_gt_impl,
        fast_ge as _fast_ge_impl,
        fast_logical_and as _fast_logical_and_impl,
        fast_logical_or as _fast_logical_or_impl,
        fast_logical_xor as _fast_logical_xor_impl,
        fast_bitwise_and as _fast_bitwise_and_impl,
        fast_bitwise_or as _fast_bitwise_or_impl,
        fast_bitwise_xor as _fast_bitwise_xor_impl,
        fast_bitwise_left_shift as _fast_bitwise_left_shift_impl,
        fast_bitwise_right_shift as _fast_bitwise_right_shift_impl,
        fast_bitwise_and_inplace as _fast_bitwise_and_inplace_impl,
        fast_bitwise_or_inplace as _fast_bitwise_or_inplace_impl,
        fast_bitwise_xor_inplace as _fast_bitwise_xor_inplace_impl,
        fast_bitwise_not_inplace as _fast_bitwise_not_inplace_impl,
    )  # pylint: disable=import-error,no-name-in-module
    _HAS_FAST_LOGICAL_NOT = True
    _HAS_FAST_BITWISE_NOT = True
    _HAS_FAST_ISCLOSE = True
    _HAS_FAST_EQ = True
    _HAS_FAST_NE = True
    _HAS_FAST_LE = True
    _HAS_FAST_LT = True
    _HAS_FAST_GT = True
    _HAS_FAST_GE = True
    _HAS_FAST_LOGICAL_AND = True
    _HAS_FAST_LOGICAL_OR = True
    _HAS_FAST_LOGICAL_XOR = True
    _HAS_FAST_BITWISE_AND = True
    _HAS_FAST_BITWISE_OR = True
    _HAS_FAST_BITWISE_XOR = True
    _HAS_FAST_BITWISE_LEFT_SHIFT = True
    _HAS_FAST_BITWISE_RIGHT_SHIFT = True
    _HAS_FAST_BITWISE_AND_INPLACE = True
    _HAS_FAST_BITWISE_OR_INPLACE = True
    _HAS_FAST_BITWISE_XOR_INPLACE = True
    _HAS_FAST_BITWISE_NOT_INPLACE = True
except ImportError:
    _fast_logical_not_impl = None  # type: ignore[assignment]
    _fast_bitwise_not_impl = None  # type: ignore[assignment]
    _fast_isclose_impl = None  # type: ignore[assignment]
    _fast_eq_impl = None  # type: ignore[assignment]
    _fast_ne_impl = None  # type: ignore[assignment]
    _fast_le_impl = None  # type: ignore[assignment]
    _fast_lt_impl = None  # type: ignore[assignment]
    _fast_gt_impl = None  # type: ignore[assignment]
    _fast_ge_impl = None  # type: ignore[assignment]
    _fast_logical_and_impl = None  # type: ignore[assignment]
    _fast_logical_or_impl = None  # type: ignore[assignment]
    _fast_logical_xor_impl = None  # type: ignore[assignment]
    _fast_bitwise_and_impl = None  # type: ignore[assignment]
    _fast_bitwise_or_impl = None  # type: ignore[assignment]
    _fast_bitwise_xor_impl = None  # type: ignore[assignment]
    _fast_bitwise_left_shift_impl = None  # type: ignore[assignment]
    _fast_bitwise_right_shift_impl = None  # type: ignore[assignment]
    _fast_bitwise_and_inplace_impl = None  # type: ignore[assignment]
    _fast_bitwise_or_inplace_impl = None  # type: ignore[assignment]
    _fast_bitwise_xor_inplace_impl = None  # type: ignore[assignment]
    _HAS_FAST_LOGICAL_NOT = False
    _HAS_FAST_BITWISE_NOT = False
    _HAS_FAST_ISCLOSE = False
    _HAS_FAST_EQ = False
    _HAS_FAST_NE = False
    _HAS_FAST_LE = False
    _HAS_FAST_LT = False
    _HAS_FAST_GT = False
    _HAS_FAST_GE = False
    _HAS_FAST_LOGICAL_AND = False
    _HAS_FAST_LOGICAL_OR = False
    _HAS_FAST_LOGICAL_XOR = False
    _HAS_FAST_BITWISE_AND = False
    _HAS_FAST_BITWISE_OR = False
    _HAS_FAST_BITWISE_XOR = False
    _HAS_FAST_BITWISE_LEFT_SHIFT = False
    _HAS_FAST_BITWISE_RIGHT_SHIFT = False
    _HAS_FAST_BITWISE_AND_INPLACE = False
    _HAS_FAST_BITWISE_OR_INPLACE = False
    _HAS_FAST_BITWISE_XOR_INPLACE = False

from ._helpers import (
    _unwrap_storage, _wrap_tensor,
    _broadcast_shape, _numel, _dtype_itemsize, _use_soc_fallback,
    _scalar_to_npu_tensor,
    bool_dtype,
    npu_typed_storage_from_ptr,
    aclnn, npu_runtime, npu_state,
)


def eq(a, b):
    if isinstance(b, (int, float, bool)):
        b = _scalar_to_npu_tensor(b, a)
    if _HAS_FAST_EQ:
        return _fast_eq_impl(a, b)
    raise RuntimeError("Cython NPU eq implementation is unavailable")


def ne(a, b):
    if isinstance(b, (int, float, bool)):
        b = _scalar_to_npu_tensor(b, a)
    if _HAS_FAST_NE:
        return _fast_ne_impl(a, b)
    raise RuntimeError("Cython NPU ne implementation is unavailable")


def le(a, b):
    if isinstance(b, (int, float, bool)):
        b = _scalar_to_npu_tensor(b, a)
    if _HAS_FAST_LE:
        return _fast_le_impl(a, b)
    raise RuntimeError("Cython NPU le implementation is unavailable")


def lt(a, b):
    if isinstance(b, (int, float, bool)):
        b = _scalar_to_npu_tensor(b, a)
    if _HAS_FAST_LT:
        return _fast_lt_impl(a, b)
    raise RuntimeError("Cython NPU lt implementation is unavailable")


def gt(a, b):
    if isinstance(b, (int, float, bool)):
        b = _scalar_to_npu_tensor(b, a)
    if _HAS_FAST_GT:
        return _fast_gt_impl(a, b)
    raise RuntimeError("Cython NPU gt implementation is unavailable")


def ge(a, b):
    if isinstance(b, (int, float, bool)):
        b = _scalar_to_npu_tensor(b, a)
    if _HAS_FAST_GE:
        return _fast_ge_impl(a, b)
    raise RuntimeError("Cython NPU ge implementation is unavailable")


def logical_and(a, b):
    if _HAS_FAST_LOGICAL_AND:
        return _fast_logical_and_impl(a, b)
    raise RuntimeError("Cython NPU logical_and implementation is unavailable")


def logical_or(a, b):
    if _HAS_FAST_LOGICAL_OR:
        return _fast_logical_or_impl(a, b)
    raise RuntimeError("Cython NPU logical_or implementation is unavailable")


def logical_not(a):
    if _HAS_FAST_LOGICAL_NOT:
        return _fast_logical_not_impl(a)
    raise RuntimeError("Cython NPU logical_not implementation is unavailable")


def logical_xor(a, b):
    if isinstance(b, (int, float, bool)):
        b = _scalar_to_npu_tensor(b, a)
    if _HAS_FAST_LOGICAL_XOR:
        return _fast_logical_xor_impl(a, b)
    raise RuntimeError("Cython NPU logical_xor implementation is unavailable")


# Bitwise operations
def bitwise_not(a):
    if _HAS_FAST_BITWISE_NOT:
        return _fast_bitwise_not_impl(a)
    raise RuntimeError("Cython NPU bitwise_not implementation is unavailable")


def bitwise_and(a, b):
    if _HAS_FAST_BITWISE_AND:
        return _fast_bitwise_and_impl(a, b)
    raise RuntimeError("Cython NPU bitwise_and implementation is unavailable")


def bitwise_or(a, b):
    if _HAS_FAST_BITWISE_OR:
        return _fast_bitwise_or_impl(a, b)
    raise RuntimeError("Cython NPU bitwise_or implementation is unavailable")


def bitwise_xor(a, b):
    if _HAS_FAST_BITWISE_XOR:
        return _fast_bitwise_xor_impl(a, b)
    raise RuntimeError("Cython NPU bitwise_xor implementation is unavailable")


def bitwise_left_shift(a, b):
    if isinstance(b, (int, float)):
        b = _scalar_to_npu_tensor(b, a)
    if _HAS_FAST_BITWISE_LEFT_SHIFT:
        return _fast_bitwise_left_shift_impl(a, b)
    raise RuntimeError("Cython NPU bitwise_left_shift implementation is unavailable")


def bitwise_right_shift(a, b):
    if isinstance(b, (int, float)):
        b = _scalar_to_npu_tensor(b, a)
    if _HAS_FAST_BITWISE_RIGHT_SHIFT:
        return _fast_bitwise_right_shift_impl(a, b)
    raise RuntimeError("Cython NPU bitwise_right_shift implementation is unavailable")


bitwise_and_ = _fast_bitwise_and_inplace_impl
bitwise_or_ = _fast_bitwise_or_inplace_impl
bitwise_xor_ = _fast_bitwise_xor_inplace_impl
bitwise_not_ = _fast_bitwise_not_inplace_impl


def equal(a, b):
    if a.shape != b.shape:
        return False
    if a.dtype != b.dtype:
        return False
    if b.device.type != "npu":
        raise ValueError("NPU equal expects NPU tensors")
    neq = ne(a, b)
    # any_ is in __init__.py; use lazy import to avoid circular dependency
    from . import any_
    return logical_not(any_(neq)).item()


def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    from ...._tensor import Tensor
    if not isinstance(a, Tensor) or not isinstance(b, Tensor):
        raise ValueError("NPU allclose expects tensors")
    if _use_soc_fallback("allclose"):
        raise RuntimeError("NPU allclose requires on-device comparison support")
    from .math import abs, sub, mul, add
    from .math import isnan
    diff = abs(sub(a, b))
    tol = add(_scalar_to_npu_tensor(atol, diff), mul(_scalar_to_npu_tensor(rtol, diff), abs(b)))
    close = le(diff, tol)
    if equal_nan:
        nan_match = logical_and(isnan(a), isnan(b))
        close = logical_or(close, nan_match)
    from . import all_
    return all_(close).item()


def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    if _HAS_FAST_ISCLOSE and _use_soc_fallback("isclose"):
        if isinstance(b, (int, float, bool)):
            b = _scalar_to_npu_tensor(b, a)
        return _fast_isclose_impl(a, b, float(rtol), float(atol), bool(equal_nan))
    from .math import abs, sub, mul, add
    from .math import isnan
    if isinstance(b, (int, float, bool)):
        b = _scalar_to_npu_tensor(b, a)

    if _use_soc_fallback("isclose"):
        diff = abs(sub(a, b))
        tol = add(_scalar_to_npu_tensor(float(atol), diff), mul(_scalar_to_npu_tensor(float(rtol), diff), abs(b)))
        close = le(diff, tol)
        if equal_nan:
            nan_both = logical_and(isnan(a), isnan(b))
            close = logical_or(close, nan_both)
        else:
            nan_any = logical_or(isnan(a), isnan(b))
            close = logical_and(close, logical_not(nan_any))
        return close

    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    out_shape = _broadcast_shape(a.shape, b.shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_ptr = npu_runtime._alloc_device(_numel(out_shape) * _dtype_itemsize(bool_dtype), runtime=runtime)
    aclnn.sisclose(
        _unwrap_storage(a).data_ptr(),
        _unwrap_storage(b).data_ptr(),
        out_ptr,
        a.shape, a.stride, b.shape, b.stride,
        out_shape, out_stride, a.dtype,
        float(rtol), float(atol), True,  # ACLNN ignores equal_nan, always pass True
        runtime, stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), bool_dtype, device=a.device)
    result = _wrap_tensor(out_storage, out_shape, out_stride)
    if not equal_nan:
        # ACLNN always treats NaN==NaN as True; mask out when equal_nan=False
        nan_both = logical_and(isnan(a), isnan(b))
        result = logical_and(result, logical_not(nan_both))
    return result
