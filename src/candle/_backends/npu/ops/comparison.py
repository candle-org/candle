"""Comparison, logical, and bitwise operations for NPU."""

from ._helpers import (
    _unwrap_storage, _wrap_tensor, _unary_op, _binary_op,
    _broadcast_shape, _numel, _dtype_itemsize, _use_soc_fallback,
    _scalar_to_npu_tensor,
    bool_dtype,
    npu_typed_storage_from_ptr,
    aclnn, npu_runtime, npu_state,
)


def eq(a, b):
    if isinstance(b, (int, float, bool)):
        b = _scalar_to_npu_tensor(b, a)
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if not aclnn.eq_tensor_symbols_ok():
        raise RuntimeError("aclnnEqTensor symbols not available")
    out_shape = _broadcast_shape(a.shape, b.shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_ptr = npu_runtime._alloc_device(_numel(out_shape) * _dtype_itemsize(bool_dtype), runtime=runtime)
    aclnn.eq_tensor(
        _unwrap_storage(a).data_ptr(),
        _unwrap_storage(b).data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        b.shape,
        b.stride,
        out_shape,
        out_stride,
        a.dtype,
        runtime,
        stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), bool_dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def ne(a, b):
    if isinstance(b, (int, float, bool)):
        b = _scalar_to_npu_tensor(b, a)
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if not aclnn.ne_tensor_symbols_ok():
        raise RuntimeError("aclnnNeTensor symbols not available")
    out_shape = _broadcast_shape(a.shape, b.shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_ptr = npu_runtime._alloc_device(_numel(out_shape) * _dtype_itemsize(bool_dtype), runtime=runtime)
    aclnn.ne_tensor(
        _unwrap_storage(a).data_ptr(),
        _unwrap_storage(b).data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        b.shape,
        b.stride,
        out_shape,
        out_stride,
        a.dtype,
        runtime,
        stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), bool_dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def le(a, b):
    if isinstance(b, (int, float, bool)):
        b = _scalar_to_npu_tensor(b, a)
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    out_shape = _broadcast_shape(a.shape, b.shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_ptr = npu_runtime._alloc_device(_numel(out_shape) * _dtype_itemsize(bool_dtype), runtime=runtime)
    aclnn.le_tensor(
        _unwrap_storage(a).data_ptr(), _unwrap_storage(b).data_ptr(), out_ptr,
        a.shape, a.stride, b.shape, b.stride,
        out_shape, out_stride, a.dtype, runtime, stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), bool_dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def lt(a, b):
    if isinstance(b, (int, float, bool)):
        b = _scalar_to_npu_tensor(b, a)
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    out_shape = _broadcast_shape(a.shape, b.shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_ptr = npu_runtime._alloc_device(_numel(out_shape) * _dtype_itemsize(bool_dtype), runtime=runtime)
    aclnn.lt_tensor(
        _unwrap_storage(a).data_ptr(), _unwrap_storage(b).data_ptr(), out_ptr,
        a.shape, a.stride, b.shape, b.stride,
        out_shape, out_stride, a.dtype, runtime, stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), bool_dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def gt(a, b):
    if isinstance(b, (int, float, bool)):
        b = _scalar_to_npu_tensor(b, a)
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    out_shape = _broadcast_shape(a.shape, b.shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_ptr = npu_runtime._alloc_device(_numel(out_shape) * _dtype_itemsize(bool_dtype), runtime=runtime)
    aclnn.gt_tensor(
        _unwrap_storage(a).data_ptr(), _unwrap_storage(b).data_ptr(), out_ptr,
        a.shape, a.stride, b.shape, b.stride,
        out_shape, out_stride, a.dtype, runtime, stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), bool_dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def ge(a, b):
    if isinstance(b, (int, float, bool)):
        b = _scalar_to_npu_tensor(b, a)
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    out_shape = _broadcast_shape(a.shape, b.shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_ptr = npu_runtime._alloc_device(_numel(out_shape) * _dtype_itemsize(bool_dtype), runtime=runtime)
    aclnn.ge_tensor(
        _unwrap_storage(a).data_ptr(), _unwrap_storage(b).data_ptr(), out_ptr,
        a.shape, a.stride, b.shape, b.stride,
        out_shape, out_stride, a.dtype, runtime, stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), bool_dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def logical_and(a, b):
    return _binary_op(a, b, aclnn.logical_and, "logical_and")


def logical_or(a, b):
    return _binary_op(a, b, aclnn.logical_or, "logical_or")


def logical_not(a):
    return _unary_op(a, aclnn.logical_not, "logical_not", out_dtype=bool_dtype)


def logical_xor(a, b):
    if isinstance(b, (int, float, bool)):
        b = _scalar_to_npu_tensor(b, a)
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    out_shape = _broadcast_shape(a.shape, b.shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_ptr = npu_runtime._alloc_device(_numel(out_shape) * _dtype_itemsize(bool_dtype), runtime=runtime)
    aclnn.logical_xor(
        _unwrap_storage(a).data_ptr(),
        _unwrap_storage(b).data_ptr(),
        out_ptr,
        a.shape, a.stride, b.shape, b.stride,
        out_shape, out_stride, a.dtype,
        runtime, stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), bool_dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


# Bitwise operations
def bitwise_not(a):
    if not aclnn.bitwise_not_symbols_ok():
        raise RuntimeError("aclnnBitwiseNot symbols not available")
    return _unary_op(a, aclnn.bitwise_not, "bitwise_not")


def bitwise_and(a, b):
    if not aclnn.bitwise_and_symbols_ok():
        raise RuntimeError("aclnnBitwiseAndTensor symbols not available")
    return _binary_op(a, b, aclnn.bitwise_and, "bitwise_and")


def bitwise_or(a, b):
    if not aclnn.bitwise_or_symbols_ok():
        raise RuntimeError("aclnnBitwiseOrTensor symbols not available")
    return _binary_op(a, b, aclnn.bitwise_or, "bitwise_or")


def bitwise_xor(a, b):
    if not aclnn.bitwise_xor_symbols_ok():
        raise RuntimeError("aclnnBitwiseXorTensor symbols not available")
    return _binary_op(a, b, aclnn.bitwise_xor, "bitwise_xor")


def equal(a, b):
    if a.shape != b.shape:
        return False
    if a.dtype != b.dtype:
        return False
    if a.device.type != "npu" or b.device.type != "npu":
        raise ValueError("NPU equal expects NPU tensors")
    neq = ne(a, b)
    # any_ is in __init__.py; use lazy import to avoid circular dependency
    from . import any_
    return logical_not(any_(neq)).item()


def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    from ...._tensor import Tensor
    from .math import abs, sub, mul, add
    from .math import isnan
    if not isinstance(a, Tensor) or not isinstance(b, Tensor):
        raise ValueError("NPU allclose expects tensors")
    diff = abs(sub(a, b))
    tol = add(_scalar_to_npu_tensor(atol, diff), mul(_scalar_to_npu_tensor(rtol, diff), abs(b)))
    close = le(diff, tol)
    if equal_nan:
        nan_match = logical_and(isnan(a), isnan(b))
        close = logical_or(close, nan_match)
    # all_ is in __init__.py; use lazy import to avoid circular dependency
    from . import all_
    return all_(close).item()


def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
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
