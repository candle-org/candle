"""Shared helper utilities for NPU ops modules."""

import ctypes
from ...._dtype import bool as bool_dtype
from ...._dtype import int32 as int32_dtype
from ...._dtype import int64 as int64_dtype
from ...._dtype import float32 as float_dtype
from ...._storage import npu_typed_storage_from_ptr
from ...common import view as view_backend
reshape = view_backend.reshape
from .. import aclnn
from .. import runtime as npu_runtime
from .. import state as npu_state
from .. import ops_soc

try:
    from .._npu_ops_fast import fast_binary_op as _fast_binary_op  # pylint: disable=no-name-in-module
    _HAS_FAST_OPS = True
except ImportError:
    try:
        from .._npu_ops_fast_fallback import fast_binary_op as _fast_binary_op
        _HAS_FAST_OPS = True
    except ImportError:
        _HAS_FAST_OPS = False


def _unwrap_storage(tensor):
    if tensor.storage().device.type != "npu":
        raise ValueError("Expected NPU storage for NPU op")
    return tensor.storage()


def _wrap_tensor(storage, shape, stride):
    from ...._tensor import Tensor

    return Tensor(storage, shape, stride)


def _broadcast_shape_checked(a_shape, b_shape, name):
    max_len = max(len(a_shape), len(b_shape))
    result = []
    for i in range(1, max_len + 1):
        a_dim = a_shape[-i] if i <= len(a_shape) else 1
        b_dim = b_shape[-i] if i <= len(b_shape) else 1
        if a_dim == 1:
            result.append(b_dim)
        elif b_dim == 1:
            result.append(a_dim)
        elif a_dim == b_dim:
            result.append(a_dim)
        else:
            raise ValueError(f"NPU {name} shape mismatch")
    return tuple(reversed(result))


def _dtype_itemsize(dtype):
    size = getattr(dtype, "itemsize", None)
    if size is not None:
        return int(size)
    name = getattr(dtype, "name", None) or str(dtype).split(".")[-1]
    return {"float16": 2, "float32": 4, "float64": 8, "bfloat16": 2, "int8": 1, "int16": 2,
            "int32": 4, "int64": 8, "uint8": 1, "bool": 1}.get(name, 4)


def _cast_tensor_dtype(a, dst_dtype):
    if a.dtype == dst_dtype:
        return a
    if not aclnn.cast_symbols_ok():
        raise RuntimeError("aclnnCast symbols not available")
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    out_ptr = npu_runtime._alloc_device(_numel(a.shape) * _dtype_itemsize(dst_dtype), runtime=runtime)
    aclnn.cast(
        _unwrap_storage(a).data_ptr(),
        out_ptr,
        a.shape,
        a.stride,
        a.dtype,
        dst_dtype,
        runtime,
        stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(a.shape), dst_dtype, device=a.device)
    return _wrap_tensor(out_storage, a.shape, a.stride)



def _broadcast_shape(a_shape, b_shape):
    max_len = max(len(a_shape), len(b_shape))
    result = []
    for i in range(1, max_len + 1):
        a_dim = a_shape[-i] if i <= len(a_shape) else 1
        b_dim = b_shape[-i] if i <= len(b_shape) else 1
        if a_dim == 1:
            result.append(b_dim)
        elif b_dim == 1:
            result.append(a_dim)
        elif a_dim == b_dim:
            result.append(a_dim)
        else:
            raise ValueError("matmul shape mismatch")
    return tuple(reversed(result))


def _npu_broadcast_to(tensor, shape):
    from ..creation import zeros_create

    shape = tuple(shape)
    if tensor.shape == shape:
        return tensor
    zeros = zeros_create(shape, dtype=tensor.dtype, device=tensor.device)
    return _binary_op(tensor, zeros, aclnn.add, "add")


def _npu_arange_1d(size, device):
    size = int(size)
    shape = (size,)

    if ops_soc.use_smallop_arange_1d():
        from ..creation import empty_create, ones_create
        from . import sub, cumsum  # lazy import from ops package

        if size == 0:
            return empty_create(shape, dtype=int64_dtype, device=device)
        ones = ones_create(shape, dtype=int64_dtype, device=device)
        return sub(cumsum(ones, dim=0), 1)

    runtime = npu_runtime.get_runtime((device.index or 0))
    stream = npu_state.current_stream((device.index or 0))
    if not aclnn.arange_symbols_ok():
        raise RuntimeError("aclnnArange symbols not available")
    stride = npu_runtime._contiguous_stride(shape)
    out_size = _numel(shape) * _dtype_itemsize(int64_dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    aclnn.arange(0, size, 1, out_ptr, shape, stride, int64_dtype, runtime, stream=stream.stream)
    storage = npu_typed_storage_from_ptr(out_ptr, _numel(shape), int64_dtype, device=device)
    return _wrap_tensor(storage, shape, stride)


def _use_soc_fallback(op_name):
    return ops_soc.use_fallback(op_name)


def _npu_add_scalar_(tensor, scalar):
    runtime = npu_runtime.get_runtime((tensor.device.index or 0))
    stream = npu_state.current_stream((tensor.device.index or 0))
    if not aclnn.add_scalar_symbols_ok():
        raise RuntimeError("aclnnAdds symbols not available")
    storage = _unwrap_storage(tensor)
    aclnn.add_scalar(
        storage.data_ptr(),
        scalar,
        storage.data_ptr(),
        tensor.shape,
        tensor.stride,
        tensor.dtype,
        runtime,
        stream=stream.stream,
    )
    return tensor


def _npu_linear_index(view_shape, view_stride, view_offset, device):
    ndim = len(view_shape)
    if ndim == 0:
        runtime = npu_runtime.get_runtime((device.index or 0))
        stream = npu_state.current_stream((device.index or 0))
        out_ptr = npu_runtime._alloc_device(_dtype_itemsize(int64_dtype), runtime=runtime)
        storage = npu_typed_storage_from_ptr(out_ptr, 1, int64_dtype, device=device)
        out = _wrap_tensor(storage, (), ())
        return _npu_add_scalar_(out, view_offset)
    linear = None
    for dim, size in enumerate(view_shape):
        idx = _npu_arange_1d(size, device)
        shape = [1] * ndim
        shape[dim] = int(size)
        idx = idx.reshape(shape)
        target_shape = _broadcast_shape_checked(tuple(shape), tuple(view_shape), "view-index")
        if target_shape != tuple(view_shape):
            raise RuntimeError("NPU view index broadcast mismatch")
        if view_stride[dim] != 1:
            idx = idx * _scalar_to_npu_tensor(view_stride[dim], idx)
        if linear is None:
            linear = idx
        else:
            linear = _binary_op(linear, idx, aclnn.add, "add")
    return _npu_add_scalar_(linear, view_offset)


def npu_index_put_impl(self_tensor, index_tensor, values, accumulate=False, unsafe=False):
    runtime = npu_runtime.get_runtime((self_tensor.device.index or 0))
    stream = npu_state.current_stream((self_tensor.device.index or 0))
    if not aclnn.index_put_impl_symbols_ok():
        raise RuntimeError("aclnnIndexPutImpl symbols not available")
    self_storage = _unwrap_storage(self_tensor)
    index_storage = _unwrap_storage(index_tensor)
    values_storage = _unwrap_storage(values)
    aclnn.index_put_impl(
        self_storage.data_ptr(),
        self_tensor.shape,
        self_tensor.stride,
        self_tensor.dtype,
        [index_storage.data_ptr()],
        [index_tensor.shape],
        [index_tensor.stride],
        [index_tensor.dtype],
        values_storage.data_ptr(),
        values.shape,
        values.stride,
        values.dtype,
        bool(accumulate),
        bool(unsafe),
        runtime,
        stream=stream.stream,
    )


def _numel(shape):
    size = 1
    for dim in shape:
        size *= dim
    return size


def _matmul_out_shape(a_shape, b_shape):
    a_dim = len(a_shape)
    b_dim = len(b_shape)

    if a_dim == 1 and b_dim == 1:
        if a_shape[0] != b_shape[0]:
            raise ValueError("matmul shape mismatch")
        return ()
    if a_dim == 1:
        k = a_shape[0]
        if b_dim < 2 or b_shape[-2] != k:
            raise ValueError("matmul shape mismatch")
        return b_shape[:-2] + (b_shape[-1],)
    if b_dim == 1:
        k = b_shape[0]
        if a_shape[-1] != k:
            raise ValueError("matmul shape mismatch")
        return a_shape[:-2] + (a_shape[-2],)
    if a_shape[-1] != b_shape[-2]:
        raise ValueError("matmul shape mismatch")
    batch = _broadcast_shape(a_shape[:-2], b_shape[:-2])
    return batch + (a_shape[-2], b_shape[-1])


def _normalize_tensor_sequence_args(tensors):
    if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
        return tuple(tensors[0])
    return tuple(tensors)


def _iter_indices(shape):
    if not shape:
        yield ()
        return
    total = 1
    for dim in shape:
        total *= dim
    for flat in range(total):
        idx = []
        rem = flat
        for dim in reversed(shape):
            idx.append(rem % dim)
            rem //= dim
        yield tuple(reversed(idx))


def _broadcast_index(index, shape, out_shape):
    if not shape:
        return ()
    offset = len(out_shape) - len(shape)
    sliced = index[offset:]
    result = []
    for i, dim in enumerate(shape):
        result.append(0 if dim == 1 else sliced[i])
    return tuple(result)


def _batch_offset(index, stride):
    return sum(i * s for i, s in zip(index, stride))


def _unary_op(a, fn, name, out_dtype=None):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError(f"NPU {name} expects NPU tensors")
    if out_dtype is None:
        out_dtype = a.dtype
    out_size = _numel(a.shape) * _dtype_itemsize(out_dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    storage = _unwrap_storage(a)
    fn(storage.data_ptr(), out_ptr, a.shape, a.stride, a.dtype, runtime, stream=stream.stream)
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(a.shape), out_dtype, device=a.device)
    return _wrap_tensor(out_storage, a.shape, a.stride)


def _binary_op(a, b, fn, name):
    if _HAS_FAST_OPS:
        return _fast_binary_op(a, b, fn, name)
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu" or b.device.type != "npu":
        raise ValueError(f"NPU {name} expects NPU tensors")
    if a.dtype != b.dtype:
        raise ValueError(f"NPU {name} requires matching dtypes")
    out_shape = _broadcast_shape(a.shape, b.shape)
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_size = _numel(out_shape) * _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    a_storage = _unwrap_storage(a)
    b_storage = _unwrap_storage(b)
    fn(
        a_storage.data_ptr(),
        b_storage.data_ptr(),
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
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def _normalize_reduction_dims(dim, ndim):
    if dim is None:
        return list(range(ndim))
    if isinstance(dim, int):
        return [dim]
    return list(dim)


def _reduce_out_shape(shape, dims, keepdim):
    out_shape = list(shape)
    for d in sorted(dims):
        out_shape[d] = 1
    if not keepdim:
        out_shape = [s for i, s in enumerate(out_shape) if i not in dims]
    return tuple(out_shape)


def _reduce_dim_sizes(shape, dims, keepdim):
    dims = sorted(dims)
    sizes = []
    for d in dims:
        sizes.append(shape[d])
    if keepdim:
        out_sizes = [1] * len(shape)
        for d, size in zip(dims, sizes):
            out_sizes[d] = size
        return tuple(out_sizes)
    return tuple(sizes)


def _broadcast_dims_to_out(dims, out_shape, keepdim):
    if keepdim:
        return dims
    offset = len(out_shape) - len(dims)
    return tuple(range(offset, offset + len(dims)))


def _scalar_to_npu_tensor(scalar, ref_tensor):
    """Convert scalar to NPU tensor matching ref_tensor's shape/dtype/device."""
    runtime = npu_runtime.get_runtime((ref_tensor.device.index or 0))
    stream = npu_state.current_stream((ref_tensor.device.index or 0))
    out_shape = ref_tensor.shape
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_size = _numel(out_shape) * _dtype_itemsize(ref_tensor.dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    # Fill on host then memcpy H2D to avoid aclnn scalar ops.
    from .. import acl_loader
    import ctypes
    import struct
    acl = acl_loader.ensure_acl()
    host_ptr, ret = acl.rt.malloc_host(int(out_size))
    if ret != npu_runtime.ACL_ERROR_CODE:
        raise RuntimeError(f"acl.rt.malloc_host failed: {ret}")
    try:
        host_buf = (ctypes.c_uint8 * int(out_size)).from_address(int(host_ptr))
        itemsize = _dtype_itemsize(ref_tensor.dtype)
        dtype_name = getattr(ref_tensor.dtype, "name", None) or str(ref_tensor.dtype).split(".")[-1]
        if dtype_name == "float16":
            from ..aclnn import _float_to_float16_bits
            bits = _float_to_float16_bits(float(scalar))
            pattern = int(bits).to_bytes(2, byteorder="little", signed=False)
        elif dtype_name == "bfloat16":
            from ..aclnn import _float_to_bfloat16_bits
            bits = _float_to_bfloat16_bits(float(scalar))
            pattern = int(bits).to_bytes(2, byteorder="little", signed=False)
        elif dtype_name == "float32":
            pattern = struct.pack("<f", float(scalar))
        elif dtype_name == "float64":
            pattern = struct.pack("<d", float(scalar))
        elif dtype_name == "int8":
            pattern = int(scalar).to_bytes(1, byteorder="little", signed=True)
        elif dtype_name == "uint8":
            pattern = int(scalar).to_bytes(1, byteorder="little", signed=False)
        elif dtype_name == "int16":
            pattern = int(scalar).to_bytes(2, byteorder="little", signed=True)
        elif dtype_name == "int32":
            pattern = int(scalar).to_bytes(4, byteorder="little", signed=True)
        elif dtype_name == "int64":
            pattern = int(scalar).to_bytes(8, byteorder="little", signed=True)
        elif dtype_name == "bool":
            pattern = (1 if bool(scalar) else 0).to_bytes(1, byteorder="little", signed=False)
        else:
            raise ValueError(f"Unsupported scalar dtype: {dtype_name}")
        for offset in range(0, int(out_size), itemsize):
            host_buf[offset:offset + itemsize] = pattern
        npu_runtime.memcpy_h2d(out_ptr, int(out_size), host_ptr, runtime=runtime)
    finally:
        acl.rt.free_host(host_ptr)
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), ref_tensor.dtype, device=ref_tensor.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def _scalar_to_npu_tensor_no_add(scalar, ref_tensor):
    """Helper to avoid recursion: create scalar using add_scalar without add()."""
    runtime = npu_runtime.get_runtime((ref_tensor.device.index or 0))
    stream = npu_state.current_stream((ref_tensor.device.index or 0))
    out_shape = ref_tensor.shape
    out_stride = npu_runtime._contiguous_stride(out_shape)
    out_size = _numel(out_shape) * _dtype_itemsize(ref_tensor.dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    aclnn.inplace_zero(
        out_ptr,
        out_shape,
        out_stride,
        ref_tensor.dtype,
        runtime,
        stream=stream.stream,
    )
    aclnn.add_scalar(
        out_ptr,
        scalar,
        out_ptr,
        out_shape,
        out_stride,
        ref_tensor.dtype,
        runtime,
        stream=stream.stream,
    )
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(out_shape), ref_tensor.dtype, device=ref_tensor.device)
    return _wrap_tensor(out_storage, out_shape, out_stride)


def _nan_like(a):
    return _scalar_to_npu_tensor(float("nan"), a)


def _normalize_dim(dim, ndim):
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise ValueError("dim out of range")
    return dim
