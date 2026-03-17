"""Pure-Python fallback for _npu_ops_fast (identical API).

Used when Cython is not available (e.g. CI without CANN, pip install
without a compiler).  The fast_binary_op function here is functionally
identical to the Cython version but runs in pure Python.
"""

_MAX_NDIM = 16


def _broadcast_shape(a_shape, b_shape):
    """Broadcast two shapes, return result tuple."""
    a_ndim = len(a_shape)
    b_ndim = len(b_shape)
    out_ndim = max(a_ndim, b_ndim)
    out = [0] * out_ndim
    for i in range(out_ndim):
        a_dim = a_shape[a_ndim - 1 - i] if i < a_ndim else 1
        b_dim = b_shape[b_ndim - 1 - i] if i < b_ndim else 1
        if a_dim == b_dim:
            out[out_ndim - 1 - i] = a_dim
        elif a_dim == 1:
            out[out_ndim - 1 - i] = b_dim
        elif b_dim == 1:
            out[out_ndim - 1 - i] = a_dim
        else:
            raise ValueError("matmul shape mismatch")
    return tuple(out)


def _contiguous_stride(shape):
    """Compute contiguous strides."""
    ndim = len(shape)
    stride = [0] * ndim
    acc = 1
    for i in range(ndim - 1, -1, -1):
        stride[i] = acc
        acc *= shape[i]
    return tuple(stride)


def _numel(shape):
    """Product of shape dims."""
    n = 1
    for d in shape:
        n *= d
    return n


def _dtype_itemsize(dtype):
    """Return byte size from a candle dtype object."""
    size = getattr(dtype, "itemsize", None)
    if size is not None:
        return int(size)
    name = getattr(dtype, "name", None)
    if name is None:
        name = str(dtype).rsplit(".", 1)[-1]
    _map = {
        "float32": 4, "int32": 4, "float64": 8, "int64": 8,
        "float16": 2, "bfloat16": 2, "int16": 2,
        "int8": 1, "uint8": 1, "bool": 1,
    }
    return _map.get(name, 4)


def fast_binary_op(a, b, fn, name):
    """Drop-in replacement for _binary_op in _helpers.py (pure Python)."""
    from . import runtime as npu_runtime
    from . import state as npu_state
    from ..._storage import npu_typed_storage_from_ptr
    from ..._tensor import Tensor

    a_dev = a.device
    b_dev = b.device
    if a_dev.type != "npu" or b_dev.type != "npu":
        raise ValueError(f"NPU {name} expects NPU tensors")
    a_dtype = a.dtype
    if a_dtype != b.dtype:
        raise ValueError(f"NPU {name} requires matching dtypes")

    dev_idx = a_dev.index or 0
    runtime = npu_runtime.get_runtime(dev_idx)
    stream = npu_state.current_stream(dev_idx)

    out_shape = _broadcast_shape(a.shape, b.shape)
    out_stride = _contiguous_stride(out_shape)
    n = _numel(out_shape)
    isize = _dtype_itemsize(a_dtype)
    out_ptr = npu_runtime._alloc_device(n * isize, runtime=runtime)

    a_storage = a.storage()
    b_storage = b.storage()

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
        a_dtype,
        runtime,
        stream=stream.stream,
    )

    out_storage = npu_typed_storage_from_ptr(
        out_ptr, n, a_dtype, device=a_dev)
    return Tensor(out_storage, out_shape, out_stride)
