# cython: language_level=3, boundscheck=False, wraparound=False
import numpy as np

from candle._dtype import float32, int64
from candle._dtype import bool as bool_dtype
cdef object _get_default_dtype():
    from candle import get_default_dtype
    return get_default_dtype()


cdef object _infer_creation_dtype(object data):
    cdef object arr

    if isinstance(data, (np.ndarray, np.generic)):
        if data.dtype == np.bool_:
            return bool_dtype
        if np.issubdtype(data.dtype, np.integer):
            return int64
        return None
    if hasattr(data, "dtype"):
        return None
    try:
        arr = np.asarray(data)
    except Exception:
        return None
    if arr.dtype == np.bool_:
        return bool_dtype
    if np.issubdtype(arr.dtype, np.integer):
        return int64
    return None


cdef object _dtype_from_numpy(object np_dtype):
    from candle._dtype import from_numpy_dtype
    return from_numpy_dtype(np_dtype)


cdef object _apply_requires_grad(object out, object requires_grad):
    if not requires_grad:
        return out
    if not (out.dtype.is_floating_point or out.dtype.is_complex):
        raise RuntimeError("Only Tensors of floating point and complex dtype can require gradients")
    out.requires_grad_(True)
    return out


cdef tuple _shape_stride_from_array(object arr):
    cdef Py_ssize_t itemsize = arr.dtype.itemsize
    cdef list stride = []
    cdef object byte_stride

    for byte_stride in arr.strides:
        stride.append(int(byte_stride) // int(itemsize))
    return tuple(arr.shape), tuple(stride)


cdef object _tensor_from_numpy_array(object arr, object dtype, object requires_grad=False):
    cdef object shape
    cdef object stride
    from candle._C import typed_storage_from_numpy
    from candle._C._tensor_impl import cy_make_tensor_from_storage

    shape, stride = _shape_stride_from_array(arr)
    return cy_make_tensor_from_storage(
        typed_storage_from_numpy(arr, dtype), shape, stride, 0, bool(requires_grad)
    )


cdef object _as_device(object dev):
    from candle import device as device_ctor
    if dev is None or hasattr(dev, "type"):
        return dev
    return device_ctor(dev)


cdef bint _device_matches(object tensor, object target_device):
    cdef object target = _as_device(target_device)
    if target is None:
        return True
    if tensor.device.type != target.type:
        return False
    if target.index is None:
        return True
    return tensor.device.index == target.index


def tensor(data, *, dtype=None, device=None, requires_grad=False):
    from candle._functional import tensor as tensor_dispatch

    if dtype is None:
        dtype = _infer_creation_dtype(data)
    if dtype is None:
        dtype = _get_default_dtype()
    return tensor_dispatch(data, dtype=dtype, device=device, requires_grad=requires_grad)


def zeros(*shape, dtype=None, device=None, memory_format=None, requires_grad=False):
    from candle._functional import zeros as zeros_dispatch

    if dtype is None:
        dtype = _get_default_dtype()
    return _apply_requires_grad(
        zeros_dispatch(*shape, dtype=dtype, device=device, memory_format=memory_format),
        requires_grad,
    )


def ones(*shape, dtype=None, device=None, memory_format=None, requires_grad=False):
    from candle._functional import ones as ones_dispatch

    if dtype is None:
        dtype = _get_default_dtype()
    return _apply_requires_grad(
        ones_dispatch(*shape, dtype=dtype, device=device, memory_format=memory_format),
        requires_grad,
    )


def empty(*shape, dtype=None, device=None, memory_format=None, requires_grad=False):
    from candle._functional import empty as empty_dispatch

    if dtype is None:
        dtype = _get_default_dtype()
    return _apply_requires_grad(
        empty_dispatch(*shape, dtype=dtype, device=device, memory_format=memory_format),
        requires_grad,
    )


def arange(start, end=None, step=1, dtype=None, device=None, requires_grad=False):
    from candle._functional import arange as arange_dispatch

    cdef list args

    if dtype is None:
        args = [start] + ([end] if end is not None else []) + [step]
        if all(isinstance(a, int) for a in args):
            dtype = int64
        else:
            dtype = _get_default_dtype()
    return _apply_requires_grad(
        arange_dispatch(start, end=end, step=step, dtype=dtype, device=device),
        requires_grad,
    )


def linspace(start, end, steps, dtype=None, device=None, requires_grad=False):
    from candle._functional import linspace as linspace_dispatch

    if dtype is None:
        dtype = _get_default_dtype()
    return _apply_requires_grad(
        linspace_dispatch(start, end, steps, dtype=dtype, device=device),
        requires_grad,
    )


def full(*args, dtype=None, device=None, requires_grad=False, memory_format=None):
    from candle._functional import full as full_dispatch

    if dtype is None:
        dtype = _get_default_dtype()
    return _apply_requires_grad(
        full_dispatch(*args, dtype=dtype, device=device, memory_format=memory_format),
        requires_grad,
    )


def logspace(start, end, steps, dtype=None, device=None, requires_grad=False):
    from candle._functional import logspace as logspace_dispatch

    if dtype is None:
        dtype = _get_default_dtype()
    return _apply_requires_grad(
        logspace_dispatch(start, end, steps, dtype=dtype, device=device),
        requires_grad,
    )


def eye(n, m=None, dtype=None, device=None, out=None, requires_grad=False):
    from candle._functional import eye as eye_dispatch

    if dtype is None:
        dtype = _get_default_dtype()
    return _apply_requires_grad(
        eye_dispatch(n, m, dtype=dtype, device=device, out=out),
        requires_grad,
    )


def range(start, end, step=1, dtype=None, device=None):
    from candle._functional import range as range_dispatch

    if dtype is None:
        dtype = _get_default_dtype()
    return range_dispatch(start, end, step=step, dtype=dtype, device=device)


def randn(*shape, dtype=None, device=None, memory_format=None, generator=None, requires_grad=False):
    from candle._functional import randn as randn_dispatch

    if dtype is None:
        dtype = _get_default_dtype()
    return _apply_requires_grad(
        randn_dispatch(*shape, dtype=dtype, device=device, memory_format=memory_format, generator=generator),
        requires_grad,
    )


def rand(*shape, dtype=None, device=None, memory_format=None, generator=None, requires_grad=False):
    from candle._functional import rand as rand_dispatch

    if dtype is None:
        dtype = _get_default_dtype()
    return _apply_requires_grad(
        rand_dispatch(*shape, dtype=dtype, device=device, memory_format=memory_format, generator=generator),
        requires_grad,
    )


def randint(low, high=None, size=None, *, dtype=None, device=None, generator=None, memory_format=None):
    from candle._functional import randint as randint_dispatch

    if size is None and isinstance(high, (tuple, list)):
        size = high
        high = None
    return randint_dispatch(
        low, high=high, size=size, dtype=dtype, device=device,
        generator=generator, memory_format=memory_format,
    )


def randperm(n, *, dtype=None, device=None, generator=None):
    from candle._functional import randperm as randperm_dispatch

    return randperm_dispatch(n, dtype=dtype, device=device, generator=generator)


def from_numpy(ndarray):
    dtype = _dtype_from_numpy(ndarray.dtype)
    return _tensor_from_numpy_array(ndarray, dtype)


def frombuffer(buffer, *, dtype, count=-1, offset=0, requires_grad=False):
    from candle._dtype import to_numpy_dtype

    cdef Py_ssize_t byte_len
    cdef Py_ssize_t itemsize
    cdef Py_ssize_t actual_count
    cdef Py_ssize_t count_int = int(count)
    cdef Py_ssize_t offset_int = int(offset)
    cdef object view
    cdef object arr
    cdef object np_dtype = to_numpy_dtype(dtype)

    try:
        view = memoryview(buffer)
    except TypeError as exc:
        raise ValueError("object does not implement Python buffer protocol.") from exc

    byte_len = int(view.nbytes)
    itemsize = int(np.dtype(np_dtype).itemsize)
    if not (byte_len > 0 and count_int != 0):
        raise ValueError(f"both buffer length ({byte_len}) and count ({count_int}) must not be 0")
    if not (offset_int >= 0 and offset_int < byte_len):
        raise ValueError(
            f"offset ({offset_int} bytes) must be non-negative and no greater than "
            f"buffer length ({byte_len} bytes) minus 1"
        )
    if not (count_int > 0 or (byte_len - offset_int) % itemsize == 0):
        raise ValueError(
            f"buffer length ({byte_len - offset_int} bytes) after offset ({offset_int} bytes) "
            f"must be a multiple of element size ({itemsize})"
        )
    if count_int < 0:
        actual_count = (byte_len - offset_int) // itemsize
    else:
        actual_count = count_int
    if not (offset_int + actual_count * itemsize <= byte_len):
        raise ValueError(
            f"requested buffer length ({actual_count} * {itemsize} bytes) after offset "
            f"({offset_int} bytes) must not be greater than actual buffer length ({byte_len} bytes)"
        )

    arr = np.frombuffer(buffer, dtype=np_dtype, count=int(actual_count), offset=int(offset_int))
    return _apply_requires_grad(_tensor_from_numpy_array(arr, dtype), requires_grad)


def as_tensor(data, dtype=None, device=None):
    from candle._functional import tensor as tensor_dispatch
    from candle._tensor import Tensor

    if isinstance(data, Tensor):
        if dtype is None and device is None:
            return data
        return data.to(device=device, dtype=dtype)

    if dtype is None:
        dtype = _infer_creation_dtype(data)
    if dtype is None:
        dtype = _get_default_dtype()
    return tensor_dispatch(data, dtype=dtype, device=device)


def asarray(obj, *, dtype=None, device=None, copy=None, requires_grad=False):
    from candle._functional import tensor as tensor_dispatch
    from candle._tensor import Tensor

    cdef bint force_copy = bool(copy) if copy is not None else False
    cdef bint force_alias = copy is False
    cdef bint wrong_device = False
    cdef bint wrong_dtype = False
    cdef bint needs_copying = False
    cdef object tensor_obj = None
    cdef object dtype_unwrapped = dtype if dtype is not None else _get_default_dtype()
    cdef object dev = _as_device(device)
    cdef object arr
    cdef object effective_dtype

    if isinstance(obj, Tensor):
        tensor_obj = obj
    elif isinstance(obj, (np.ndarray, np.generic)):
        if isinstance(obj, np.generic):
            if force_alias:
                raise ValueError("can't alias NumPy scalars. Either remove copy=False or transform it in a ndarray. ")
            arr = np.asarray(obj)
            force_copy = False
        else:
            arr = obj
        effective_dtype = dtype if dtype is not None else _dtype_from_numpy(arr.dtype)
        if copy is False and not arr.flags.c_contiguous:
            raise ValueError("can't alias non-contiguous NumPy array into a tensor.")
        tensor_obj = _tensor_from_numpy_array(arr, effective_dtype)
    else:
        try:
            memoryview(obj)
        except TypeError:
            pass
        else:
            tensor_obj = frombuffer(obj, dtype=dtype_unwrapped, count=-1, offset=0, requires_grad=requires_grad)

    if tensor_obj is not None:
        wrong_device = dev is not None and not _device_matches(tensor_obj, dev)
        wrong_dtype = dtype is not None and tensor_obj.dtype != dtype
        needs_copying = copy is None and (wrong_device or wrong_dtype)
        if force_copy or needs_copying:
            if wrong_device or wrong_dtype:
                tensor_obj = tensor_obj.to(
                    device=dev if dev is not None else tensor_obj.device,
                    dtype=dtype if dtype is not None else tensor_obj.dtype,
                )
            else:
                tensor_obj = tensor_obj.clone()
        else:
            if wrong_device:
                raise ValueError(f"can't alias tensor from device '{tensor_obj.device}' to '{dev}'.")
            if wrong_dtype:
                raise ValueError(f"can't alias tensor with dtype '{tensor_obj.dtype}' into dtype '{dtype}'.")
        return _apply_requires_grad(tensor_obj, requires_grad)

    if force_alias:
        raise ValueError("can't alias arbitrary sequence into a tensor.")
    if dtype is None:
        dtype = _infer_creation_dtype(obj)
    if dtype is None:
        dtype = dtype_unwrapped
    return _apply_requires_grad(tensor_dispatch(obj, dtype=dtype, device=device), requires_grad)


def normal(mean, std, size=None, *, generator=None, out=None):
    from candle._functional import normal as normal_dispatch

    return normal_dispatch(mean, std, size=size, generator=generator, out=out)
