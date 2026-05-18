# cython: language_level=3, boundscheck=False, wraparound=False
import numpy as np

from candle._dtype import float32, int64
from candle._dtype import bool as bool_dtype

cdef bint _frombuffer_writable_warned = False
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


cdef _validate_layout(object layout_arg):
    if layout_arg is None:
        return
    from candle import strided, layout as _layout_cls
    if layout_arg is strided:
        return
    if isinstance(layout_arg, _layout_cls):
        raise RuntimeError(
            f"candle currently only supports torch.strided layout, got {layout_arg!r}"
        )
    raise TypeError(
        f"candle currently only supports torch.strided layout, got {layout_arg!r}"
    )


cdef _check_no_internal_overlap(object out):
    cdef object shape = tuple(out.shape)
    cdef object stride = tuple(out.stride())
    for size_i, stride_i in zip(shape, stride):
        if size_i > 1 and stride_i == 0:
            raise RuntimeError(
                "unsupported operation: more than one element of the written-to tensor "
                "refers to a single memory location. Please clone() the tensor before "
                "performing the operation."
            )


cdef object _finalize_out(object value, object out):
    if out is None:
        return value
    _check_no_internal_overlap(out)
    if tuple(out.shape) != tuple(value.shape):
        if int(out.numel()) != 0 and int(out.numel()) == int(value.numel()):
            out.copy_(value.reshape(tuple(out.shape)))
            return out
        if int(out.numel()) != 0:
            import warnings
            warnings.warn(
                "The out tensor will be resized to match the shape of the result. "
                "This behavior is deprecated, and in a future PyTorch release outputs will not "
                "be resized unless they have zero elements.",
                UserWarning,
                stacklevel=2,
            )
        out.cy_set_data_runtime_truth_from(value)
        return out
    out.copy_(value)
    return out


def finalize_out(value, out):
    return _finalize_out(value, out)


def tensor(data, *, dtype=None, device=None, requires_grad=False):
    from candle._functional import tensor as tensor_dispatch

    if dtype is None:
        dtype = _infer_creation_dtype(data)
    if dtype is None:
        dtype = _get_default_dtype()
    return tensor_dispatch(data, dtype=dtype, device=device, requires_grad=requires_grad)


def _resolve_size_kwarg(tuple shape, object size):
    if size is None:
        return shape
    if len(shape) != 0:
        raise TypeError("received an invalid combination of arguments")
    return (size,)


def zeros(*shape, dtype=None, device=None, memory_format=None, layout=None, out=None, requires_grad=False, size=None):
    from candle._functional import zeros as zeros_dispatch

    shape = _resolve_size_kwarg(shape, size)
    _validate_layout(layout)
    if dtype is None:
        dtype = _get_default_dtype() if out is None else out.dtype
    elif out is not None and out.dtype != dtype:
        raise RuntimeError(
            f"zeros() expected a tensor of dtype {dtype} but got dtype {out.dtype} for argument 'out'"
        )
    value = zeros_dispatch(*shape, dtype=dtype, device=device, memory_format=memory_format)
    if out is not None:
        return _apply_requires_grad(_finalize_out(value, out), requires_grad)
    return _apply_requires_grad(value, requires_grad)


def ones(*shape, dtype=None, device=None, memory_format=None, layout=None, out=None, requires_grad=False, size=None):
    from candle._functional import ones as ones_dispatch

    shape = _resolve_size_kwarg(shape, size)
    _validate_layout(layout)
    if dtype is None:
        dtype = _get_default_dtype() if out is None else out.dtype
    elif out is not None and out.dtype != dtype:
        raise RuntimeError(
            f"ones() expected a tensor of dtype {dtype} but got dtype {out.dtype} for argument 'out'"
        )
    value = ones_dispatch(*shape, dtype=dtype, device=device, memory_format=memory_format)
    if out is not None:
        return _apply_requires_grad(_finalize_out(value, out), requires_grad)
    return _apply_requires_grad(value, requires_grad)


def empty(*shape, dtype=None, device=None, memory_format=None, layout=None, out=None, requires_grad=False, size=None):
    from candle._functional import empty as empty_dispatch

    shape = _resolve_size_kwarg(shape, size)
    _validate_layout(layout)
    if dtype is None:
        dtype = _get_default_dtype() if out is None else out.dtype
    elif out is not None and out.dtype != dtype:
        raise RuntimeError(
            f"empty() expected a tensor of dtype {dtype} but got dtype {out.dtype} for argument 'out'"
        )
    value = empty_dispatch(*shape, dtype=dtype, device=device, memory_format=memory_format)
    if out is not None:
        return _apply_requires_grad(_finalize_out(value, out), requires_grad)
    return _apply_requires_grad(value, requires_grad)


def empty_strided(size, stride, *, dtype=None, device=None, layout=None, requires_grad=False, pin_memory=False):
    cdef tuple size_t
    cdef tuple stride_t
    cdef Py_ssize_t required = 0
    cdef Py_ssize_t max_index = 0
    cdef Py_ssize_t dim
    cdef Py_ssize_t step
    cdef object base

    _validate_layout(layout)
    if dtype is None:
        dtype = _get_default_dtype()
    if pin_memory:
        raise RuntimeError("Need to provide pin_memory allocator to use pin memory.")
    size_t = tuple(int(s) for s in size)
    stride_t = tuple(int(s) for s in stride)
    if len(size_t) != len(stride_t):
        raise RuntimeError("mismatch in length of strides and shape")
    if size_t and all(dim > 0 for dim in size_t):
        for dim, step in zip(size_t, stride_t):
            max_index += (dim - 1) * step
        required = max_index + 1
    base = empty((required,), dtype=dtype, device=device, requires_grad=requires_grad)
    return base.as_strided(size_t, stride_t)


def _format_range_endpoint(object value):
    if value == float("inf"):
        return "inf"
    if value == float("-inf"):
        return "-inf"
    if value != value:
        return "nan"
    return str(value)


def _validate_arange_args(object start, object end, object step):
    cdef object actual_start = 0 if end is None else start
    cdef object actual_end = start if end is None else end
    import math

    if step == 0:
        raise RuntimeError("step must be nonzero")
    if isinstance(actual_start, (int, float)) and isinstance(actual_end, (int, float)) and isinstance(step, (int, float)):
        if not math.isfinite(float(actual_start)) or not math.isfinite(float(actual_end)):
            raise RuntimeError(
                f"unsupported range: {_format_range_endpoint(actual_start)} -> {_format_range_endpoint(actual_end)}"
            )
        if math.isfinite(float(step)) and step != 0:
            span = (float(actual_end) - float(actual_start)) / float(step)
            if span > 9223372036854775807:
                raise RuntimeError("overflow when unpacking long")


def arange(start, end=None, step=1, dtype=None, device=None, layout=None, out=None, requires_grad=False):
    from candle._functional import arange as arange_dispatch

    cdef list args

    _validate_arange_args(start, end, step)
    _validate_layout(layout)
    if dtype is None:
        if out is not None:
            dtype = out.dtype
        else:
            args = [start] + ([end] if end is not None else []) + [step]
            if all(isinstance(a, int) for a in args):
                dtype = int64
            else:
                dtype = _get_default_dtype()
    value = arange_dispatch(start, end=end, step=step, dtype=dtype, device=device)
    if out is not None:
        return _apply_requires_grad(_finalize_out(value, out), requires_grad)
    return _apply_requires_grad(value, requires_grad)


cdef object _is_complex_scalar(object value):
    try:
        if isinstance(value, complex):
            return True
    except TypeError:
        pass
    return False


cdef object _check_linspace_logspace_dtype(str fn_name, object start, object end, object dtype):
    if not (_is_complex_scalar(start) or _is_complex_scalar(end)):
        return dtype
    from candle._dtype import complex64, float32
    inferred = complex64
    if dtype is None:
        return inferred
    name = getattr(dtype, "name", str(dtype))
    if name in ("complex32", "complex64", "complex128"):
        return dtype
    cxx_name = {
        "complex32": "c10::complex<at::Half>",
        "complex64": "c10::complex<float>",
        "complex128": "c10::complex<double>",
    }.get(getattr(inferred, "name", "complex64"), "c10::complex<float>")
    passed_name = {
        "float16": "Half",
        "float32": "Float",
        "float64": "Double",
        "bfloat16": "BFloat16",
        "bool": "Bool",
        "uint8": "Byte",
        "int8": "Char",
        "int16": "Short",
        "int32": "Int",
        "int64": "Long",
    }.get(name, name)
    raise RuntimeError(
        f"torch.{fn_name}(): inferred dtype {cxx_name} can't be safely cast to passed dtype {passed_name}"
    )


def linspace(start, end, steps, dtype=None, device=None, layout=None, out=None, requires_grad=False):
    from candle._functional import linspace as linspace_dispatch

    _validate_layout(layout)
    if int(steps) < 0:
        raise RuntimeError(f"number of steps must be non-negative, but got steps={int(steps)}")
    dtype_for_check = dtype if dtype is not None else (out.dtype if out is not None else None)
    dtype = _check_linspace_logspace_dtype("linspace", start, end, dtype_for_check)
    if dtype is None:
        dtype = _get_default_dtype()
    value = linspace_dispatch(start, end, steps, dtype=dtype, device=device)
    if out is not None:
        return _apply_requires_grad(_finalize_out(value, out), requires_grad)
    return _apply_requires_grad(value, requires_grad)


def full(*args, dtype=None, device=None, requires_grad=False, memory_format=None, layout=None, out=None):
    from candle._functional import full as full_dispatch

    _validate_layout(layout)
    cdef Py_ssize_t _nargs = len(args)
    fill_value = args[_nargs - 1] if _nargs > 0 else None
    if dtype is None:
        if out is not None:
            dtype = out.dtype
        elif isinstance(fill_value, bool):
            dtype = bool_dtype
        elif isinstance(fill_value, int):
            dtype = int64
        elif isinstance(fill_value, complex):
            default_dtype = _get_default_dtype()
            if getattr(default_dtype, "name", None) == "float64":
                from candle._dtype import complex128 as _complex128
                dtype = _complex128
            else:
                from candle._dtype import complex64 as _complex64
                dtype = _complex64
        else:
            dtype = _get_default_dtype()
    elif out is not None and out.dtype != dtype:
        raise RuntimeError(
            "full() expected a tensor of dtype "
            f"{dtype} but got dtype {out.dtype} for argument 'out'"
        )
    value = full_dispatch(*args, dtype=dtype, device=device, memory_format=memory_format)
    if out is not None:
        return _apply_requires_grad(_finalize_out(value, out), requires_grad)
    return _apply_requires_grad(value, requires_grad)


def logspace(start, end, steps, base=10.0, dtype=None, device=None, layout=None, out=None, requires_grad=False):
    from candle._functional import logspace as logspace_dispatch

    _validate_layout(layout)
    if int(steps) < 0:
        raise RuntimeError(f"number of steps must be non-negative, but got steps={int(steps)}")
    dtype_for_check = dtype if dtype is not None else (out.dtype if out is not None else None)
    dtype = _check_linspace_logspace_dtype("logspace", start, end, dtype_for_check)
    if dtype is None:
        dtype = _get_default_dtype()
    if base != 10.0:
        # Fallback: compute via linspace + base ** x to support arbitrary base.
        from candle._functional import linspace as linspace_dispatch
        exponents = linspace_dispatch(start, end, steps, dtype=dtype, device=device)
        value = base ** exponents
    else:
        value = logspace_dispatch(start, end, steps, dtype=dtype, device=device)
    if out is not None:
        return _apply_requires_grad(_finalize_out(value, out), requires_grad)
    return _apply_requires_grad(value, requires_grad)


def eye(n, m=None, dtype=None, device=None, layout=None, out=None, requires_grad=False):
    from candle._functional import eye as eye_dispatch

    _validate_layout(layout)
    if dtype is None:
        dtype = _get_default_dtype() if out is None else out.dtype
    value = eye_dispatch(n, m, dtype=dtype, device=device, out=None)
    if out is not None:
        return _apply_requires_grad(_finalize_out(value, out), requires_grad)
    return _apply_requires_grad(value, requires_grad)


def range(start, end, step=1, dtype=None, device=None, out=None):
    from candle._functional import range as range_dispatch

    if dtype is None:
        dtype = _get_default_dtype() if out is None else out.dtype
    value = range_dispatch(start, end, step=step, dtype=dtype, device=device)
    return _finalize_out(value, out)


def randn(*shape, dtype=None, device=None, memory_format=None, generator=None, layout=None, out=None, requires_grad=False):
    from candle._functional import randn as randn_dispatch

    _validate_layout(layout)
    if dtype is None:
        dtype = _get_default_dtype() if out is None else out.dtype
    value = randn_dispatch(*shape, dtype=dtype, device=device, memory_format=memory_format, generator=generator)
    if out is not None:
        return _apply_requires_grad(_finalize_out(value, out), requires_grad)
    return _apply_requires_grad(value, requires_grad)


def rand(*shape, dtype=None, device=None, memory_format=None, generator=None, layout=None, out=None, requires_grad=False):
    from candle._functional import rand as rand_dispatch

    _validate_layout(layout)
    if dtype is None:
        dtype = _get_default_dtype() if out is None else out.dtype
    value = rand_dispatch(*shape, dtype=dtype, device=device, memory_format=memory_format, generator=generator)
    if out is not None:
        return _apply_requires_grad(_finalize_out(value, out), requires_grad)
    return _apply_requires_grad(value, requires_grad)


def randint(low, high=None, size=None, *, dtype=None, device=None, generator=None, memory_format=None, layout=None, out=None):
    from candle._functional import randint as randint_dispatch

    _validate_layout(layout)
    if size is None and isinstance(high, (tuple, list)):
        size = high
        high = None
    if dtype is None and out is not None:
        dtype = out.dtype
    value = randint_dispatch(
        low, high=high, size=size, dtype=dtype, device=device,
        generator=generator, memory_format=memory_format,
    )
    return _finalize_out(value, out)


def randperm(n, *, dtype=None, device=None, generator=None, layout=None, out=None):
    from candle._functional import randperm as randperm_dispatch

    _validate_layout(layout)
    if dtype is None and out is not None:
        dtype = out.dtype
    value = randperm_dispatch(n, dtype=dtype, device=device, generator=generator)
    return _finalize_out(value, out)


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

    try:
        readonly = bool(view.readonly)
    except AttributeError:
        readonly = False
    global _frombuffer_writable_warned
    if readonly:
        from candle import is_warn_always_enabled
        if is_warn_always_enabled() or not _frombuffer_writable_warned:
            import warnings
            warnings.warn(
                "The given buffer is not writable, and PyTorch does not support non-writable tensors. "
                "This means you can write to the underlying (supposedly non-writable) buffer using the tensor. "
                "You may want to copy the buffer to protect its data or make it writable before converting it to a tensor. "
                "This type of warning will be suppressed for the rest of this program.",
                UserWarning,
                stacklevel=2,
            )
            _frombuffer_writable_warned = True

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
        if tensor_obj.requires_grad != bool(requires_grad):
            tensor_obj.requires_grad_(bool(requires_grad))
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
