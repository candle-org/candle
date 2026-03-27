# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython hot wrappers for candle._tensor.Tensor methods.

These functions are installed onto ``candle._tensor.Tensor`` when the extension
is available so the hottest Tensor forwarding paths run through compiled code
while preserving the existing Python fallback behavior in ``candle._tensor``.
"""

import numpy as np

cdef object _BaseTensor = None
cdef object _Device = None
cdef object _from_name_fn = None
cdef object _backward_fn = None
cdef object _current_pipeline_fn = None

cdef object _add_fn = None
cdef object _mul_fn = None
cdef object _matmul_fn = None
cdef object _relu_fn = None
cdef object _neg_fn = None
cdef object _reshape_dispatch_fn = None
cdef object _transpose_dispatch_fn = None
cdef object _view_dispatch_fn = None
cdef object _to_dispatch_fn = None
cdef object _dispatch_fn = None
cdef object _cy_make_view_tensor_fn = None
cdef object _cy_make_tensor_from_storage_fn = None
cdef object _HookHandle_cls = None
cdef object _is_grad_enabled_fn = None

cdef object _typed_storage_from_numpy_fn = None
cdef object _meta_typed_storage_from_shape_fn = None
cdef object _mps_typed_storage_from_numpy_fn = None
cdef object _to_numpy_dtype_fn = None
cdef object _bfloat16_dtype = None
cdef object _bf16_to_f32_fn = None
cdef object _f32_to_bf16_fn = None
cdef object _cast_tensor_dtype_npu_fn = None


cdef inline void _ensure_base():
    global _BaseTensor
    if _BaseTensor is None:
        from candle._tensor import Tensor
        _BaseTensor = Tensor


cdef inline void _ensure_device_ref():
    global _Device
    if _Device is None:
        from candle._device import device as Device
        _Device = Device


cdef inline void _ensure_dtype_ref():
    global _from_name_fn
    if _from_name_fn is None:
        from candle._dtype import from_name
        _from_name_fn = from_name


cdef inline void _ensure_backward_ref():
    global _backward_fn
    if _backward_fn is None:
        from candle.autograd.engine import backward
        _backward_fn = backward


cdef inline void _ensure_pipeline_ref():
    global _current_pipeline_fn
    if _current_pipeline_fn is None:
        from candle._dispatch.pipeline import current_pipeline
        _current_pipeline_fn = current_pipeline


cdef inline void _ensure_dispatch_ref():
    global _dispatch_fn
    if _dispatch_fn is None:
        from candle._dispatch import dispatch
        _dispatch_fn = dispatch


cdef inline void _ensure_view_factory_ref():
    global _cy_make_view_tensor_fn
    if _cy_make_view_tensor_fn is None:
        from candle._cython._tensor_impl import cy_make_view_tensor
        _cy_make_view_tensor_fn = cy_make_view_tensor


cdef inline void _ensure_tensor_factory_ref():
    global _cy_make_tensor_from_storage_fn
    if _cy_make_tensor_from_storage_fn is None:
        from candle._cython._tensor_impl import cy_make_tensor_from_storage
        _cy_make_tensor_from_storage_fn = cy_make_tensor_from_storage


cdef inline void _ensure_hook_handle_ref():
    global _HookHandle_cls
    if _HookHandle_cls is None:
        from candle._tensor import _HookHandle
        _HookHandle_cls = _HookHandle


cdef inline void _ensure_grad_mode_ref():
    global _is_grad_enabled_fn
    if _is_grad_enabled_fn is None:
        from candle.autograd.grad_mode import is_grad_enabled
        _is_grad_enabled_fn = is_grad_enabled


cdef inline void _ensure_functional_refs():
    global _add_fn, _mul_fn, _matmul_fn, _relu_fn, _neg_fn
    global _reshape_dispatch_fn, _transpose_dispatch_fn, _view_dispatch_fn
    global _to_dispatch_fn

    if _add_fn is None:
        from candle._functional import (
            add as add_fn,
            matmul as matmul_fn,
            mul as mul_fn,
            neg as neg_fn,
            relu as relu_fn,
            reshape as reshape_dispatch_fn,
            to as to_dispatch_fn,
            transpose as transpose_dispatch_fn,
            view as view_dispatch_fn,
        )
        _add_fn = add_fn
        _mul_fn = mul_fn
        _matmul_fn = matmul_fn
        _relu_fn = relu_fn
        _neg_fn = neg_fn
        _reshape_dispatch_fn = reshape_dispatch_fn
        _transpose_dispatch_fn = transpose_dispatch_fn
        _view_dispatch_fn = view_dispatch_fn
        _to_dispatch_fn = to_dispatch_fn


cdef inline void _ensure_conversion_refs():
    global _typed_storage_from_numpy_fn, _meta_typed_storage_from_shape_fn
    global _mps_typed_storage_from_numpy_fn, _to_numpy_dtype_fn
    global _bfloat16_dtype, _bf16_to_f32_fn, _f32_to_bf16_fn
    global _cast_tensor_dtype_npu_fn

    if _typed_storage_from_numpy_fn is None:
        from candle._storage import (
            typed_storage_from_numpy,
            meta_typed_storage_from_shape,
            mps_typed_storage_from_numpy,
        )
        from candle._dtype import to_numpy_dtype, bfloat16
        from candle._tensor import _bf16_to_f32, _f32_to_bf16
        from candle._backends.npu.ops._helpers import _cast_tensor_dtype

        _typed_storage_from_numpy_fn = typed_storage_from_numpy
        _meta_typed_storage_from_shape_fn = meta_typed_storage_from_shape
        _mps_typed_storage_from_numpy_fn = mps_typed_storage_from_numpy
        _to_numpy_dtype_fn = to_numpy_dtype
        _bfloat16_dtype = bfloat16
        _bf16_to_f32_fn = _bf16_to_f32
        _f32_to_bf16_fn = _f32_to_bf16
        _cast_tensor_dtype_npu_fn = _cast_tensor_dtype


cdef inline void _flush_pending(object tensor):
    cdef object pipe

    if tensor._pending:
        _ensure_pipeline_ref()
        pipe = _current_pipeline_fn()
        if pipe is not None:
            pipe.flush()


def tensor_add(self, other):
    _ensure_functional_refs()
    return _add_fn(self, other)


def tensor_sub(self, other):
    _ensure_base()
    _ensure_functional_refs()
    if isinstance(other, _BaseTensor):
        return _add_fn(self, _neg_fn(other))
    return _add_fn(self, -other)


def tensor_mul(self, other):
    _ensure_functional_refs()
    return _mul_fn(self, other)


def tensor_matmul(self, other):
    _ensure_functional_refs()
    return _matmul_fn(self, other)


def tensor_iadd(self, other):
    self._check_inplace()
    self.add_(other)
    return self


def tensor_imul(self, other):
    self._check_inplace()
    self.mul_(other)
    return self


def tensor_neg(self):
    _ensure_functional_refs()
    return _neg_fn(self)


def tensor_clone(self):
    _ensure_functional_refs()
    return _to_dispatch_fn(self, self.device, copy=True)


def tensor_detach(self):
    cdef object out

    _ensure_base()
    out = _BaseTensor(self._storage, self.shape, self.stride, self.offset, requires_grad=False)
    out.grad_fn = None
    out.grad = None
    out._pending = self._pending
    out._version_counter = self._version_counter
    return out


def tensor_detach_(self):
    self.requires_grad = False
    self.grad_fn = None
    self._retain_grad = False
    return self


def tensor_to(self, *args, **kwargs):
    cdef object device = None
    cdef object dtype = None
    cdef object non_blocking
    cdef object copy
    cdef object memory_format
    cdef object result = self
    cdef object arg
    cdef object dt

    _ensure_device_ref()
    _ensure_dtype_ref()
    _ensure_functional_refs()

    _flush_pending(self)

    non_blocking = kwargs.get("non_blocking", False)
    copy = kwargs.get("copy", False)
    memory_format = kwargs.get("memory_format", None)

    for arg in args:
        if isinstance(arg, _Device):
            device = arg
        elif isinstance(arg, str):
            dt = _from_name_fn(arg)
            if dt is not None:
                dtype = dt
            else:
                device = _Device(arg)
        elif hasattr(arg, "name") and hasattr(arg, "itemsize"):
            dtype = arg
        else:
            device = _Device(str(arg))

    if "device" in kwargs:
        device = kwargs["device"]
        if isinstance(device, str):
            device = _Device(device)

    if "dtype" in kwargs:
        dtype = kwargs["dtype"]

    if dtype is not None and dtype != self.dtype:
        result = result._to_dtype(dtype)

    if device is not None:
        result = _to_dispatch_fn(
            result,
            device,
            dtype=dtype,
            non_blocking=non_blocking,
            copy=copy,
            memory_format=memory_format,
        )

    if result is self and dtype is None and device is None:
        return self
    return result


def tensor_backward(self, gradient=None, retain_graph=False, create_graph=False, inputs=None):
    _flush_pending(self)
    _ensure_backward_ref()
    _backward_fn(
        self,
        gradient,
        retain_graph=retain_graph,
        create_graph=create_graph,
        inputs=inputs,
    )


def tensor_relu(self):
    _ensure_functional_refs()
    return _relu_fn(self)


def tensor_reshape(self, *shape):
    if not shape:
        raise TypeError("reshape() missing shape arguments")
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    _ensure_functional_refs()
    return _reshape_dispatch_fn(self, shape)


def tensor_transpose(self, dim0, dim1):
    _ensure_functional_refs()
    return _transpose_dispatch_fn(self, dim0, dim1)


def tensor_view(self, *shape):
    if not shape:
        raise TypeError(
            "view() received an invalid combination of arguments - got (), but expected one of:\n"
            " * (torch.dtype dtype)\n"
            " * (tuple of ints size)\n"
        )
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    _ensure_functional_refs()
    return _view_dispatch_fn(self, shape)


def tensor_is_contiguous(self, memory_format=None):
    cdef tuple expected
    expected = _contiguous_stride_tuple(self.shape)
    return self.stride == expected


def tensor_contiguous(self, memory_format=None):
    if self.is_contiguous(memory_format=memory_format):
        return self
    _ensure_dispatch_ref()
    return _dispatch_fn("contiguous", self.device.type, self)


def tensor_flatten(self, start_dim=0, end_dim=-1):
    cdef Py_ssize_t ndim = len(self.shape)
    cdef Py_ssize_t i
    cdef Py_ssize_t flattened = 1
    cdef tuple new_shape

    if ndim == 0:
        return self.reshape((1,))
    if start_dim < 0:
        start_dim += ndim
    if end_dim < 0:
        end_dim += ndim
    if start_dim < 0 or start_dim >= ndim:
        raise IndexError("Dimension out of range")
    if end_dim < 0 or end_dim >= ndim:
        raise IndexError("Dimension out of range")
    if start_dim > end_dim:
        raise RuntimeError("flatten() has invalid args: start_dim cannot come after end_dim")

    for i in range(start_dim, end_dim + 1):
        flattened *= self.shape[i]
    new_shape = self.shape[:start_dim] + (flattened,) + self.shape[end_dim + 1:]
    return self.reshape(new_shape)


def tensor_t(self):
    cdef Py_ssize_t ndim = len(self.shape)
    if ndim > 2:
        raise RuntimeError(f"t() expects a tensor with <= 2 dimensions, but self is {ndim}D")
    if ndim < 2:
        return self
    return self.transpose(0, 1)


def tensor_as_strided(self, size, stride, storage_offset=None):
    cdef object offset = storage_offset if storage_offset is not None else self.offset
    _ensure_view_factory_ref()
    return _cy_make_view_tensor_fn(self, self._storage, tuple(size), tuple(stride), offset)


def tensor_size(self, dim=None):
    cdef Py_ssize_t ndim

    if dim is None:
        return self.shape
    ndim = len(self.shape)
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise IndexError("Dimension out of range")
    return self.shape[dim]


def tensor_dim(self):
    return self._ndim


def tensor_retain_grad(self):
    self._retain_grad = True


def tensor_requires_grad_(self, requires_grad=True):
    self.requires_grad = bool(requires_grad)
    if not self.requires_grad:
        self.grad_fn = None
    return self


def tensor_register_hook(self, hook):
    cdef object hooks
    cdef object handle
    if not callable(hook):
        raise TypeError("hook must be callable")
    hooks = getattr(self, "_backward_hooks", None)
    if hooks is None:
        hooks = {}
        self._backward_hooks = hooks
    _ensure_hook_handle_ref()
    handle = _HookHandle_cls(hooks)
    hooks[handle.id] = hook
    return handle


def tensor_is_view(self):
    return self._base is not None


def tensor_check_inplace(self):
    cdef object view_meta
    cdef object creation_mode
    cdef object creation_kind
    cdef object grad_fn_name
    cdef object display_name

    _ensure_grad_mode_ref()
    if not _is_grad_enabled_fn():
        return
    if not self.requires_grad:
        return
    if self._is_view() and self._base is not None:
        view_meta = getattr(self, "_view_meta", None) or {}
        creation_mode = view_meta.get("creation_mode")
        creation_kind = view_meta.get("creation_kind")
        grad_fn_name = self.grad_fn.name() if self.grad_fn is not None and hasattr(self.grad_fn, "name") else "<unknown>"
        if creation_kind == "multi_view":
            raise RuntimeError(
                f"Output 0 of {grad_fn_name} is a view and is being modified inplace. This view is the output of a function that returns multiple views. Such functions do not allow the output views to be modified inplace. You should replace the inplace operation by an out-of-place one."
            )
        if creation_kind == "custom_function":
            display_name = grad_fn_name.removesuffix("Backward0") if grad_fn_name.endswith("Backward0") else grad_fn_name
            raise RuntimeError(
                f"Output 0 of {display_name} is a view and is being modified inplace. This view was created inside a custom Function (or because an input was returned as-is) and the autograd logic to handle view+inplace would override the custom backward associated with the custom Function, leading to incorrect gradients. This behavior is forbidden. You can fix this by cloning the output of the custom Function."
            )
        if creation_mode == "no_grad":
            if creation_kind == "view_of_view":
                raise RuntimeError(
                    "a view of a view which is being modified inside the no_grad block."
                )
            if creation_kind == "view":
                raise RuntimeError(
                    "A view was created in no_grad mode and is being modified inplace with grad mode enabled."
                )
        if creation_mode == "inference_mode":
            if creation_kind == "view_of_view":
                raise RuntimeError(
                    "a view of a view which is being modified inside the inference_mode."
                )
            if creation_kind == "view":
                raise RuntimeError(
                    "A view was created in inference_mode and is being modified inplace in normal mode."
                )
        if self._base.grad_fn is None and self._base.requires_grad:
            raise RuntimeError("a view of a leaf Variable that requires grad is being used in an in-place operation.")
    if self.grad_fn is None and not self._is_view():
        raise RuntimeError("a leaf Variable that requires grad is being used in an in-place operation.")


def tensor_to_dtype(self, dtype):
    cdef object arr
    cdef object src_dtype
    cdef object target_np
    cdef object storage
    cdef object stride

    _ensure_conversion_refs()
    _ensure_tensor_factory_ref()

    if self.device.type == "cpu":
        arr = self._numpy_view()
        src_dtype = self.dtype
        target_np = _to_numpy_dtype_fn(dtype)
        if src_dtype == _bfloat16_dtype:
            arr = _bf16_to_f32_fn(arr)
        if dtype == _bfloat16_dtype:
            arr = arr.astype(np.float32)
            arr = _f32_to_bf16_fn(arr)
        else:
            arr = arr.astype(target_np)
        storage = _typed_storage_from_numpy_fn(arr, dtype, device=self.device)
        stride = tuple(np.array(arr.strides) // arr.itemsize)
        return _cy_make_tensor_from_storage_fn(storage, arr.shape, stride, 0, False)
    elif self.device.type == "npu":
        return _cast_tensor_dtype_npu_fn(self, dtype)
    elif self.device.type == "mps":
        arr = self._numpy_view()
        src_dtype = self.dtype
        target_np = _to_numpy_dtype_fn(dtype)
        if src_dtype == _bfloat16_dtype:
            arr = _bf16_to_f32_fn(arr)
        if dtype == _bfloat16_dtype:
            arr = arr.astype(np.float32)
            arr = _f32_to_bf16_fn(arr)
        else:
            arr = arr.astype(target_np)
        storage = _mps_typed_storage_from_numpy_fn(np.ascontiguousarray(arr), dtype, device=self.device)
        stride = tuple(np.array(arr.strides) // arr.itemsize) if arr.ndim > 0 else ()
        return _cy_make_tensor_from_storage_fn(storage, arr.shape, stride, 0, False)
    elif self.device.type == "meta":
        storage = _meta_typed_storage_from_shape_fn(self.shape, dtype, device=self.device)
        return _cy_make_tensor_from_storage_fn(storage, self.shape, _contiguous_stride_tuple(self.shape), 0, False)
    else:
        raise RuntimeError(
            f"dtype conversion not yet supported on device {self.device.type}"
        )


def tensor_cpu(self, memory_format=None):
    if memory_format is None:
        return self.to("cpu")
    return self.to("cpu", memory_format=memory_format)


def tensor_npu(self, device=None, non_blocking=False, memory_format=None):
    if device is None:
        device = "npu"
    return self.to(device, non_blocking=non_blocking, memory_format=memory_format)


def tensor_mps(self, memory_format=None):
    if memory_format is None:
        return self.to("mps")
    return self.to("mps", memory_format=memory_format)


def tensor_cuda(self, device=None, non_blocking=False, memory_format=None):
    cdef object target
    if device is None:
        target = "cuda"
    elif isinstance(device, str):
        target = device
    else:
        target = f"cuda:{int(device)}"
    return self.to(target, non_blocking=non_blocking, memory_format=memory_format)


cdef inline tuple _contiguous_stride_tuple(tuple shape):
    cdef Py_ssize_t i
    cdef Py_ssize_t ndim = len(shape)
    cdef list strides = [0] * ndim
    cdef Py_ssize_t acc = 1
    for i in range(ndim - 1, -1, -1):
        strides[i] = acc
        acc *= shape[i]
    return tuple(strides)
