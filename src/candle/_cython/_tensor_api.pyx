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
cdef object _functional_add_fn = None
cdef object _functional_sub_fn = None
cdef object _functional_expand_fn = None
cdef object _functional_expand_copy_fn = None
cdef object _functional_sum_to_size_fn = None
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
cdef object _pinned_cpu_typed_storage_from_numpy_fn = None
cdef object _npu_available_fn = None


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


cdef inline void _ensure_functional_add_sub_ref():
    global _functional_add_fn, _functional_sub_fn
    if _functional_add_fn is None:
        from candle._functional import add as _fadd, sub as _fsub
        _functional_add_fn = _fadd
        _functional_sub_fn = _fsub


cdef inline void _ensure_functional_expand_ref():
    global _functional_expand_fn, _functional_expand_copy_fn
    if _functional_expand_fn is None:
        from candle._functional import expand as _fexpand, expand_copy as _fexpand_copy
        _functional_expand_fn = _fexpand
        _functional_expand_copy_fn = _fexpand_copy


cdef inline void _ensure_functional_sum_to_size_ref():
    global _functional_sum_to_size_fn
    if _functional_sum_to_size_fn is None:
        from candle._functional import sum_to_size as _fsum_to_size
        _functional_sum_to_size_fn = _fsum_to_size


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
    global _mps_typed_storage_from_numpy_fn, _pinned_cpu_typed_storage_from_numpy_fn
    global _to_numpy_dtype_fn, _bfloat16_dtype, _bf16_to_f32_fn, _f32_to_bf16_fn
    global _cast_tensor_dtype_npu_fn, _npu_available_fn

    if _typed_storage_from_numpy_fn is None:
        from candle._storage import (
            typed_storage_from_numpy,
            meta_typed_storage_from_shape,
            mps_typed_storage_from_numpy,
            pinned_cpu_typed_storage_from_numpy,
        )
        from candle._dtype import to_numpy_dtype, bfloat16
        from candle._tensor import _bf16_to_f32, _f32_to_bf16
        from candle._backends.npu.ops._helpers import _cast_tensor_dtype
        from candle import npu as npu_api

        _typed_storage_from_numpy_fn = typed_storage_from_numpy
        _meta_typed_storage_from_shape_fn = meta_typed_storage_from_shape
        _mps_typed_storage_from_numpy_fn = mps_typed_storage_from_numpy
        _pinned_cpu_typed_storage_from_numpy_fn = pinned_cpu_typed_storage_from_numpy
        _to_numpy_dtype_fn = to_numpy_dtype
        _bfloat16_dtype = bfloat16
        _bf16_to_f32_fn = _bf16_to_f32
        _f32_to_bf16_fn = _f32_to_bf16
        _cast_tensor_dtype_npu_fn = _cast_tensor_dtype
        _npu_available_fn = npu_api.is_available


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


def tensor_set_device_from_storage(self, dev):
    cdef object dt
    cdef object idx
    cdef int devt
    cdef unsigned int dk

    _ensure_base()
    self._device_obj = dev
    dt = getattr(dev, "type", str(dev))
    devt = _BaseTensor._DEVICE_MAP.get(dt, -1)
    self._device_type = devt
    idx = getattr(dev, "index", None)
    self._device_index = idx if idx is not None else -1

    if devt == 0:
        dk = _BaseTensor._DK_CPU
    elif devt == 1:
        dk = _BaseTensor._DK_NPU
    elif devt == 2:
        dk = _BaseTensor._DK_CUDA
    elif devt == 3:
        dk = _BaseTensor._DK_MPS
    elif devt == 4:
        dk = _BaseTensor._DK_META
    else:
        dk = _BaseTensor._DK_CPU
    if self.requires_grad:
        dk |= _BaseTensor._DK_ADINPLACEORVIEW | _BaseTensor._DK_AUTOGRAD
        if devt == 0:
            dk |= _BaseTensor._DK_AUTOGRAD_CPU
        elif devt == 1:
            dk |= _BaseTensor._DK_AUTOGRAD_NPU
        elif devt == 2:
            dk |= _BaseTensor._DK_AUTOGRAD_CUDA
        elif devt == 3:
            dk |= _BaseTensor._DK_AUTOGRAD_MPS
        elif devt == 4:
            dk |= _BaseTensor._DK_AUTOGRAD_META
    self._dispatch_keys = dk


def tensor_set_dtype_from_storage(self, dtype):
    cdef object name
    self._dtype_obj = dtype
    self._itemsize = getattr(dtype, "itemsize", 4)
    name = getattr(dtype, "name", "")
    self._dtype_code = {
        "float32": 0,
        "float16": 1,
        "float64": 2,
        "bfloat16": 3,
        "int32": 4,
        "int64": 5,
        "int16": 6,
        "int8": 7,
        "uint8": 8,
        "bool": 9,
    }.get(name, -1)


def tensor_delattr(self, name):
    if name == "grad":
        object.__setattr__(self, "grad", None)
        return
    if name in {"data", "requires_grad", "_grad_fn", "grad_fn", "_backward_hooks"}:
        raise RuntimeError(f"cannot delete {name}")
    object.__delattr__(self, name)


def tensor_set_data(self, new_data):
    _ensure_base()
    if not isinstance(new_data, _BaseTensor):
        raise TypeError(f"data must be a Tensor, got {type(new_data).__name__}")
    if new_data.shape != self.shape:
        raise RuntimeError(f"shape mismatch: expected {self.shape}, got {new_data.shape}")
    if new_data.dtype != self.dtype:
        raise RuntimeError(f"dtype mismatch: expected {self.dtype}, got {new_data.dtype}")
    self._storage = new_data._storage
    self.stride = new_data.stride
    self.offset = new_data.offset
    self._bump_version()


def tensor_fw_get(self, level):
    cdef object tangents = getattr(self, "_fw_tangents", None)
    if not tangents:
        return None
    return tangents.get(level)


def tensor_fw_set(self, level, tangent):
    cdef object tangents = getattr(self, "_fw_tangents", None)
    if tangents is None:
        tangents = {}
        self._fw_tangents = tangents
    tangents[level] = tangent


def tensor_fw_clear(self, level):
    cdef object tangents = getattr(self, "_fw_tangents", None)
    if not tangents:
        return
    tangents.pop(level, None)
    if not tangents:
        self._fw_tangents = {}


def tensor_fw_has(self, level):
    cdef object tangents = getattr(self, "_fw_tangents", None)
    return bool(tangents) and level in tangents


def tensor_untyped_storage(self):
    return self._storage.untyped_storage()


def tensor_record_stream(self, stream):
    cdef object alloc
    if self.device.type != "npu":
        return
    from candle._backends.npu import allocator as npu_allocator
    alloc = npu_allocator.get_allocator(self.device.index or 0)
    alloc.record_stream(self.storage().data_ptr(), stream.stream)


def tensor_is_pinned(self):
    return self._storage.is_pinned()

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


def tensor_isub(self, other):
    self._check_inplace()
    self.sub_(other)
    return self


def tensor_imul(self, other):
    self._check_inplace()
    self.mul_(other)
    return self


def tensor_itruediv(self, other):
    self._check_inplace()
    self.div_(other)
    return self


def tensor_baddbmm(self, batch1, batch2, beta=1, alpha=1):
    _ensure_dispatch_ref()
    return _dispatch_fn("baddbmm", self.device.type, self, batch1, batch2, beta=beta, alpha=alpha)


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


def tensor_getitem(self, key):
    _ensure_dispatch_ref()
    return _dispatch_fn("getitem", self.device.type, self, key)


def tensor_setitem(self, key, value):
    _ensure_dispatch_ref()
    self._check_inplace()
    _dispatch_fn("setitem", self.device.type, self, key, value)


def tensor_add_(self, other, alpha=1):
    cdef object rhs = other
    _ensure_dispatch_ref()
    _ensure_functional_refs()
    self._check_inplace()
    if alpha != 1:
        rhs = _mul_fn(other, alpha)
    return _dispatch_fn("add_", self.device.type, self, rhs)


def tensor_mul_(self, other):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("mul_", self.device.type, self, other)


def tensor_relu_(self):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("relu_", self.device.type, self)


def tensor_zero_(self):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("zero_", self.device.type, self)


def tensor_fill_(self, value):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("fill_", self.device.type, self, value)


def tensor_copy_(self, src):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("copy_", self.device.type, self, src)


def tensor_abs_(self):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("abs_", self.device.type, self)


def tensor_neg_(self):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("neg_", self.device.type, self)


def tensor_exp_(self):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("exp_", self.device.type, self)


def tensor_log_(self):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("log_", self.device.type, self)


def tensor_log2_(self):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("log2_", self.device.type, self)


def tensor_log10_(self):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("log10_", self.device.type, self)


def tensor_sqrt_(self):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("sqrt_", self.device.type, self)


def tensor_sin_(self):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("sin_", self.device.type, self)


def tensor_cos_(self):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("cos_", self.device.type, self)


def tensor_tan_(self):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("tan_", self.device.type, self)


def tensor_tanh_(self):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("tanh_", self.device.type, self)


def tensor_sigmoid_(self):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("sigmoid_", self.device.type, self)


def tensor_floor_(self):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("floor_", self.device.type, self)


def tensor_ceil_(self):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("ceil_", self.device.type, self)


def tensor_round_(self):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("round_", self.device.type, self)


def tensor_trunc_(self):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("trunc_", self.device.type, self)


def tensor_pow_(self, exponent):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("pow_", self.device.type, self, exponent)


def tensor_reciprocal_(self):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("reciprocal_", self.device.type, self)


def tensor_erfinv_(self):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("erfinv_", self.device.type, self)


def tensor_sub_(self, other, alpha=1):
    cdef object rhs = other
    _ensure_dispatch_ref()
    _ensure_functional_refs()
    self._check_inplace()
    if alpha != 1:
        rhs = _mul_fn(other, alpha)
    return _dispatch_fn("sub_", self.device.type, self, rhs)


def tensor_clamp_(self, min=None, max=None):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("clamp_", self.device.type, self, min, max)


def tensor_uniform_(self, low=0.0, high=1.0, generator=None):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("uniform_", self.device.type, self, low, high, generator=generator)


def tensor_normal_(self, mean=0.0, std=1.0, generator=None):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("normal_", self.device.type, self, mean, std, generator=generator)


def tensor_random_(self, from_=0, to=None, generator=None):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("random_", self.device.type, self, from_, to, generator=generator)


def tensor_randint_(self, low, high=None, generator=None):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("randint_", self.device.type, self, low, high, generator=generator)


def tensor_bernoulli_(self, p=0.5, generator=None):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("bernoulli_", self.device.type, self, p, generator=generator)


def tensor_exponential_(self, lambd=1.0, generator=None):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("exponential_", self.device.type, self, lambd, generator=generator)


def tensor_log_normal_(self, mean=1.0, std=2.0, generator=None):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("log_normal_", self.device.type, self, mean, std, generator=generator)


def tensor_cauchy_(self, median=0.0, sigma=1.0, generator=None):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("cauchy_", self.device.type, self, median, sigma, generator=generator)


def tensor_geometric_(self, p, generator=None):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("geometric_", self.device.type, self, p, generator=generator)


def tensor_transpose_(self, dim0, dim1):
    cdef Py_ssize_t ndim = len(self.shape)
    cdef Py_ssize_t d0 = dim0 if dim0 >= 0 else dim0 + ndim
    cdef Py_ssize_t d1 = dim1 if dim1 >= 0 else dim1 + ndim
    cdef list shape
    cdef list stride

    if d0 < 0 or d0 >= ndim or d1 < 0 or d1 >= ndim:
        raise IndexError("Dimension out of range")
    shape = list(self.shape)
    stride = list(self.stride)
    shape[d0], shape[d1] = shape[d1], shape[d0]
    stride[d0], stride[d1] = stride[d1], stride[d0]
    return self.as_strided_(tuple(shape), tuple(stride))


def tensor_t_(self):
    cdef Py_ssize_t ndim = len(self.shape)
    if ndim > 2:
        raise RuntimeError(f"t_() expects a tensor with <= 2 dimensions, but self is {ndim}D")
    self._check_inplace()
    if ndim < 2:
        return self
    return self.transpose_(0, 1)


def tensor_squeeze_(self, dim=None):
    cdef Py_ssize_t ndim
    cdef list shape
    cdef list stride
    cdef list pairs
    cdef object targets
    cdef object item
    cdef Py_ssize_t d

    if dim is not None:
        if isinstance(dim, (list, tuple)):
            if dim:
                ndim = len(self.shape)
                targets = set()
                for item in dim:
                    d = item if item >= 0 else item + ndim
                    targets.add(d)
                pairs = [
                    (s, st)
                    for idx, (s, st) in enumerate(zip(self.shape, self.stride))
                    if idx not in targets or s != 1
                ]
                shape = [p[0] for p in pairs]
                stride = [p[1] for p in pairs]
            else:
                shape = list(self.shape)
                stride = list(self.stride)
        else:
            d = dim if dim >= 0 else dim + len(self.shape)
            shape = list(self.shape)
            stride = list(self.stride)
            if 0 <= d < len(shape) and shape[d] == 1:
                del shape[d]
                del stride[d]
    else:
        pairs = [(s, st) for s, st in zip(self.shape, self.stride) if s != 1]
        shape = [p[0] for p in pairs]
        stride = [p[1] for p in pairs]
    return self.as_strided_(tuple(shape), tuple(stride))


def tensor_unsqueeze_(self, dim):
    cdef Py_ssize_t ndim = len(self.shape)
    cdef Py_ssize_t d = dim if dim >= 0 else dim + ndim + 1
    cdef list shape
    cdef list stride
    cdef Py_ssize_t new_stride

    if d < 0 or d > ndim:
        raise IndexError("Dimension out of range")
    shape = list(self.shape)
    stride = list(self.stride)
    new_stride = stride[d] * shape[d] if d < ndim else 1
    shape.insert(d, 1)
    stride.insert(d, new_stride)
    return self.as_strided_(tuple(shape), tuple(stride))


def tensor_as_strided_(self, size, stride, storage_offset=None):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("as_strided_", self.device.type, self, size, stride, storage_offset)


def tensor_swapdims_(self, dim0, dim1):
    return self.transpose_(dim0, dim1)


def tensor_swapaxes_(self, axis0, axis1):
    return self.swapdims_(axis0, axis1)


def tensor_scatter_(self, dim, index, src):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("scatter_", self.device.type, self, dim, index, src)


def tensor_scatter_add_(self, dim, index, src):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("scatter_add_", self.device.type, self, dim, index, src)


def tensor_masked_fill_(self, mask, value):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("masked_fill_", self.device.type, self, mask, value)


def tensor_masked_scatter_(self, mask, source):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("masked_scatter_", self.device.type, self, mask, source)


def tensor_index_put_(self, indices, values, accumulate=False):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("index_put_", self.device.type, self, indices, values, accumulate)


def tensor_index_copy_(self, dim, index, source):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("index_copy_", self.device.type, self, dim, index, source)


def tensor_index_fill_(self, dim, index, value):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("index_fill_", self.device.type, self, dim, index, value)


def tensor_index_add_(self, dim, index, source, alpha=1.0):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("index_add_", self.device.type, self, dim, index, source, alpha)


def tensor_numpy_view(self):
    cdef object base
    cdef object itemsize
    cdef tuple strides
    if self.device.type == "meta":
        raise RuntimeError("meta tensor has no data")
    if self.device.type != "cpu":
        return self.to("cpu")._numpy_view()
    base = self._storage.data.ravel()
    itemsize = base.itemsize
    strides = tuple(s * itemsize for s in self.stride)
    return np.lib.stride_tricks.as_strided(
        base[self.offset:], shape=self.shape, strides=strides
    )


def tensor_numpy(self):
    _flush_pending(self)
    if self.device.type == "meta":
        raise RuntimeError("meta tensor has no data")
    if self.device.type != "cpu":
        raise RuntimeError("numpy() is only available for CPU tensors")
    return self._numpy_view()


def tensor_pin_memory(self):
    cdef object storage
    _ensure_conversion_refs()
    _ensure_tensor_factory_ref()
    if self.device.type != "cpu":
        raise RuntimeError("pin_memory only supports CPU tensors")
    if not _npu_available_fn():
        raise RuntimeError("Cannot access accelerator device when none is available.")
    if self.is_pinned():
        return self
    storage = _pinned_cpu_typed_storage_from_numpy_fn(self._numpy_view(), self.dtype, device=self.device)
    return _cy_make_tensor_from_storage_fn(storage, self.shape, self.stride, self.offset, self.requires_grad)


def tensor_new_empty(self, size, *, dtype=None, device=None, requires_grad=False):
    from candle._creation import empty
    cdef object dt = dtype if dtype is not None else self.dtype
    cdef object dev = device if device is not None else self.device
    return empty(size, dtype=dt, device=dev)


def tensor_new_tensor(self, data, *, dtype=None, device=None, requires_grad=False):
    from candle._creation import tensor
    cdef object dt = dtype if dtype is not None else self.dtype
    cdef object dev = device if device is not None else self.device
    return tensor(data, dtype=dt, device=dev)


def tensor_new_empty_strided(self, size, stride, *, dtype=None, device=None, requires_grad=False):
    from candle._creation import empty
    cdef object dt = dtype if dtype is not None else self.dtype
    cdef object dev = device if device is not None else self.device
    cdef Py_ssize_t numel = 1
    cdef object arr
    cdef object storage
    cdef object t
    cdef object s

    if dev.type == "cpu":
        _ensure_conversion_refs()
        _ensure_tensor_factory_ref()
        for s in size:
            numel *= s
        arr = np.empty(numel, dtype=_to_numpy_dtype_fn(dt))
        storage = _typed_storage_from_numpy_fn(arr, dt, device=dev)
        return _cy_make_tensor_from_storage_fn(storage, tuple(size), tuple(stride), 0, requires_grad)
    t = empty(size, dtype=dt, device=dev)
    return t


def tensor_ones_like(self):
    cdef object storage
    cdef object arr
    cdef object stride
    cdef object tensor

    _ensure_conversion_refs()
    _ensure_tensor_factory_ref()

    if self.device.type == "meta":
        storage = _meta_typed_storage_from_shape_fn(self.shape, self.dtype, device=self.device)
        return _cy_make_tensor_from_storage_fn(storage, self.shape, self.stride, 0, False)
    arr = np.ones(self.shape, dtype=_to_numpy_dtype_fn(self.dtype))
    storage = _typed_storage_from_numpy_fn(arr, self.dtype, device=self.device if self.device.type == "cpu" else None)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    tensor = _cy_make_tensor_from_storage_fn(storage, arr.shape, stride, 0, False)
    if self.device.type != "cpu":
        return tensor.to(self.device)
    return tensor


def tensor_new_ones(self, size, dtype=None, device=None):
    from candle._creation import ones
    if dtype is None:
        dtype = self.dtype
    if device is None:
        device = self.device
    return ones(size, dtype=dtype, device=device)


def tensor_new_zeros(self, size, dtype=None, device=None):
    from candle._creation import zeros
    if dtype is None:
        dtype = self.dtype
    if device is None:
        device = self.device
    return zeros(size, dtype=dtype, device=device)


def tensor_new_full(self, size, fill_value, *, dtype=None, device=None, requires_grad=False):
    from candle._creation import full
    cdef object dt = dtype if dtype is not None else self.dtype
    cdef object dev = device if device is not None else self.device
    return full(size, fill_value, dtype=dt, device=dev)


def tensor_div_(self, other):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("div_", self.device.type, self, other)


def tensor_unflatten(self, dim, sizes):
    cdef Py_ssize_t ndim = len(self.shape)
    if dim < 0:
        dim += ndim
    return self.view(self.shape[:dim] + tuple(sizes) + self.shape[dim + 1:])


def tensor_bitwise_and_(self, other):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("bitwise_and_", self.device.type, self, other)


def tensor_bitwise_or_(self, other):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("bitwise_or_", self.device.type, self, other)


def tensor_bitwise_xor_(self, other):
    _ensure_dispatch_ref()
    self._check_inplace()
    return _dispatch_fn("bitwise_xor_", self.device.type, self, other)


def tensor_type(self, dtype=None):
    cdef object dt
    if dtype is None:
        return f"torch.{self.dtype.name.capitalize()}Tensor"
    if isinstance(dtype, str):
        _ensure_dtype_ref()
        from candle._dtype import float32, float64, float16, bfloat16, int64, int32, int16, int8, uint8
        from candle._dtype import bool as dtype_bool
        _type_map = {
            "torch.FloatTensor": float32,
            "torch.DoubleTensor": float64,
            "torch.HalfTensor": float16,
            "torch.BFloat16Tensor": bfloat16,
            "torch.LongTensor": int64,
            "torch.IntTensor": int32,
            "torch.ShortTensor": int16,
            "torch.CharTensor": int8,
            "torch.ByteTensor": uint8,
            "torch.BoolTensor": dtype_bool,
        }
        dt = _type_map.get(dtype) or _from_name_fn(dtype)
        if dt is None:
            raise RuntimeError(f"Unknown type: {dtype}")
        return self._to_dtype(dt)
    return self._to_dtype(dtype)


def tensor_type_as(self, other):
    return self.to(other.dtype)


def tensor_reshape_as(self, other):
    return self.reshape(other.shape)


def tensor_put_(self, indices, values, accumulate=False):
    cdef object cont
    cdef object numel_idx
    cdef object shape
    cdef object idx
    cdef object val
    cdef list multi_idx
    cdef object tmp
    cdef object d
    cdef object i

    self._check_inplace()
    if not self.is_contiguous():
        cont = self.contiguous()
        self._storage = cont._storage
        self.stride = cont.stride
    numel_idx = indices.numel()
    shape = self.shape
    for i in range(numel_idx):
        idx = int(indices.reshape((numel_idx,))[i].item())
        val = values.reshape((numel_idx,))[i]
        multi_idx = []
        tmp = idx
        for d in reversed(shape):
            multi_idx.append(tmp % d)
            tmp //= d
        multi_idx = list(reversed(multi_idx))
        if accumulate:
            self[tuple(multi_idx)] = self[tuple(multi_idx)] + val
        else:
            self[tuple(multi_idx)] = val
    self._bump_version()
    return self


def tensor_scatter_add(self, dim, index, src):
    cdef object out = self.clone()
    out.scatter_add_(dim, index, src)
    return out


def tensor_index_fill(self, dim, index, value):
    cdef object out = self.clone()
    out.index_fill_(dim, index, value)
    return out


def tensor_index_copy(self, dim, index, source):
    cdef object out = self.clone()
    out.index_copy_(dim, index, source)
    return out


def tensor_index_add(self, dim, index, source, alpha=1):
    cdef object out = self.clone()
    out.index_add_(dim, index, source, alpha)
    return out


def tensor_permute(self, *dims):
    _ensure_dispatch_ref()
    if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
        dims = tuple(dims[0])
    return _dispatch_fn("permute", self.device.type, self, dims)


def tensor_mean(self, dim=None, keepdim=False, dtype=None, axis=None):
    from candle._functional import mean as mean_dispatch
    if axis is not None:
        dim = axis
    return mean_dispatch(self, dim=dim, keepdim=keepdim, dtype=dtype)


def tensor_std(self, dim=None, keepdim=False, unbiased=True, axis=None):
    from candle._functional import std as std_dispatch
    if axis is not None:
        dim = axis
    return std_dispatch(self, dim=dim, keepdim=keepdim, unbiased=unbiased)


def tensor_repeat(self, *repeats):
    from candle._functional import repeat as repeat_dispatch
    if len(repeats) == 1 and isinstance(repeats[0], (tuple, list)):
        repeats = tuple(repeats[0])
    return repeat_dispatch(self, repeats)


def tensor_tile(self, *dims):
    from candle._functional import tile as tile_dispatch
    if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
        dims = tuple(dims[0])
    return tile_dispatch(self, dims)


def tensor_flip(self, dims):
    from candle._functional import flip as flip_dispatch
    if isinstance(dims, int):
        dims = [dims]
    return flip_dispatch(self, dims)


def tensor_logsumexp(self, dim, keepdim=False):
    _ensure_dispatch_ref()
    return _dispatch_fn("logsumexp", self.device.type, self, dim, keepdim)


def tensor_trace(self):
    _ensure_dispatch_ref()
    return _dispatch_fn("trace", self.device.type, self)


def tensor_det(self):
    _ensure_dispatch_ref()
    return _dispatch_fn("det", self.device.type, self)


def tensor_matrix_power(self, n):
    _ensure_dispatch_ref()
    return _dispatch_fn("matrix_power", self.device.type, self, n)


def tensor_dist(self, other, p=2):
    _ensure_dispatch_ref()
    return _dispatch_fn("dist", self.device.type, self, other, p)


def tensor_renorm(self, p, dim, maxnorm):
    _ensure_dispatch_ref()
    return _dispatch_fn("renorm", self.device.type, self, p, dim, maxnorm)


def tensor_nansum(self, dim=None, keepdim=False):
    _ensure_dispatch_ref()
    return _dispatch_fn("nansum", self.device.type, self, dim, keepdim)


def tensor_nanmean(self, dim=None, keepdim=False):
    _ensure_dispatch_ref()
    return _dispatch_fn("nanmean", self.device.type, self, dim, keepdim)


def tensor_argwhere(self):
    _ensure_dispatch_ref()
    return _dispatch_fn("argwhere", self.device.type, self)


def tensor_logical_and(self, other):
    _ensure_dispatch_ref()
    return _dispatch_fn("logical_and", self.device.type, self, other)


def tensor_logical_or(self, other):
    _ensure_dispatch_ref()
    return _dispatch_fn("logical_or", self.device.type, self, other)


def tensor_logical_xor(self, other):
    _ensure_dispatch_ref()
    return _dispatch_fn("logical_xor", self.device.type, self, other)


def tensor_logical_not(self):
    _ensure_dispatch_ref()
    return _dispatch_fn("logical_not", self.device.type, self)


def tensor_bitwise_and(self, other):
    _ensure_dispatch_ref()
    return _dispatch_fn("bitwise_and", self.device.type, self, other)


def tensor_bitwise_or(self, other):
    _ensure_dispatch_ref()
    return _dispatch_fn("bitwise_or", self.device.type, self, other)


def tensor_bitwise_xor(self, other):
    _ensure_dispatch_ref()
    return _dispatch_fn("bitwise_xor", self.device.type, self, other)


def tensor_bitwise_not(self):
    _ensure_dispatch_ref()
    return _dispatch_fn("bitwise_not", self.device.type, self)


def tensor_logsumexp(self, dim, keepdim=False):
    _ensure_dispatch_ref()
    return _dispatch_fn("logsumexp", self.device.type, self, dim, keepdim)


def tensor_trace(self):
    _ensure_dispatch_ref()
    return _dispatch_fn("trace", self.device.type, self)


def tensor_det(self):
    _ensure_dispatch_ref()
    return _dispatch_fn("det", self.device.type, self)


def tensor_matrix_power(self, n):
    _ensure_dispatch_ref()
    return _dispatch_fn("matrix_power", self.device.type, self, n)


def tensor_dist(self, other, p=2):
    _ensure_dispatch_ref()
    return _dispatch_fn("dist", self.device.type, self, other, p)


def tensor_renorm(self, p, dim, maxnorm):
    _ensure_dispatch_ref()
    return _dispatch_fn("renorm", self.device.type, self, p, dim, maxnorm)


def tensor_nansum(self, dim=None, keepdim=False):
    _ensure_dispatch_ref()
    return _dispatch_fn("nansum", self.device.type, self, dim, keepdim)


def tensor_nanmean(self, dim=None, keepdim=False):
    _ensure_dispatch_ref()
    return _dispatch_fn("nanmean", self.device.type, self, dim, keepdim)


def tensor_argwhere(self):
    _ensure_dispatch_ref()
    return _dispatch_fn("argwhere", self.device.type, self)


def tensor_movedim(self, source, destination):
    _ensure_dispatch_ref()
    return _dispatch_fn("movedim", self.device.type, self, source, destination)


def tensor_diagonal(self, offset=0, dim1=0, dim2=1):
    _ensure_dispatch_ref()
    return _dispatch_fn("diagonal", self.device.type, self, offset, dim1, dim2)


def tensor_unbind(self, dim=0):
    _ensure_dispatch_ref()
    return _dispatch_fn("unbind", self.device.type, self, dim)


def tensor_vsplit(self, split_size_or_sections):
    _ensure_dispatch_ref()
    return _dispatch_fn("vsplit", self.device.type, self, split_size_or_sections)


def tensor_hsplit(self, split_size_or_sections):
    _ensure_dispatch_ref()
    return _dispatch_fn("hsplit", self.device.type, self, split_size_or_sections)


def tensor_dsplit(self, split_size_or_sections):
    _ensure_dispatch_ref()
    return _dispatch_fn("dsplit", self.device.type, self, split_size_or_sections)


def tensor_take_along_dim(self, indices, dim):
    _ensure_dispatch_ref()
    return _dispatch_fn("take_along_dim", self.device.type, self, indices, dim)


def tensor_cummin(self, dim):
    _ensure_dispatch_ref()
    return _dispatch_fn("cummin", self.device.type, self, dim)


def tensor_log1p(self):
    _ensure_dispatch_ref()
    return _dispatch_fn("log1p", self.device.type, self)


def tensor_expm1(self):
    _ensure_dispatch_ref()
    return _dispatch_fn("expm1", self.device.type, self)


def tensor_lt(self, other):
    _ensure_dispatch_ref()
    return _dispatch_fn("lt", self.device.type, self, other)


def tensor_le(self, other):
    _ensure_dispatch_ref()
    return _dispatch_fn("le", self.device.type, self, other)


def tensor_gt(self, other):
    _ensure_dispatch_ref()
    return _dispatch_fn("gt", self.device.type, self, other)


def tensor_ge(self, other):
    _ensure_dispatch_ref()
    return _dispatch_fn("ge", self.device.type, self, other)


# ── unary element-wise ops ────────────────────────────────────────────────────

def tensor_abs(self):
    _ensure_dispatch_ref()
    return _dispatch_fn("abs", self.device.type, self)


def tensor_exp(self):
    _ensure_dispatch_ref()
    return _dispatch_fn("exp", self.device.type, self)


def tensor_log(self):
    _ensure_dispatch_ref()
    return _dispatch_fn("log", self.device.type, self)


def tensor_sqrt(self):
    _ensure_dispatch_ref()
    return _dispatch_fn("sqrt", self.device.type, self)


def tensor_sin(self):
    _ensure_dispatch_ref()
    return _dispatch_fn("sin", self.device.type, self)


def tensor_cos(self):
    _ensure_dispatch_ref()
    return _dispatch_fn("cos", self.device.type, self)


def tensor_tan(self):
    _ensure_dispatch_ref()
    return _dispatch_fn("tan", self.device.type, self)


def tensor_tanh(self):
    _ensure_dispatch_ref()
    return _dispatch_fn("tanh", self.device.type, self)


def tensor_sigmoid(self):
    _ensure_dispatch_ref()
    return _dispatch_fn("sigmoid", self.device.type, self)


def tensor_floor(self):
    _ensure_dispatch_ref()
    return _dispatch_fn("floor", self.device.type, self)


def tensor_ceil(self):
    _ensure_dispatch_ref()
    return _dispatch_fn("ceil", self.device.type, self)


def tensor_round(self):
    _ensure_dispatch_ref()
    return _dispatch_fn("round", self.device.type, self)


def tensor_trunc(self):
    _ensure_dispatch_ref()
    return _dispatch_fn("trunc", self.device.type, self)


def tensor_frac(self):
    _ensure_dispatch_ref()
    return _dispatch_fn("frac", self.device.type, self)


def tensor_log2(self):
    _ensure_dispatch_ref()
    return _dispatch_fn("log2", self.device.type, self)


def tensor_log10(self):
    _ensure_dispatch_ref()
    return _dispatch_fn("log10", self.device.type, self)


def tensor_exp2(self):
    _ensure_dispatch_ref()
    return _dispatch_fn("exp2", self.device.type, self)


def tensor_rsqrt(self):
    _ensure_dispatch_ref()
    return _dispatch_fn("rsqrt", self.device.type, self)


def tensor_sign(self):
    _ensure_dispatch_ref()
    return _dispatch_fn("sign", self.device.type, self)


def tensor_signbit(self):
    _ensure_dispatch_ref()
    return _dispatch_fn("signbit", self.device.type, self)


def tensor_square(self):
    _ensure_dispatch_ref()
    return _dispatch_fn("square", self.device.type, self)


def tensor_isnan(self):
    _ensure_dispatch_ref()
    return _dispatch_fn("isnan", self.device.type, self)


def tensor_isinf(self):
    _ensure_dispatch_ref()
    return _dispatch_fn("isinf", self.device.type, self)


def tensor_isfinite(self):
    _ensure_dispatch_ref()
    return _dispatch_fn("isfinite", self.device.type, self)


def tensor_sinh(self):
    _ensure_dispatch_ref()
    return _dispatch_fn("sinh", self.device.type, self)


def tensor_cosh(self):
    _ensure_dispatch_ref()
    return _dispatch_fn("cosh", self.device.type, self)


def tensor_asinh(self):
    _ensure_dispatch_ref()
    return _dispatch_fn("asinh", self.device.type, self)


def tensor_acosh(self):
    _ensure_dispatch_ref()
    return _dispatch_fn("acosh", self.device.type, self)


def tensor_atanh(self):
    _ensure_dispatch_ref()
    return _dispatch_fn("atanh", self.device.type, self)


def tensor_erf(self):
    _ensure_dispatch_ref()
    return _dispatch_fn("erf", self.device.type, self)


def tensor_erfc(self):
    _ensure_dispatch_ref()
    return _dispatch_fn("erfc", self.device.type, self)


def tensor_reciprocal(self):
    _ensure_dispatch_ref()
    return _dispatch_fn("reciprocal", self.device.type, self)


def tensor_tril(self, diagonal=0):
    _ensure_dispatch_ref()
    return _dispatch_fn("tril", self.device.type, self, diagonal)


def tensor_triu(self, diagonal=0):
    _ensure_dispatch_ref()
    return _dispatch_fn("triu", self.device.type, self, diagonal)


def tensor_diag(self, diagonal=0):
    _ensure_dispatch_ref()
    return _dispatch_fn("diag", self.device.type, self, diagonal)


# ── operator sugar ────────────────────────────────────────────────────────────

def tensor_add_method(self, other, *, alpha=1):
    _ensure_functional_add_sub_ref()
    return _functional_add_fn(self, other, alpha=alpha)


def tensor_sub_method(self, other, *, alpha=1):
    _ensure_functional_add_sub_ref()
    return _functional_sub_fn(self, other, alpha=alpha)


def tensor_mul_method(self, other):
    _ensure_dispatch_ref()
    return _dispatch_fn("mul", self.device.type, self, other)


def tensor_div_method(self, other, *, rounding_mode=None):
    _ensure_dispatch_ref()
    return _dispatch_fn("div", self.device.type, self, other)


def tensor_pow_method(self, exponent):
    _ensure_dispatch_ref()
    return _dispatch_fn("pow", self.device.type, self, exponent)


def tensor_matmul_method(self, other):
    _ensure_dispatch_ref()
    return _dispatch_fn("matmul", self.device.type, self, other)


def tensor_rsub(self, other):
    """other - self  (reflected sub)"""
    _ensure_dispatch_ref()
    neg_self = _dispatch_fn("neg", self.device.type, self)
    return _dispatch_fn("add", self.device.type, neg_self, other)


def tensor_rmul(self, other):
    _ensure_dispatch_ref()
    return _dispatch_fn("mul", self.device.type, self, other)


def tensor_truediv(self, other):
    _ensure_dispatch_ref()
    return _dispatch_fn("true_divide", self.device.type, self, other)


def tensor_rtruediv(self, other):
    _ensure_dispatch_ref()
    return _dispatch_fn("true_divide", self.device.type, other, self)


def tensor_pow_op(self, exponent):
    _ensure_dispatch_ref()
    return _dispatch_fn("pow", self.device.type, self, exponent)


def tensor_rpow(self, base):
    _ensure_dispatch_ref()
    return _dispatch_fn("pow", self.device.type, base, self)


def tensor_floordiv(self, other):
    _ensure_dispatch_ref()
    return _dispatch_fn("floor_divide", self.device.type, self, other)


def tensor_rfloordiv(self, other):
    _ensure_dispatch_ref()
    return _dispatch_fn("floor_divide", self.device.type, other, self)


def tensor_mod(self, other):
    _ensure_dispatch_ref()
    return _dispatch_fn("remainder", self.device.type, self, other)


def tensor_rmod(self, other):
    _ensure_dispatch_ref()
    return _dispatch_fn("remainder", self.device.type, other, self)


def tensor_rmatmul(self, other):
    _ensure_dispatch_ref()
    return _dispatch_fn("matmul", self.device.type, other, self)


def tensor_and(self, other):
    _ensure_dispatch_ref()
    try:
        s_bool = self.bool()
    except Exception:
        s_bool = self
    try:
        o_bool = other.bool() if hasattr(other, 'bool') else bool(other)
    except Exception:
        o_bool = other
    return _dispatch_fn("mul", self.device.type, s_bool, o_bool)


def tensor_or(self, other):
    _ensure_dispatch_ref()
    try:
        s_bool = self.bool()
    except Exception:
        s_bool = self
    try:
        o_bool = other.bool() if hasattr(other, 'bool') else bool(other)
    except Exception:
        o_bool = other
    return _dispatch_fn("add", self.device.type, s_bool, o_bool)


def tensor_xor(self, other):
    _ensure_dispatch_ref()
    try:
        s_bool = self.bool()
    except Exception:
        s_bool = self
    try:
        o_bool = other.bool() if hasattr(other, 'bool') else bool(other)
    except Exception:
        o_bool = other
    return _dispatch_fn("ne", self.device.type, s_bool, o_bool)


# ── reduction ops ─────────────────────────────────────────────────────────────

def tensor_all_method(self, dim=None, keepdim=False):
    _ensure_dispatch_ref()
    return _dispatch_fn("all", self.device.type, self, dim=dim, keepdim=keepdim)


def tensor_any_method(self, dim=None, keepdim=False):
    _ensure_dispatch_ref()
    return _dispatch_fn("any", self.device.type, self, dim=dim, keepdim=keepdim)


def tensor_sum_method(self, dim=None, keepdim=False, *, dtype=None):
    _ensure_dispatch_ref()
    return _dispatch_fn("sum", self.device.type, self, dim=dim, keepdim=keepdim, dtype=dtype)


def tensor_prod_method(self, dim=None, keepdim=False):
    _ensure_dispatch_ref()
    return _dispatch_fn("prod", self.device.type, self, dim=dim, keepdim=keepdim)


def tensor_var_method(self, dim=None, keepdim=False, unbiased=True):
    _ensure_dispatch_ref()
    return _dispatch_fn("var", self.device.type, self, dim=dim, keepdim=keepdim, unbiased=unbiased)


def tensor_var_mean_method(self, dim=None, keepdim=False, unbiased=True):
    _ensure_dispatch_ref()
    return _dispatch_fn("var_mean", self.device.type, self, dim=dim, keepdim=keepdim, unbiased=unbiased)


def tensor_norm_method(self, p=2, dim=None, keepdim=False):
    _ensure_dispatch_ref()
    return _dispatch_fn("norm", self.device.type, self, p=p, dim=dim, keepdim=keepdim)


def tensor_count_nonzero_method(self, dim=None, keepdim=False):
    _ensure_dispatch_ref()
    return _dispatch_fn("count_nonzero", self.device.type, self, dim=dim, keepdim=keepdim)


def tensor_cumsum_method(self, dim=0):
    _ensure_dispatch_ref()
    return _dispatch_fn("cumsum", self.device.type, self, dim)


def tensor_cumprod_method(self, dim=0):
    _ensure_dispatch_ref()
    return _dispatch_fn("cumprod", self.device.type, self, dim)


def tensor_cummax_method(self, dim=0):
    _ensure_dispatch_ref()
    return _dispatch_fn("cummax", self.device.type, self, dim)


def tensor_argsort_method(self, dim=-1, descending=False, stable=False):
    _ensure_dispatch_ref()
    return _dispatch_fn("argsort", self.device.type, self, dim=dim, descending=descending, stable=stable)


def tensor_sort_method(self, dim=-1, descending=False, stable=False):
    _ensure_dispatch_ref()
    return _dispatch_fn("sort", self.device.type, self, dim=dim, descending=descending, stable=stable)


def tensor_topk_method(self, k, dim=-1, largest=True, sorted=True):
    _ensure_dispatch_ref()
    return _dispatch_fn("topk", self.device.type, self, k, dim=dim, largest=largest, sorted=sorted)


# ── comparison ops ────────────────────────────────────────────────────────────

def tensor_eq_method(self, other):
    return self.__eq__(other)


def tensor_ne_method(self, other):
    return self.__ne__(other)


def tensor_allclose_method(self, other, rtol=1e-05, atol=1e-08, equal_nan=False):
    _ensure_dispatch_ref()
    return _dispatch_fn("allclose", self.device.type, self, other, rtol=rtol, atol=atol, equal_nan=equal_nan)


def tensor_isclose_method(self, other, rtol=1e-05, atol=1e-08, equal_nan=False):
    _ensure_dispatch_ref()
    return _dispatch_fn("isclose", self.device.type, self, other, rtol=rtol, atol=atol, equal_nan=equal_nan)


def tensor_equal_method(self, other):
    _ensure_dispatch_ref()
    return _dispatch_fn("equal", self.device.type, self, other)


# ── view / shape alias ops ────────────────────────────────────────────────────

def tensor_view_as(self, other):
    return self.view(other.shape)


def tensor_expand_method(self, *sizes):
    _ensure_functional_expand_ref()
    return _functional_expand_fn(self, *sizes)


def tensor_expand_as_method(self, other):
    _ensure_functional_expand_ref()
    return _functional_expand_fn(self, *other.shape)


def tensor_expand_copy_method(self, *sizes):
    _ensure_functional_expand_ref()
    return _functional_expand_copy_fn(self, sizes)


def tensor_narrow_method(self, dim, start, length):
    _ensure_dispatch_ref()
    return _dispatch_fn("narrow", self.device.type, self, dim, start, length)


def tensor_select_method(self, dim, index):
    _ensure_dispatch_ref()
    return _dispatch_fn("select", self.device.type, self, dim, index)


def tensor_unfold_method(self, dimension, size, step):
    _ensure_dispatch_ref()
    return _dispatch_fn("unfold", self.device.type, self, dimension, size, step)


def tensor_moveaxis_method(self, source, destination):
    return self.movedim(source, destination)


def tensor_swapdims_method(self, dim0, dim1):
    return self.transpose(dim0, dim1)


def tensor_swapaxes_method(self, axis0, axis1):
    return self.transpose(axis0, axis1)


# ── indexing / scatter ops ────────────────────────────────────────────────────

def tensor_gather_method(self, dim, index):
    _ensure_dispatch_ref()
    return _dispatch_fn("gather", self.device.type, self, dim, index)


def tensor_scatter_method(self, dim, index, src):
    _ensure_dispatch_ref()
    return _dispatch_fn("scatter", self.device.type, self, dim, index, src)


def tensor_index_select_method(self, dim, index):
    _ensure_dispatch_ref()
    return _dispatch_fn("index_select", self.device.type, self, dim, index)


def tensor_take_method(self, index):
    _ensure_dispatch_ref()
    return _dispatch_fn("take", self.device.type, self, index)


def tensor_masked_fill_method(self, mask, value):
    _ensure_dispatch_ref()
    return _dispatch_fn("masked_fill", self.device.type, self, mask, value)


def tensor_masked_select_method(self, mask):
    _ensure_dispatch_ref()
    return _dispatch_fn("masked_select", self.device.type, self, mask)


def tensor_index_put_method(self, indices, values, accumulate=False):
    _ensure_dispatch_ref()
    return _dispatch_fn("index_put", self.device.type, self, indices, values, accumulate)


def tensor_slice_method(self, dim, start=0, end=9223372036854775807, step=1):
    _ensure_dispatch_ref()
    return _dispatch_fn("slice", self.device.type, self, dim, start, end, step)


def tensor_slice_copy_method(self, dim, start=0, end=9223372036854775807, step=1):
    _ensure_dispatch_ref()
    return _dispatch_fn("slice_copy", self.device.type, self, dim, start, end, step)


def tensor_slice_scatter_method(self, src, dim, start=0, end=9223372036854775807, step=1):
    _ensure_dispatch_ref()
    return _dispatch_fn("slice_scatter", self.device.type, self, src, dim, start, end, step)


def tensor_nonzero_method(self, as_tuple=False):
    _ensure_dispatch_ref()
    return _dispatch_fn("nonzero", self.device.type, self, as_tuple=as_tuple)


def tensor_sum_to_size_method(self, *size):
    _ensure_functional_sum_to_size_ref()
    return _functional_sum_to_size_fn(self, *size)


cdef inline tuple _contiguous_stride_tuple(tuple shape):



    cdef Py_ssize_t i
    cdef Py_ssize_t ndim = len(shape)
    cdef list strides = [0] * ndim
    cdef Py_ssize_t acc = 1
    for i in range(ndim - 1, -1, -1):
        strides[i] = acc
        acc *= shape[i]
    return tuple(strides)
