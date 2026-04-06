# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython hot wrappers for candle._tensor.Tensor methods.

These functions are installed onto ``candle._tensor.Tensor`` when the extension
is available so the hottest Tensor forwarding paths run through compiled code
while preserving the existing Python fallback behavior in ``candle._tensor``.
"""

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


cdef inline void _flush_pending(object tensor):
    cdef object pipe

    if tensor._pending:
        _ensure_pipeline_ref()
        pipe = _current_pipeline_fn()
        if pipe is not None:
            pipe.flush()


cdef inline object _annotate_transpose_view(object source, object view):
    cdef object source_view_meta
    cdef object creation_mode
    cdef object creation_kind

    source_view_meta = getattr(source, "_view_meta", None) or {}
    from candle.autograd.grad_mode import current_creation_mode
    creation_mode = current_creation_mode() or source_view_meta.get("creation_mode")
    creation_kind = source_view_meta.get("creation_kind")
    if creation_mode is not None:
        if source._is_view():
            creation_kind = "view_of_view"
        else:
            creation_kind = "view"
    view._view_meta = {
        "op": "transpose",
        "shape": tuple(view.shape),
        "stride": tuple(view.stride),
        "offset": int(view.offset),
        "creation_mode": creation_mode,
        "creation_kind": creation_kind,
    }
    return view
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


def tensor_flatten(self, start_dim=0, end_dim=-1):
    cdef object ndim
    cdef object start
    cdef object end
    cdef object flattened
    cdef object d
    cdef object new_shape
    cdef object v
    cdef object source_view_meta
    cdef object creation_mode
    cdef object creation_kind

    ndim = len(self.shape)
    if ndim == 0:
        return self.cy_view((1,))

    start = start_dim if start_dim >= 0 else start_dim + ndim
    end = end_dim if end_dim >= 0 else end_dim + ndim
    if start < 0 or start >= ndim:
        raise IndexError("Dimension out of range")
    if end < 0 or end >= ndim:
        raise IndexError("Dimension out of range")
    if start > end:
        raise RuntimeError("flatten() has invalid args: start_dim cannot come after end_dim")

    flattened = 1
    for d in self.shape[start:end + 1]:
        flattened *= d
    new_shape = self.shape[:start] + (flattened,) + self.shape[end + 1:]

    v = self.cy_view(tuple(new_shape))
    source_view_meta = getattr(self, "_view_meta", None) or {}
    from candle.autograd.grad_mode import current_creation_mode
    creation_mode = current_creation_mode() or source_view_meta.get("creation_mode")
    creation_kind = source_view_meta.get("creation_kind")
    if creation_mode is not None:
        if self._is_view():
            creation_kind = "view_of_view"
        else:
            creation_kind = "view"
    v._view_meta = {
        "op": "flatten",
        "shape": tuple(v.shape),
        "stride": tuple(v.stride),
        "offset": int(v.offset),
        "creation_mode": creation_mode,
        "creation_kind": creation_kind,
    }
    return v


def tensor_reshape(self, *shape):
    if not shape:
        raise TypeError("reshape() missing shape arguments")
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    _ensure_functional_refs()
    return _reshape_dispatch_fn(self, shape)


def tensor_transpose(self, dim0, dim1):
    cdef object v

    if self.requires_grad:
        from candle._dispatch import dispatch
        return dispatch("transpose", self.device.type, self, dim0, dim1)
    v = self.cy_transpose(dim0, dim1)
    return _annotate_transpose_view(self, v)


def tensor_view(self, *shape):
    cdef object v
    cdef object source_view_meta
    cdef object creation_mode
    cdef object creation_kind
    cdef object infer_idx
    cdef object known_size
    cdef object shape_list
    cdef object idx
    cdef object dim
    cdef object size
    cdef object new_size

    if not shape:
        raise TypeError(
            "view() received an invalid combination of arguments - got (), but expected one of:\n"
            " * (torch.dtype dtype)\n"
            " * (tuple of ints size)\n"
        )
    if len(shape) == 1:
        if isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        else:
            shape = (shape[0],)
    else:
        shape = tuple(shape)

    if not self.is_contiguous():
        raise RuntimeError(
            "view size is not compatible with input tensor's size and stride "
            "(at least one dimension spans across two contiguous subspaces). "
            "Use .reshape(...) instead."
        )

    size = 1
    for dim in self.shape:
        size *= dim

    infer_idx = None
    known_size = 1
    shape_list = list(shape)
    for idx, dim in enumerate(shape_list):
        if dim == -1:
            if infer_idx is not None:
                raise RuntimeError("only one dimension can be inferred")
            infer_idx = idx
            continue
        known_size *= dim

    if infer_idx is not None:
        if known_size == 0 or size % known_size != 0:
            raise RuntimeError(f"shape '{list(shape)}' is invalid for input of size {size}")
        shape_list[infer_idx] = size // known_size

    shape = tuple(shape_list)
    new_size = 1
    for dim in shape:
        new_size *= dim
    if size != new_size:
        raise ValueError("view size mismatch")

    if self.requires_grad:
        from candle._dispatch import dispatch
        return dispatch("view", self.device.type, self, shape)

    v = self.cy_view(shape)
    source_view_meta = getattr(self, "_view_meta", None) or {}
    from candle.autograd.grad_mode import current_creation_mode
    creation_mode = current_creation_mode() or source_view_meta.get("creation_mode")
    creation_kind = source_view_meta.get("creation_kind")
    if creation_mode is not None:
        if self._is_view():
            creation_kind = "view_of_view"
        else:
            creation_kind = "view"
    v._view_meta = {
        "op": "view",
        "shape": tuple(v.shape),
        "stride": tuple(v.stride),
        "offset": int(v.offset),
        "creation_mode": creation_mode,
        "creation_kind": creation_kind,
    }
    from candle.autograd import forward_ad
    level = forward_ad._current_level()
    if level >= 0:
        tangent = forward_ad.get_tangent(self, level)
        if tangent is not None:
            v._fw_set(level, tangent.view(shape))
    return v


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
