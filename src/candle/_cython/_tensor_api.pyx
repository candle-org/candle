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
    out._version_value = self._version_value
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
