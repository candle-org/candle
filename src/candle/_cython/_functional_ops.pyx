# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython hot wrappers for candle._functional.

This module accelerates the most frequently used functional entry points while
preserving the pure-Python fallback functions in ``candle._functional`` for
stable ``__torch_function__`` identities and fallback behavior.
"""

# Cached reference to base Tensor class
cdef object _BaseTensor = None

# Cached Python callables
cdef object _dispatch_fn = None
cdef object _py_add_fn = None
cdef object _py_mul_fn = None
cdef object _py_matmul_fn = None
cdef object _py_relu_fn = None
cdef object _py_neg_fn = None

# NPU fast-path: cached references for direct kernel calls
cdef object _npu_add_fn = None
cdef object _npu_mul_fn = None
cdef object _grad_mode_state = None
cdef object _is_functionalize_fn = None
cdef object _current_pipeline_fn = None
cdef bint _npu_refs_loaded = False


cdef inline void _ensure_base():
    global _BaseTensor
    if _BaseTensor is None:
        from candle._tensor import Tensor
        _BaseTensor = Tensor


cdef inline void _ensure_dispatch():
    global _dispatch_fn
    if _dispatch_fn is None:
        from candle._dispatch.dispatcher import dispatch
        _dispatch_fn = dispatch


cdef inline bint _is_base_tensor(object t):
    """True if t is exactly the base Tensor class (not a subclass)."""
    _ensure_base()
    return type(t) is _BaseTensor


cdef bint _check_value(object val):
    cdef object cls
    cdef object item

    if isinstance(val, _BaseTensor) and type(val) is not _BaseTensor:
        cls = type(val)
        if cls.__torch_function__ is not _BaseTensor.__torch_function__:
            return True

    if isinstance(val, (list, tuple)):
        for item in val:
            if _check_value(item):
                return True

    return False


cdef void _collect_types(object val, object types):
    cdef object cls
    cdef object item

    if isinstance(val, _BaseTensor) and type(val) is not _BaseTensor:
        cls = type(val)
        if cls.__torch_function__ is not _BaseTensor.__torch_function__:
            types.add(cls)

    if isinstance(val, (list, tuple)):
        for item in val:
            _collect_types(item, types)


cdef inline void _ensure_originals():
    global _py_add_fn, _py_mul_fn, _py_matmul_fn, _py_relu_fn, _py_neg_fn

    _ensure_dispatch()

    if _py_add_fn is None:
        from candle._functional import _py_add, _py_mul, _py_matmul, _py_relu, _py_neg
        _py_add_fn = _py_add
        _py_mul_fn = _py_mul
        _py_matmul_fn = _py_matmul
        _py_relu_fn = _py_relu
        _py_neg_fn = _py_neg


cdef inline void _ensure_npu_refs():
    """Load NPU op refs and guard state once."""
    global _npu_add_fn, _npu_mul_fn
    global _grad_mode_state, _is_functionalize_fn, _current_pipeline_fn
    global _npu_refs_loaded

    if _npu_refs_loaded:
        return

    from candle._backends.npu.ops import add as _nadd, mul as _nmul
    from candle.autograd.grad_mode import _GRAD_MODE_STATE as _gms
    from candle._dispatch.functionalize import is_functionalize_enabled as _ife
    from candle._dispatch.pipeline import current_pipeline as _cp

    _npu_add_fn = _nadd
    _npu_mul_fn = _nmul
    _grad_mode_state = _gms
    _is_functionalize_fn = _ife
    _current_pipeline_fn = _cp
    _npu_refs_loaded = True


cdef inline bint _is_npu_tensor_pair(object a, object b):
    """True only when both operands are tensors on the NPU device."""
    cdef object a_dev = getattr(a, "device", None)
    cdef object b_dev = getattr(b, "device", None)

    if a_dev is None or b_dev is None:
        return False
    return a_dev.type == "npu" and b_dev.type == "npu"


cdef inline bint _npu_fast_ok(object a, object b):
    """True if both are NPU tensors and we can call the NPU kernel directly."""
    cdef object b_dev = getattr(b, "device", None)

    if b_dev is None:
        return False
    if a.device.type != "npu" or b_dev.type != "npu":
        return False

    # Grad: skip fast-path if grad enabled and any tensor requires grad
    cdef bint grad_on = getattr(_grad_mode_state, "enabled", True)
    if grad_on and (getattr(a, "requires_grad", False) or getattr(b, "requires_grad", False)):
        return False
    if _is_functionalize_fn():
        return False
    if _current_pipeline_fn() is not None:
        return False
    return True


def _has_torch_function(args, kwargs):
    """Fast check: do any tensor args have __torch_function__ overrides?"""
    cdef object val

    _ensure_base()

    for val in args:
        if _check_value(val):
            return True

    if kwargs:
        for val in kwargs.values():
            if _check_value(val):
                return True

    return False


def _handle_torch_function(func, args, kwargs):
    """Dispatch to __torch_function__ if any arg is an overriding tensor subclass."""
    cdef object types
    cdef object val
    cdef object sorted_types
    cdef object cls
    cdef object result

    if not _has_torch_function(args, kwargs):
        return NotImplemented

    types = set()

    for val in args:
        _collect_types(val, types)

    if kwargs:
        for val in kwargs.values():
            _collect_types(val, types)

    sorted_types = sorted(types, key=lambda c: len(c.__mro__), reverse=True)
    for cls in sorted_types:
        result = cls.__torch_function__(func, types, args, kwargs or {})
        if result is not NotImplemented:
            return result

    return NotImplemented


def add(a=None, b=None, *, alpha=1, out=None):
    """Fast add: skip __torch_function__ when both args are base Tensor."""
    cdef object r

    _ensure_originals()

    if a is None or b is None:
        # Delegate to original for proper fallback behavior.
        return _py_add_fn(a, b, alpha=alpha, out=out) if a is not None else _py_add_fn()

    if _is_base_tensor(a) and (_is_base_tensor(b) or not hasattr(b, "__torch_function__")):
        if alpha != 1:
            b = _dispatch_fn("mul", None, b, alpha)
        elif _is_npu_tensor_pair(a, b):
            _ensure_npu_refs()
            if _npu_fast_ok(a, b):
                return _npu_add_fn(a, b)
        return _dispatch_fn("add", None, a, b)

    r = _handle_torch_function(_py_add_fn, (a, b), {"alpha": alpha, "out": out})
    if r is not NotImplemented:
        return r

    if alpha != 1:
        b = _dispatch_fn("mul", None, b, alpha)
    return _dispatch_fn("add", None, a, b)


def mul(a, b):
    """Fast mul: skip __torch_function__ when both args are base Tensor."""
    cdef object r

    _ensure_originals()

    if _is_base_tensor(a) and (_is_base_tensor(b) or not hasattr(b, "__torch_function__")):
        if _is_npu_tensor_pair(a, b):
            _ensure_npu_refs()
            if _npu_fast_ok(a, b):
                return _npu_mul_fn(a, b)
        return _dispatch_fn("mul", None, a, b)

    r = _handle_torch_function(_py_mul_fn, (a, b), {})
    if r is not NotImplemented:
        return r

    return _dispatch_fn("mul", None, a, b)


def matmul(a, b):
    """Fast matmul: skip __torch_function__ when both args are base Tensor."""
    cdef object r

    _ensure_originals()

    if _is_base_tensor(a) and _is_base_tensor(b):
        return _dispatch_fn("matmul", None, a, b)

    r = _handle_torch_function(_py_matmul_fn, (a, b), {})
    if r is not NotImplemented:
        return r

    return _dispatch_fn("matmul", None, a, b)


def relu(a):
    """Fast relu: skip __torch_function__ for base Tensor."""
    cdef object r

    _ensure_originals()

    if _is_base_tensor(a):
        return _dispatch_fn("relu", None, a)

    r = _handle_torch_function(_py_relu_fn, (a,), {})
    if r is not NotImplemented:
        return r

    return _dispatch_fn("relu", None, a)


def transpose(*args, **kwargs):
    _ensure_dispatch()
    return _dispatch_fn("transpose", None, *args, **kwargs)


def reshape(*args, **kwargs):
    _ensure_dispatch()
    return _dispatch_fn("reshape", None, *args, **kwargs)


def neg(a):
    """Fast neg: skip __torch_function__ for base Tensor."""
    cdef object r

    _ensure_originals()

    if _is_base_tensor(a):
        return _dispatch_fn("neg", a.device.type, a)

    r = _handle_torch_function(_py_neg_fn, (a,), {})
    if r is not NotImplemented:
        return r

    return _dispatch_fn("neg", a.device.type, a)
