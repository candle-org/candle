# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython fast-path for top-10 ops — skip __torch_function__ for base Tensor.

``type(a) is _BaseTensor`` is a single CPython pointer comparison (~1ns).
When both args are base Tensor, we skip the _has_torch_function scan entirely.
"""

# Cached reference to base Tensor class
cdef object _BaseTensor = None

cdef inline void _ensure_base():
    global _BaseTensor
    if _BaseTensor is None:
        from candle._tensor import Tensor
        _BaseTensor = Tensor


cdef inline bint _is_base_tensor(object t):
    """True if t is exactly the base Tensor class (not a subclass)."""
    _ensure_base()
    return type(t) is _BaseTensor


# Cache original Python implementations (loaded once)
cdef object _py_add_fn = None
cdef object _py_mul_fn = None
cdef object _py_matmul_fn = None
cdef object _py_sub_fn = None
cdef object _py_div_fn = None
cdef object _py_relu_fn = None
cdef object _py_neg_fn = None
cdef object _handle_tf = None
cdef object _dispatch_fn = None

cdef inline void _ensure_originals():
    global _py_add_fn, _py_mul_fn, _py_matmul_fn, _py_sub_fn
    global _py_div_fn, _py_relu_fn, _py_neg_fn, _handle_tf, _dispatch_fn
    if _dispatch_fn is None:
        from candle._dispatch.dispatcher import dispatch
        _dispatch_fn = dispatch
    if _handle_tf is None:
        from candle._functional import _handle_torch_function
        _handle_tf = _handle_torch_function
    if _py_add_fn is None:
        from candle._functional import _py_add, _py_mul, _py_matmul
        from candle._functional import _py_sub, _py_div, _py_relu, _py_neg
        _py_add_fn = _py_add
        _py_mul_fn = _py_mul
        _py_matmul_fn = _py_matmul
        _py_sub_fn = _py_sub
        _py_div_fn = _py_div
        _py_relu_fn = _py_relu
        _py_neg_fn = _py_neg


def add(a=None, b=None, *, alpha=1, out=None):
    """Fast add: skip __torch_function__ when both args are base Tensor."""
    _ensure_originals()
    if a is None or b is None:
        # Delegate to original for proper error message
        return _py_add_fn(a, b, alpha=alpha, out=out) if a is not None else _py_add_fn()
    if _is_base_tensor(a) and (_is_base_tensor(b) or not hasattr(b, "__torch_function__")):
        if alpha != 1:
            b = _dispatch_fn("mul", None, b, alpha)
        return _dispatch_fn("add", None, a, b)
    r = _handle_tf(_py_add_fn, (a, b), {"alpha": alpha, "out": out})
    if r is not NotImplemented:
        return r
    if alpha != 1:
        b = _dispatch_fn("mul", None, b, alpha)
    return _dispatch_fn("add", None, a, b)


def mul(a, b):
    """Fast mul: skip __torch_function__ when both args are base Tensor."""
    _ensure_originals()
    if _is_base_tensor(a) and (_is_base_tensor(b) or not hasattr(b, "__torch_function__")):
        return _dispatch_fn("mul", None, a, b)
    r = _handle_tf(_py_mul_fn, (a, b), {})
    if r is not NotImplemented:
        return r
    return _dispatch_fn("mul", None, a, b)


def matmul(a, b):
    """Fast matmul: skip __torch_function__ when both args are base Tensor."""
    _ensure_originals()
    if _is_base_tensor(a) and _is_base_tensor(b):
        return _dispatch_fn("matmul", None, a, b)
    r = _handle_tf(_py_matmul_fn, (a, b), {})
    if r is not NotImplemented:
        return r
    return _dispatch_fn("matmul", None, a, b)


def sub(a, b, *, alpha=1):
    """Fast sub: skip __torch_function__ when both args are base Tensor."""
    _ensure_originals()
    if _is_base_tensor(a) and (_is_base_tensor(b) or not hasattr(b, "__torch_function__")):
        if alpha != 1:
            b = _dispatch_fn("mul", None, b, alpha)
        return _dispatch_fn("sub", None, a, b)
    r = _handle_tf(_py_sub_fn, (a, b), {"alpha": alpha})
    if r is not NotImplemented:
        return r
    if alpha != 1:
        b = _dispatch_fn("mul", None, b, alpha)
    return _dispatch_fn("sub", None, a, b)


def div(a, b, *, rounding_mode=None):
    """Fast div: skip __torch_function__ when both args are base Tensor."""
    _ensure_originals()
    if _is_base_tensor(a) and (_is_base_tensor(b) or not hasattr(b, "__torch_function__")):
        if rounding_mode == "trunc":
            return _dispatch_fn("trunc_divide", None, a, b)
        if rounding_mode == "floor":
            return _dispatch_fn("floor_divide", None, a, b)
        return _dispatch_fn("true_divide", None, a, b)
    r = _handle_tf(_py_div_fn, (a, b), {"rounding_mode": rounding_mode})
    if r is not NotImplemented:
        return r
    if rounding_mode == "trunc":
        return _dispatch_fn("trunc_divide", None, a, b)
    if rounding_mode == "floor":
        return _dispatch_fn("floor_divide", None, a, b)
    return _dispatch_fn("true_divide", None, a, b)


def relu(a):
    """Fast relu: skip __torch_function__ for base Tensor."""
    _ensure_originals()
    if _is_base_tensor(a):
        return _dispatch_fn("relu", None, a)
    r = _handle_tf(_py_relu_fn, (a,), {})
    if r is not NotImplemented:
        return r
    return _dispatch_fn("relu", None, a)


def neg(a):
    """Fast neg: skip __torch_function__ for base Tensor."""
    _ensure_originals()
    if _is_base_tensor(a):
        return _dispatch_fn("neg", a.device.type, a)
    r = _handle_tf(_py_neg_fn, (a,), {})
    if r is not NotImplemented:
        return r
    return _dispatch_fn("neg", a.device.type, a)
