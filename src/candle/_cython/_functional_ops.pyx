# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython hot wrappers for candle._functional.

This module accelerates the most frequently used functional entry points while
preserving the pure-Python fallback functions in ``candle._functional`` for
stable ``__torch_function__`` identities and fallback behavior.
"""

import builtins as _builtins

# Cached reference to base Tensor class
cdef object _BaseTensor = None

# Cached Python callables
cdef object _dispatch_fn = None
cdef object _py_add_fn = None
cdef object _py_mul_fn = None
cdef object _py_matmul_fn = None
cdef object _py_sub_fn = None
cdef object _py_div_fn = None
cdef object _py_relu_fn = None
cdef object _py_neg_fn = None

# NPU fast-path: cached references for direct kernel calls
cdef object _npu_add_fn = None
cdef object _npu_mul_fn = None
cdef object _npu_sub_fn = None
cdef object _npu_div_fn = None
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
    global _py_add_fn, _py_mul_fn, _py_matmul_fn, _py_sub_fn, _py_div_fn
    global _py_relu_fn, _py_neg_fn

    _ensure_dispatch()

    if _py_add_fn is None:
        from candle._functional import _py_add, _py_mul, _py_matmul, _py_sub, _py_div, _py_relu, _py_neg
        _py_add_fn = _py_add
        _py_mul_fn = _py_mul
        _py_matmul_fn = _py_matmul
        _py_sub_fn = _py_sub
        _py_div_fn = _py_div
        _py_relu_fn = _py_relu
        _py_neg_fn = _py_neg


cdef inline void _ensure_npu_refs():
    """Load NPU op refs and guard state once."""
    global _npu_add_fn, _npu_mul_fn, _npu_sub_fn, _npu_div_fn
    global _grad_mode_state, _is_functionalize_fn, _current_pipeline_fn
    global _npu_refs_loaded

    if _npu_refs_loaded:
        return

    from candle._cython._npu_ops import fast_add as _nadd  # pylint: disable=import-error,no-name-in-module
    try:
        from candle._cython._npu_ops import fast_mul as _nmul  # pylint: disable=import-error,no-name-in-module
    except ImportError:
        from candle._backends.npu.ops import mul as _nmul
    try:
        from candle._cython._npu_ops import fast_sub as _nsub  # pylint: disable=import-error,no-name-in-module
    except ImportError:
        from candle._backends.npu.ops import sub as _nsub
    try:
        from candle._cython._npu_ops import fast_div as _ndiv  # pylint: disable=import-error,no-name-in-module
    except ImportError:
        from candle._backends.npu.ops import div as _ndiv
    from candle.autograd.grad_mode import _GRAD_MODE_STATE as _gms
    from candle._dispatch.functionalize import is_functionalize_enabled as _ife
    from candle._dispatch.pipeline import current_pipeline as _cp

    _npu_add_fn = _nadd
    _npu_mul_fn = _nmul
    _npu_sub_fn = _nsub
    _npu_div_fn = _ndiv
    _grad_mode_state = _gms
    _is_functionalize_fn = _ife
    _current_pipeline_fn = _cp
    _npu_refs_loaded = True


cdef inline bint _profiler_active():
    try:
        from candle.profiler.profiler import is_profiler_enabled
        return bool(is_profiler_enabled())
    except Exception:
        return False


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
            if _npu_fast_ok(a, b) and not _profiler_active():
                return _npu_add_fn(a, b)
        return _dispatch_fn("add", None, a, b)

    r = _handle_torch_function(_py_add_fn, (a, b), {"alpha": alpha, "out": out})
    if r is not NotImplemented:
        return r

    if alpha != 1:
        b = _dispatch_fn("mul", None, b, alpha)
    return _dispatch_fn("add", None, a, b)


def mul(a, b, *, out=None):
    """Fast mul: skip __torch_function__ when both args are base Tensor."""
    cdef object r
    cdef object result
    cdef object kwargs

    _ensure_originals()

    if _is_base_tensor(a) and (_is_base_tensor(b) or not hasattr(b, "__torch_function__")):
        if _is_npu_tensor_pair(a, b):
            _ensure_npu_refs()
            if _npu_fast_ok(a, b) and not _profiler_active():
                result = _npu_mul_fn(a, b)
                if out is not None:
                    out.copy_(result)
                    return out
                return result
        result = _dispatch_fn("mul", None, a, b)
        if out is not None:
            out.copy_(result)
            return out
        return result

    kwargs = {}
    if out is not None:
        kwargs["out"] = out
    r = _handle_torch_function(_py_mul_fn, (a, b), kwargs)
    if r is not NotImplemented:
        return r

    result = _dispatch_fn("mul", None, a, b)
    if out is not None:
        out.copy_(result)
        return out
    return result


def sub(a, b, *, alpha=1):
    """Fast sub: skip __torch_function__ when both args are base Tensor."""
    cdef object r

    _ensure_originals()

    if _is_base_tensor(a) and (_is_base_tensor(b) or not hasattr(b, "__torch_function__")):
        if alpha != 1:
            b = _dispatch_fn("mul", None, b, alpha)
        elif _is_npu_tensor_pair(a, b):
            _ensure_npu_refs()
            if _npu_fast_ok(a, b):
                return _npu_sub_fn(a, b)
        return _dispatch_fn("sub", None, a, b)

    r = _handle_torch_function(_py_sub_fn, (a, b), {"alpha": alpha})
    if r is not NotImplemented:
        return r

    if alpha != 1:
        b = _dispatch_fn("mul", None, b, alpha)
    return _dispatch_fn("sub", None, a, b)


def div(a, b, *, rounding_mode=None):
    """Fast div: skip __torch_function__ when both args are base Tensor."""
    cdef object r

    _ensure_originals()

    if _is_base_tensor(a) and (_is_base_tensor(b) or not hasattr(b, "__torch_function__")):
        if rounding_mode == "trunc":
            return _dispatch_fn("trunc_divide", None, a, b)
        if rounding_mode == "floor":
            return _dispatch_fn("floor_divide", None, a, b)
        if _is_npu_tensor_pair(a, b):
            _ensure_npu_refs()
            if _npu_fast_ok(a, b):
                return _npu_div_fn(a, b)
        return _dispatch_fn("true_divide", None, a, b)

    r = _handle_torch_function(_py_div_fn, (a, b), {"rounding_mode": rounding_mode})
    if r is not NotImplemented:
        return r

    if rounding_mode == "trunc":
        return _dispatch_fn("trunc_divide", None, a, b)
    if rounding_mode == "floor":
        return _dispatch_fn("floor_divide", None, a, b)
    return _dispatch_fn("true_divide", None, a, b)


def matmul(a, b, *, out=None):
    """Fast matmul: skip __torch_function__ when both args are base Tensor."""
    cdef object r
    cdef object result
    cdef object kwargs

    _ensure_originals()

    if _is_base_tensor(a) and _is_base_tensor(b):
        result = _dispatch_fn("matmul", None, a, b)
        if out is not None:
            out.copy_(result)
            return out
        return result

    kwargs = {}
    if out is not None:
        kwargs["out"] = out
    r = _handle_torch_function(_py_matmul_fn, (a, b), kwargs)
    if r is not NotImplemented:
        return r

    result = _dispatch_fn("matmul", None, a, b)
    if out is not None:
        out.copy_(result)
        return out
    return result


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
    cdef object a
    cdef object dim0
    cdef object dim1
    cdef object source_view_meta
    cdef object creation_mode
    cdef object creation_kind
    cdef object v

    _ensure_base()
    _ensure_dispatch()

    if not args:
        return _dispatch_fn("transpose", None, *args, **kwargs)

    a = args[0]
    if len(args) >= 3:
        dim0 = args[1]
        dim1 = args[2]
    else:
        dim0 = kwargs.get("dim0")
        dim1 = kwargs.get("dim1")

    if not _is_base_tensor(a):
        return _dispatch_fn("transpose", None, *args, **kwargs)
    if type(dim0) is not int or type(dim1) is not int:
        return _dispatch_fn("transpose", None, *args, **kwargs)

    v = a.cy_transpose(dim0, dim1)

    source_view_meta = getattr(a, "_view_meta", None) or {}
    from candle.autograd.grad_mode import current_creation_mode
    creation_mode = current_creation_mode() or source_view_meta.get("creation_mode")
    creation_kind = source_view_meta.get("creation_kind")
    if creation_mode is not None:
        if a._is_view():
            creation_kind = "view_of_view"
        else:
            creation_kind = "view"
    v._view_meta = {
        "op": "transpose",
        "shape": tuple(v.shape),
        "stride": tuple(v.stride),
        "offset": int(v.offset),
        "creation_mode": creation_mode,
        "creation_kind": creation_kind,
    }
    return v


def reshape(*args, **kwargs):
    cdef object a
    cdef object shape
    cdef object source_view_meta
    cdef object creation_mode
    cdef object creation_kind
    cdef object v
    cdef object shape_list
    cdef object infer_idx
    cdef object known_size
    cdef object size
    cdef object new_size
    cdef object idx
    cdef object dim

    _ensure_base()
    _ensure_dispatch()

    if not args:
        return _dispatch_fn("reshape", None, *args, **kwargs)

    a = args[0]
    if len(args) >= 2:
        shape = args[1]
    else:
        shape = kwargs.get("shape")

    if not _is_base_tensor(a):
        return _dispatch_fn("reshape", None, *args, **kwargs)

    if isinstance(shape, int):
        shape = (shape,)
    elif isinstance(shape, (tuple, list)):
        shape = tuple(shape)
    else:
        return _dispatch_fn("reshape", None, *args, **kwargs)

    if not a.is_contiguous():
        return _dispatch_fn("reshape", None, *args, **kwargs)

    size = 1
    for dim in a.shape:
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
        raise ValueError("reshape size mismatch")

    v = a.cy_view(shape)

    source_view_meta = getattr(a, "_view_meta", None) or {}
    from candle.autograd.grad_mode import current_creation_mode
    creation_mode = current_creation_mode() or source_view_meta.get("creation_mode")
    creation_kind = source_view_meta.get("creation_kind")
    if creation_mode is not None:
        if a._is_view():
            creation_kind = "view_of_view"
        else:
            creation_kind = "view"
    v._view_meta = {
        "op": "reshape",
        "shape": tuple(shape),
        "stride": tuple(v.stride),
        "offset": int(v.offset),
        "creation_mode": creation_mode,
        "creation_kind": creation_kind,
    }
    from candle.autograd import forward_ad
    level = forward_ad._current_level()
    if level >= 0:
        tangent = forward_ad.get_tangent(a, level)
        if tangent is not None:
            v._fw_set(level, tangent.reshape(shape))
    return v


def flatten(a, start_dim=0, end_dim=-1):
    cdef object ndim
    cdef object start
    cdef object end
    cdef object flattened
    cdef object d
    cdef object new_shape
    cdef object source_view_meta
    cdef object creation_mode
    cdef object creation_kind
    cdef object v

    _ensure_base()
    _ensure_dispatch()

    if not _is_base_tensor(a):
        return _dispatch_fn("flatten", a.device.type, a, start_dim, end_dim)

    ndim = len(a.shape)
    if ndim == 0:
        return reshape(a, (1,))

    start = start_dim if start_dim >= 0 else start_dim + ndim
    end = end_dim if end_dim >= 0 else end_dim + ndim
    if start < 0 or start >= ndim:
        raise IndexError("Dimension out of range")
    if end < 0 or end >= ndim:
        raise IndexError("Dimension out of range")
    if start > end:
        raise RuntimeError("flatten() has invalid args: start_dim cannot come after end_dim")

    flattened = 1
    for d in a.shape[start:end + 1]:
        flattened *= d
    new_shape = a.shape[:start] + (flattened,) + a.shape[end + 1:]

    v = reshape(a, new_shape)
    source_view_meta = getattr(a, "_view_meta", None) or {}
    creation_mode = (getattr(v, "_view_meta", None) or {}).get("creation_mode", source_view_meta.get("creation_mode"))
    creation_kind = (getattr(v, "_view_meta", None) or {}).get("creation_kind", source_view_meta.get("creation_kind"))
    v._view_meta = {
        "op": "flatten",
        "shape": tuple(v.shape),
        "stride": tuple(v.stride),
        "offset": int(v.offset),
        "creation_mode": creation_mode,
        "creation_kind": creation_kind,
    }
    return v


def unflatten(a, dim, sizes):
    cdef object ndim
    cdef object d
    cdef object shape
    cdef object size_idx
    cdef object infer_idx
    cdef object known_size
    cdef object old_size
    cdef object new_size
    cdef object source_view_meta
    cdef object creation_mode
    cdef object creation_kind
    cdef object v

    _ensure_base()
    _ensure_dispatch()

    if not _is_base_tensor(a):
        return _dispatch_fn("unflatten", a.device.type, a, dim, sizes)

    if not isinstance(dim, int):
        return _dispatch_fn("unflatten", a.device.type, a, dim, sizes)

    if isinstance(sizes, list):
        sizes = tuple(sizes)
    elif not isinstance(sizes, tuple):
        return _dispatch_fn("unflatten", a.device.type, a, dim, sizes)

    if not a.is_contiguous():
        return _dispatch_fn("unflatten", a.device.type, a, dim, sizes)

    ndim = len(a.shape)
    d = dim if dim >= 0 else dim + ndim
    if d < 0 or d >= ndim:
        raise IndexError("Dimension out of range")

    old_size = a.shape[d]
    infer_idx = None
    known_size = 1
    for size_idx, new_size in enumerate(sizes):
        if new_size == -1:
            if infer_idx is not None:
                raise RuntimeError("only one dimension can be inferred")
            infer_idx = size_idx
            continue
        known_size *= new_size

    sizes = list(sizes)
    if infer_idx is not None:
        if known_size == 0 or old_size % known_size != 0:
            raise RuntimeError(f"unflatten: Provided sizes {tuple(sizes)} don't multiply up to the size of dim {d} ({old_size}) in the input tensor")
        sizes[infer_idx] = old_size // known_size

    new_size = 1
    for size_idx in sizes:
        new_size *= size_idx
    if new_size != old_size:
        raise RuntimeError(f"unflatten: Provided sizes {tuple(sizes)} don't multiply up to the size of dim {d} ({old_size}) in the input tensor")

    shape = a.shape[:d] + tuple(sizes) + a.shape[d + 1:]
    v = a.cy_view(shape)

    source_view_meta = getattr(a, "_view_meta", None) or {}
    from candle.autograd.grad_mode import current_creation_mode
    creation_mode = current_creation_mode() or source_view_meta.get("creation_mode")
    creation_kind = source_view_meta.get("creation_kind")
    if creation_mode is not None:
        if a._is_view():
            creation_kind = "view_of_view"
        else:
            creation_kind = "view"
    v._view_meta = {
        "op": "unflatten",
        "shape": tuple(v.shape),
        "stride": tuple(v.stride),
        "offset": int(v.offset),
        "creation_mode": creation_mode,
        "creation_kind": creation_kind,
    }
    return v


def squeeze(a, dim=None):
    cdef object shape
    cdef object stride
    cdef object pairs
    cdef object ndim
    cdef object targets
    cdef object item
    cdef object d
    cdef object source_view_meta
    cdef object creation_mode
    cdef object creation_kind
    cdef object v

    _ensure_base()
    _ensure_dispatch()

    if not _is_base_tensor(a):
        return _dispatch_fn("squeeze", a.device.type, a, dim)
    if getattr(a, "requires_grad", False):
        return _dispatch_fn("squeeze", a.device.type, a, dim)

    shape = list(a.shape)
    stride = list(a.stride)
    if dim is not None:
        if isinstance(dim, (list, tuple)):
            if dim:
                ndim = len(shape)
                targets = set()
                for item in dim:
                    d = item if item >= 0 else item + ndim
                    targets.add(d)
                pairs = [
                    (s, st)
                    for idx, (s, st) in enumerate(zip(shape, stride))
                    if idx not in targets or s != 1
                ]
                shape = [p[0] for p in pairs]
                stride = [p[1] for p in pairs]
        else:
            d = dim if dim >= 0 else dim + len(shape)
            if 0 <= d < len(shape) and shape[d] == 1:
                del shape[d]
                del stride[d]
    else:
        pairs = [(s, st) for s, st in zip(shape, stride) if s != 1]
        shape = [p[0] for p in pairs]
        stride = [p[1] for p in pairs]

    v = a.cy_as_strided(tuple(shape), tuple(stride), a.offset)

    source_view_meta = getattr(a, "_view_meta", None) or {}
    from candle.autograd.grad_mode import current_creation_mode
    creation_mode = current_creation_mode() or source_view_meta.get("creation_mode")
    creation_kind = source_view_meta.get("creation_kind")
    if creation_mode is not None:
        if a._is_view():
            creation_kind = "view_of_view"
        else:
            creation_kind = "view"
    v._view_meta = {
        "op": "squeeze",
        "shape": tuple(v.shape),
        "stride": tuple(v.stride),
        "offset": int(v.offset),
        "creation_mode": creation_mode,
        "creation_kind": creation_kind,
    }
    return v


def unsqueeze(a, dim):
    cdef object ndim
    cdef object d
    cdef object shape
    cdef object stride
    cdef object new_stride
    cdef object source_view_meta
    cdef object creation_mode
    cdef object creation_kind
    cdef object v

    _ensure_base()
    _ensure_dispatch()

    if not _is_base_tensor(a):
        return _dispatch_fn("unsqueeze", a.device.type, a, dim)
    if getattr(a, "requires_grad", False):
        return _dispatch_fn("unsqueeze", a.device.type, a, dim)

    ndim = len(a.shape)
    d = dim if dim >= 0 else dim + ndim + 1
    shape = list(a.shape)
    stride = list(a.stride)
    new_stride = stride[d] * shape[d] if d < ndim else 1
    shape.insert(d, 1)
    stride.insert(d, new_stride)

    v = a.cy_as_strided(tuple(shape), tuple(stride), a.offset)

    source_view_meta = getattr(a, "_view_meta", None) or {}
    from candle.autograd.grad_mode import current_creation_mode
    creation_mode = current_creation_mode() or source_view_meta.get("creation_mode")
    creation_kind = source_view_meta.get("creation_kind")
    if creation_mode is not None:
        if a._is_view():
            creation_kind = "view_of_view"
        else:
            creation_kind = "view"
    v._view_meta = {
        "op": "unsqueeze",
        "shape": tuple(v.shape),
        "stride": tuple(v.stride),
        "offset": int(v.offset),
        "creation_mode": creation_mode,
        "creation_kind": creation_kind,
    }
    return v


def permute(a, dims):
    cdef object ndim
    cdef object normalized
    cdef object d
    cdef object shape
    cdef object stride
    cdef object source_view_meta
    cdef object creation_mode
    cdef object creation_kind
    cdef object v

    _ensure_base()
    _ensure_dispatch()

    if not _is_base_tensor(a):
        return _dispatch_fn("permute", a.device.type, a, dims)
    if getattr(a, "requires_grad", False):
        return _dispatch_fn("permute", a.device.type, a, dims)

    ndim = len(a.shape)
    if isinstance(dims, list):
        dims = tuple(dims)
    elif not isinstance(dims, tuple):
        return _dispatch_fn("permute", a.device.type, a, dims)

    normalized = []
    for d in dims:
        d = d if d >= 0 else d + ndim
        normalized.append(d)

    shape = [a.shape[d] for d in normalized]
    stride = [a.stride[d] for d in normalized]
    v = a.cy_as_strided(tuple(shape), tuple(stride), a.offset)

    source_view_meta = getattr(a, "_view_meta", None) or {}
    from candle.autograd.grad_mode import current_creation_mode
    creation_mode = current_creation_mode() or source_view_meta.get("creation_mode")
    creation_kind = source_view_meta.get("creation_kind")
    if creation_mode is not None:
        if a._is_view():
            creation_kind = "view_of_view"
        else:
            creation_kind = "view"
    v._view_meta = {
        "op": "permute",
        "shape": tuple(v.shape),
        "stride": tuple(v.stride),
        "offset": int(v.offset),
        "creation_mode": creation_mode,
        "creation_kind": creation_kind,
    }
    return v


def slice(input, dim, start=0, end=9223372036854775807, step=1):
    cdef object ndim
    cdef object d
    cdef object dim_size
    cdef object start_idx
    cdef object end_idx
    cdef object step_idx
    cdef object length
    cdef object new_shape
    cdef object new_stride
    cdef object new_offset
    cdef object source_view_meta
    cdef object creation_mode
    cdef object creation_kind
    cdef object v

    _ensure_base()
    _ensure_dispatch()

    if not _is_base_tensor(input):
        return _dispatch_fn("slice", input.device.type, input, dim, start, end, step)

    ndim = len(input.shape)
    d = dim if dim >= 0 else dim + ndim
    if d < 0 or d >= ndim:
        raise IndexError("Dimension out of range")

    dim_size = input.shape[d]
    step_idx = int(step)
    if step_idx == 0:
        raise ValueError("slice step cannot be zero")

    start_idx, end_idx, step_idx = _builtins.slice(start, end, step_idx).indices(dim_size)
    length = len(range(start_idx, end_idx, step_idx))
    new_shape = list(input.shape)
    new_shape[d] = length
    new_stride = list(input.stride)
    new_stride[d] = new_stride[d] * step_idx
    new_offset = input.offset + start_idx * input.stride[d]
    v = input.cy_as_strided(tuple(new_shape), tuple(new_stride), new_offset)

    source_view_meta = getattr(input, "_view_meta", None) or {}
    from candle.autograd.grad_mode import current_creation_mode
    creation_mode = current_creation_mode() or source_view_meta.get("creation_mode")
    creation_kind = source_view_meta.get("creation_kind")
    if creation_mode is not None:
        if input._is_view():
            creation_kind = "view_of_view"
        else:
            creation_kind = "view"
    v._view_meta = {
        "op": "slice",
        "shape": tuple(v.shape),
        "stride": tuple(v.stride),
        "offset": int(v.offset),
        "creation_mode": creation_mode,
        "creation_kind": creation_kind,
    }
    return v


def narrow(a, dim, start, length):
    cdef object ndim
    cdef object d
    cdef object new_shape
    cdef object new_offset
    cdef object source_view_meta
    cdef object creation_mode
    cdef object creation_kind
    cdef object v

    _ensure_base()
    _ensure_dispatch()

    if not _is_base_tensor(a):
        return _dispatch_fn("narrow", a.device.type, a, dim, start, length)
    if getattr(a, "requires_grad", False):
        return _dispatch_fn("narrow", a.device.type, a, dim, start, length)

    ndim = len(a.shape)
    d = dim if dim >= 0 else dim + ndim
    new_shape = list(a.shape)
    new_shape[d] = int(length)
    new_offset = a.offset + int(start) * a.stride[d]
    v = a.cy_as_strided(tuple(new_shape), tuple(a.stride), new_offset)

    source_view_meta = getattr(a, "_view_meta", None) or {}
    from candle.autograd.grad_mode import current_creation_mode
    creation_mode = current_creation_mode() or source_view_meta.get("creation_mode")
    creation_kind = source_view_meta.get("creation_kind")
    if creation_mode is not None:
        if a._is_view():
            creation_kind = "view_of_view"
        else:
            creation_kind = "view"
    v._view_meta = {
        "op": "narrow",
        "shape": tuple(v.shape),
        "stride": tuple(v.stride),
        "offset": int(v.offset),
        "creation_mode": creation_mode,
        "creation_kind": creation_kind,
    }
    return v


def select(a, dim, index):
    cdef object ndim
    cdef object d
    cdef object idx
    cdef object new_shape
    cdef object new_stride
    cdef object new_offset
    cdef object source_view_meta
    cdef object creation_mode
    cdef object creation_kind
    cdef object v

    _ensure_base()
    _ensure_dispatch()

    if not _is_base_tensor(a):
        return _dispatch_fn("select", a.device.type, a, dim, index)
    if getattr(a, "requires_grad", False):
        return _dispatch_fn("select", a.device.type, a, dim, index)

    ndim = len(a.shape)
    d = dim if dim >= 0 else dim + ndim
    idx = int(index)
    if idx < 0:
        idx += a.shape[d]
    new_shape = list(a.shape)
    del new_shape[d]
    new_stride = list(a.stride)
    new_offset = a.offset + idx * a.stride[d]
    del new_stride[d]
    v = a.cy_as_strided(tuple(new_shape), tuple(new_stride), new_offset)

    source_view_meta = getattr(a, "_view_meta", None) or {}
    from candle.autograd.grad_mode import current_creation_mode
    creation_mode = current_creation_mode() or source_view_meta.get("creation_mode")
    creation_kind = source_view_meta.get("creation_kind")
    if creation_mode is not None:
        if a._is_view():
            creation_kind = "view_of_view"
        else:
            creation_kind = "view"
    v._view_meta = {
        "op": "select",
        "shape": tuple(v.shape),
        "stride": tuple(v.stride),
        "offset": int(v.offset),
        "creation_mode": creation_mode,
        "creation_kind": creation_kind,
    }
    return v


def expand(a, sizes):
    cdef object sizes_tuple
    cdef object ndiff
    cdef object src_shape
    cdef object src_stride
    cdef object out_shape
    cdef object out_stride
    cdef object i
    cdef object sz
    cdef object source_view_meta
    cdef object creation_mode
    cdef object creation_kind
    cdef object v

    _ensure_base()
    _ensure_dispatch()

    if not _is_base_tensor(a):
        return _dispatch_fn("expand", a.device.type, a, sizes)
    if getattr(a, "requires_grad", False):
        return _dispatch_fn("expand", a.device.type, a, sizes)

    if isinstance(sizes, list):
        sizes_tuple = tuple(sizes)
    elif isinstance(sizes, tuple):
        sizes_tuple = sizes
    else:
        return _dispatch_fn("expand", a.device.type, a, sizes)

    ndiff = len(sizes_tuple) - len(a.shape)
    if ndiff < 0:
        raise RuntimeError("expand: number of sizes must be >= tensor dim")

    src_shape = (1,) * ndiff + a.shape
    src_stride = (0,) * ndiff + a.stride
    out_shape = []
    out_stride = []
    for i, sz in enumerate(sizes_tuple):
        if sz == -1:
            out_shape.append(src_shape[i])
            out_stride.append(src_stride[i])
        elif src_shape[i] == 1:
            out_shape.append(sz)
            out_stride.append(0)
        elif src_shape[i] == sz:
            out_shape.append(sz)
            out_stride.append(src_stride[i])
        else:
            raise RuntimeError(
                f"expand: size {sz} not compatible with dim size {src_shape[i]}"
            )

    v = a.cy_as_strided(tuple(out_shape), tuple(out_stride), a.offset)

    source_view_meta = getattr(a, "_view_meta", None) or {}
    from candle.autograd.grad_mode import current_creation_mode
    creation_mode = current_creation_mode() or source_view_meta.get("creation_mode")
    creation_kind = source_view_meta.get("creation_kind")
    if creation_mode is not None:
        if a._is_view():
            creation_kind = "view_of_view"
        else:
            creation_kind = "view"
    v._view_meta = {
        "op": "expand",
        "shape": tuple(v.shape),
        "stride": tuple(v.stride),
        "offset": int(v.offset),
        "creation_mode": creation_mode,
        "creation_kind": creation_kind,
    }
    return v


def movedim(a, source, destination):
    cdef object ndim
    cdef object source_tuple
    cdef object destination_tuple
    cdef object order
    cdef object dst_order
    cdef object dst_idx
    cdef object shape
    cdef object stride
    cdef object source_view_meta
    cdef object creation_mode
    cdef object creation_kind
    cdef object v

    _ensure_base()
    _ensure_dispatch()

    if not _is_base_tensor(a):
        return _dispatch_fn("movedim", a.device.type, a, source, destination)

    ndim = len(a.shape)
    if isinstance(source, int):
        source_tuple = (source,)
    elif isinstance(source, list):
        source_tuple = tuple(source)
    else:
        source_tuple = source

    if isinstance(destination, int):
        destination_tuple = (destination,)
    elif isinstance(destination, list):
        destination_tuple = tuple(destination)
    else:
        destination_tuple = destination

    if not isinstance(source_tuple, tuple) or not isinstance(destination_tuple, tuple):
        return _dispatch_fn("movedim", a.device.type, a, source, destination)

    source_tuple = tuple(s % ndim for s in source_tuple)
    destination_tuple = tuple(d % ndim for d in destination_tuple)

    order = [n for n in range(ndim) if n not in source_tuple]
    dst_order = sorted(range(len(destination_tuple)), key=lambda i: destination_tuple[i])
    for dst_idx in dst_order:
        order.insert(destination_tuple[dst_idx], source_tuple[dst_idx])

    shape = [a.shape[d] for d in order]
    stride = [a.stride[d] for d in order]
    v = a.cy_as_strided(tuple(shape), tuple(stride), a.offset)

    source_view_meta = getattr(a, "_view_meta", None) or {}
    from candle.autograd.grad_mode import current_creation_mode
    creation_mode = current_creation_mode() or source_view_meta.get("creation_mode")
    creation_kind = source_view_meta.get("creation_kind")
    if creation_mode is not None:
        if a._is_view():
            creation_kind = "view_of_view"
        else:
            creation_kind = "view"
    v._view_meta = {
        "op": "movedim",
        "shape": tuple(v.shape),
        "stride": tuple(v.stride),
        "offset": int(v.offset),
        "creation_mode": creation_mode,
        "creation_kind": creation_kind,
    }
    return v


def diagonal(a, offset=0, dim1=0, dim2=1):
    cdef object ndim
    cdef object d1
    cdef object d2
    cdef object shape
    cdef object stride
    cdef object size1
    cdef object size2
    cdef object diag_len
    cdef object base_offset
    cdef object out_shape
    cdef object out_stride
    cdef object i
    cdef object source_view_meta
    cdef object creation_mode
    cdef object creation_kind
    cdef object v

    _ensure_base()
    _ensure_dispatch()

    if not _is_base_tensor(a):
        return _dispatch_fn("diagonal", a.device.type, a, offset, dim1, dim2)

    ndim = len(a.shape)
    d1 = dim1 if dim1 >= 0 else dim1 + ndim
    d2 = dim2 if dim2 >= 0 else dim2 + ndim

    shape = list(a.shape)
    stride = list(a.stride)
    size1 = shape[d1]
    size2 = shape[d2]

    if offset >= 0:
        diag_len = max(0, min(size1, size2 - offset))
        base_offset = a.offset + offset * stride[d2]
    else:
        diag_len = max(0, min(size1 + offset, size2))
        base_offset = a.offset + (-offset) * stride[d1]

    out_shape = [shape[i] for i in range(ndim) if i not in (d1, d2)]
    out_stride = [stride[i] for i in range(ndim) if i not in (d1, d2)]
    out_shape.append(diag_len)
    out_stride.append(stride[d1] + stride[d2])

    v = a.cy_as_strided(tuple(out_shape), tuple(out_stride), base_offset)

    source_view_meta = getattr(a, "_view_meta", None) or {}
    from candle.autograd.grad_mode import current_creation_mode
    creation_mode = current_creation_mode() or source_view_meta.get("creation_mode")
    creation_kind = source_view_meta.get("creation_kind")
    if creation_mode is not None:
        if a._is_view():
            creation_kind = "view_of_view"
        else:
            creation_kind = "view"
    v._view_meta = {
        "op": "diagonal",
        "shape": tuple(v.shape),
        "stride": tuple(v.stride),
        "offset": int(v.offset),
        "creation_mode": creation_mode,
        "creation_kind": creation_kind,
    }
    return v


def view_as_real(a):
    cdef object out_dtype
    cdef object shape
    cdef object stride
    cdef object source_view_meta
    cdef object creation_mode
    cdef object creation_kind
    cdef object v

    _ensure_base()
    _ensure_dispatch()

    if not _is_base_tensor(a):
        return _dispatch_fn("view_as_real", a.device.type, a)

    if not a.is_complex():
        raise RuntimeError("view_as_real expects a complex tensor")
    if a.dtype.itemsize % 2 != 0:
        raise RuntimeError("view_as_real expects complex dtype with even itemsize")
    if a.dtype.name == "complex64":
        from candle import float32 as out_dtype
    elif a.dtype.name == "complex128":
        from candle import float64 as out_dtype
    elif a.dtype.name == "complex32":
        from candle import float16 as out_dtype
    else:
        raise RuntimeError("view_as_real expects a supported complex dtype")

    shape = tuple(a.shape) + (2,)
    stride = tuple(s * 2 for s in a.stride) + (1,)
    v = a.cy_as_strided(shape, stride, a.offset * 2)
    v._storage = a.storage()._reinterpret(out_dtype)

    source_view_meta = getattr(a, "_view_meta", None) or {}
    from candle.autograd.grad_mode import current_creation_mode
    creation_mode = current_creation_mode() or source_view_meta.get("creation_mode")
    creation_kind = source_view_meta.get("creation_kind")
    if creation_mode is not None:
        if a._is_view():
            creation_kind = "view_of_view"
        else:
            creation_kind = "view"
    v._view_meta = {
        "op": "view_as_real",
        "shape": tuple(v.shape),
        "stride": tuple(v.stride),
        "offset": int(v.offset),
        "creation_mode": creation_mode,
        "creation_kind": creation_kind,
    }
    return v


def view_as_complex(a):
    cdef object out_dtype
    cdef object ndim
    cdef object last_dim
    cdef object shape
    cdef object stride
    cdef object source_view_meta
    cdef object creation_mode
    cdef object creation_kind
    cdef object v

    _ensure_base()
    _ensure_dispatch()

    if not _is_base_tensor(a):
        return _dispatch_fn("view_as_complex", a.device.type, a)

    ndim = len(a.shape)
    if ndim == 0:
        raise RuntimeError("view_as_complex expects last dimension of size 2")
    if a.is_complex():
        raise RuntimeError("view_as_complex expects a non-complex tensor")
    last_dim = a.shape[ndim - 1]
    if last_dim != 2:
        raise RuntimeError("view_as_complex expects last dimension of size 2")
    if a.dtype.name == "float16":
        from candle import complex32 as out_dtype
    elif a.dtype.name == "float32":
        from candle import complex64 as out_dtype
    elif a.dtype.name == "float64":
        from candle import complex128 as out_dtype
    else:
        raise RuntimeError("view_as_complex is only supported for half, float and double tensors")

    shape = tuple(a.shape[i] for i in range(ndim - 1))
    stride = tuple(a.stride[i] // 2 for i in range(ndim - 1))
    v = a.cy_as_strided(shape, stride, a.offset // 2)
    v._storage = a.storage()._reinterpret(out_dtype)

    source_view_meta = getattr(a, "_view_meta", None) or {}
    from candle.autograd.grad_mode import current_creation_mode
    creation_mode = current_creation_mode() or source_view_meta.get("creation_mode")
    creation_kind = source_view_meta.get("creation_kind")
    if creation_mode is not None:
        if a._is_view():
            creation_kind = "view_of_view"
        else:
            creation_kind = "view"
    v._view_meta = {
        "op": "view_as_complex",
        "shape": tuple(v.shape),
        "stride": tuple(v.stride),
        "offset": int(v.offset),
        "creation_mode": creation_mode,
        "creation_kind": creation_kind,
    }
    return v


def unfold(a, dimension, size, step):
    cdef object ndim
    cdef object d
    cdef object dim_size
    cdef object n_windows
    cdef object shape
    cdef object stride
    cdef object source_view_meta
    cdef object creation_mode
    cdef object creation_kind
    cdef object v

    _ensure_base()
    _ensure_dispatch()

    if not _is_base_tensor(a):
        return _dispatch_fn("unfold", a.device.type, a, dimension, size, step)

    ndim = len(a.shape)
    d = dimension if dimension >= 0 else dimension + ndim
    dim_size = a.shape[d]
    n_windows = max(0, (dim_size - size) // step + 1)

    shape = list(a.shape)
    stride = list(a.stride)
    shape[d] = n_windows
    shape.append(size)
    stride[d] = stride[d] * step
    stride.append(a.stride[d])

    v = a.cy_as_strided(tuple(shape), tuple(stride), a.offset)

    source_view_meta = getattr(a, "_view_meta", None) or {}
    from candle.autograd.grad_mode import current_creation_mode
    creation_mode = current_creation_mode() or source_view_meta.get("creation_mode")
    creation_kind = source_view_meta.get("creation_kind")
    if creation_mode is not None:
        if a._is_view():
            creation_kind = "view_of_view"
        else:
            creation_kind = "view"
    v._view_meta = {
        "op": "unfold",
        "shape": tuple(v.shape),
        "stride": tuple(v.stride),
        "offset": int(v.offset),
        "creation_mode": creation_mode,
        "creation_kind": creation_kind,
    }
    return v


def unbind(a, dim=0):
    cdef object ndim
    cdef object d
    cdef object dim_size
    cdef object i
    cdef object idx
    cdef object new_shape
    cdef object new_stride
    cdef object new_offset
    cdef object source_view_meta
    cdef object creation_mode
    cdef object creation_kind
    cdef object outputs
    cdef object v

    _ensure_base()
    _ensure_dispatch()

    if not _is_base_tensor(a):
        return _dispatch_fn("unbind", a.device.type, a, dim)
    if getattr(a, "requires_grad", False):
        return _dispatch_fn("unbind", a.device.type, a, dim)

    ndim = len(a.shape)
    d = dim if dim >= 0 else dim + ndim
    dim_size = a.shape[d]
    outputs = []

    source_view_meta = getattr(a, "_view_meta", None) or {}
    from candle.autograd.grad_mode import current_creation_mode
    creation_mode = current_creation_mode() or source_view_meta.get("creation_mode")
    creation_kind = source_view_meta.get("creation_kind")
    if creation_mode is not None:
        if a._is_view():
            creation_kind = "view_of_view"
        else:
            creation_kind = "view"
    else:
        creation_kind = "multi_view"

    new_shape = list(a.shape)
    del new_shape[d]
    new_stride = list(a.stride)
    del new_stride[d]

    for i in range(dim_size):
        idx = int(i)
        new_offset = a.offset + idx * a.stride[d]
        v = a.cy_as_strided(tuple(new_shape), tuple(new_stride), new_offset)
        v._view_meta = {
            "op": "select",
            "shape": tuple(v.shape),
            "stride": tuple(v.stride),
            "offset": int(v.offset),
            "creation_mode": creation_mode,
            "creation_kind": creation_kind,
        }
        outputs.append(v)
    return tuple(outputs)


def split(a, split_size_or_sections, dim=0):
    cdef object ndim
    cdef object d
    cdef object dim_size
    cdef object outputs
    cdef object step
    cdef object start
    cdef object end
    cdef object size
    cdef object new_shape
    cdef object new_offset
    cdef object source_view_meta
    cdef object creation_mode
    cdef object creation_kind
    cdef object v

    _ensure_base()
    _ensure_dispatch()

    if not _is_base_tensor(a):
        return _dispatch_fn("split", a.device.type, a, split_size_or_sections, dim)
    if getattr(a, "requires_grad", False):
        return _dispatch_fn("split", a.device.type, a, split_size_or_sections, dim)

    ndim = len(a.shape)
    d = dim if dim >= 0 else dim + ndim
    dim_size = a.shape[d]
    outputs = []

    source_view_meta = getattr(a, "_view_meta", None) or {}
    from candle.autograd.grad_mode import current_creation_mode
    creation_mode = current_creation_mode() or source_view_meta.get("creation_mode")
    creation_kind = source_view_meta.get("creation_kind")
    if creation_mode is not None:
        if a._is_view():
            creation_kind = "view_of_view"
        else:
            creation_kind = "view"
    else:
        creation_kind = "multi_view"

    if isinstance(split_size_or_sections, int):
        if split_size_or_sections <= 0:
            raise ValueError("split_size must be > 0")
        step = split_size_or_sections
        for start in range(0, dim_size, step):
            end = start + step
            if end > dim_size:
                end = dim_size
            new_shape = list(a.shape)
            new_shape[d] = end - start
            new_offset = a.offset + int(start) * a.stride[d]
            v = a.cy_as_strided(tuple(new_shape), tuple(a.stride), new_offset)
            v._view_meta = {
                "op": "narrow",
                "shape": tuple(v.shape),
                "stride": tuple(v.stride),
                "offset": int(v.offset),
                "creation_mode": creation_mode,
                "creation_kind": creation_kind,
            }
            outputs.append(v)
    else:
        if sum(split_size_or_sections) != dim_size:
            raise ValueError("split sections must sum to dim size")
        start = 0
        for size in split_size_or_sections:
            new_shape = list(a.shape)
            new_shape[d] = size
            new_offset = a.offset + int(start) * a.stride[d]
            v = a.cy_as_strided(tuple(new_shape), tuple(a.stride), new_offset)
            v._view_meta = {
                "op": "narrow",
                "shape": tuple(v.shape),
                "stride": tuple(v.stride),
                "offset": int(v.offset),
                "creation_mode": creation_mode,
                "creation_kind": creation_kind,
            }
            outputs.append(v)
            start += size
    return tuple(outputs)


def chunk(a, chunks, dim=0):
    cdef object ndim
    cdef object d
    cdef object dim_size
    cdef object actual_chunks
    cdef object chunk_size
    cdef object size

    _ensure_base()
    _ensure_dispatch()

    if not _is_base_tensor(a):
        return _dispatch_fn("chunk", a.device.type, a, chunks, dim)

    ndim = len(a.shape)
    d = dim if dim >= 0 else dim + ndim
    dim_size = a.shape[d]
    if chunks <= 0:
        raise ValueError("chunks must be > 0")
    actual_chunks = chunks if dim_size == 0 else min(chunks, dim_size)
    if actual_chunks == 0:
        return tuple()
    chunk_size = (dim_size + actual_chunks - 1) // actual_chunks
    return split(a, chunk_size, d)


def vsplit(a, split_size_or_sections):
    cdef object sizes
    cdef object sections
    cdef object dim_size
    cdef object size
    cdef object extra

    _ensure_base()
    _ensure_dispatch()

    if not _is_base_tensor(a):
        return _dispatch_fn("vsplit", a.device.type, a, split_size_or_sections)

    if isinstance(split_size_or_sections, int):
        sections = split_size_or_sections
        if sections <= 0:
            raise ValueError("sections must be > 0")
        dim_size = a.shape[0]
        size, extra = divmod(dim_size, sections)
        sizes = [size + 1] * extra + [size] * (sections - extra)
        return split(a, tuple(sizes), 0)
    return split(a, split_size_or_sections, 0)


def hsplit(a, split_size_or_sections):
    cdef object dim
    cdef object sizes
    cdef object sections
    cdef object dim_size
    cdef object size
    cdef object extra

    _ensure_base()
    _ensure_dispatch()

    if not _is_base_tensor(a):
        return _dispatch_fn("hsplit", a.device.type, a, split_size_or_sections)

    dim = 0 if a.dim() == 1 else 1
    if isinstance(split_size_or_sections, int):
        sections = split_size_or_sections
        if sections <= 0:
            raise ValueError("sections must be > 0")
        dim_size = a.shape[dim]
        size, extra = divmod(dim_size, sections)
        sizes = [size + 1] * extra + [size] * (sections - extra)
        return split(a, tuple(sizes), dim)
    return split(a, split_size_or_sections, dim)


def dsplit(a, split_size_or_sections):
    cdef object sizes
    cdef object sections
    cdef object dim_size
    cdef object size
    cdef object extra

    _ensure_base()
    _ensure_dispatch()

    if not _is_base_tensor(a):
        return _dispatch_fn("dsplit", a.device.type, a, split_size_or_sections)

    if a.dim() < 3:
        raise ValueError("dsplit expects input with at least 3 dimensions")

    if isinstance(split_size_or_sections, int):
        sections = split_size_or_sections
        if sections <= 0:
            raise ValueError("sections must be > 0")
        dim_size = a.shape[2]
        size, extra = divmod(dim_size, sections)
        sizes = [size + 1] * extra + [size] * (sections - extra)
        return split(a, tuple(sizes), 2)
    return split(a, split_size_or_sections, 2)


def view(*args, **kwargs):
    cdef object a
    cdef object shape
    cdef object source_view_meta
    cdef object creation_mode
    cdef object creation_kind
    cdef object v
    cdef object shape_list
    cdef object infer_idx
    cdef object known_size
    cdef object size
    cdef object new_size
    cdef object idx
    cdef object dim

    _ensure_base()
    _ensure_dispatch()

    if not args:
        return _dispatch_fn("view", None, *args, **kwargs)

    a = args[0]
    if len(args) >= 2:
        shape = args[1]
    else:
        shape = kwargs.get("shape")

    if not _is_base_tensor(a):
        return _dispatch_fn("view", None, *args, **kwargs)

    if isinstance(shape, int):
        shape = (shape,)
    elif isinstance(shape, (tuple, list)):
        shape = tuple(shape)
    else:
        return _dispatch_fn("view", None, *args, **kwargs)

    if not a.is_contiguous():
        raise RuntimeError(
            "view size is not compatible with input tensor's size and stride "
            "(at least one dimension spans across two contiguous subspaces). "
            "Use .reshape(...) instead."
        )

    size = 1
    for dim in a.shape:
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

    v = a.cy_view(shape)

    source_view_meta = getattr(a, "_view_meta", None) or {}
    from candle.autograd.grad_mode import current_creation_mode
    creation_mode = current_creation_mode() or source_view_meta.get("creation_mode")
    creation_kind = source_view_meta.get("creation_kind")
    if creation_mode is not None:
        if a._is_view():
            creation_kind = "view_of_view"
        else:
            creation_kind = "view"
    v._view_meta = {
        "op": "view",
        "shape": tuple(shape),
        "stride": tuple(v.stride),
        "offset": int(v.offset),
        "creation_mode": creation_mode,
        "creation_kind": creation_kind,
    }
    return v



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
