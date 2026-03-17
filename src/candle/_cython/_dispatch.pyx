# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython hot-path for the dispatch system.

Accelerates tensor extraction, keyset construction, TLS mask application,
kernel lookup, and kwargs preparation. The full dispatch logic (functionalize,
autocast, pipeline, profiler) remains in Python dispatcher.py.
"""

from libc.stdint cimport int64_t, uint32_t

# ---------------------------------------------------------------------------
# FastDispatchKeySet — bitmask-based keyset with C operations
# ---------------------------------------------------------------------------

cdef class FastDispatchKeySet:
    """Cython replacement for DispatchKeySet with C bitmask ops."""
    cdef public unsigned int mask

    def __init__(self, mask=0):
        if isinstance(mask, (set, list, tuple)):
            self.mask = 0
            for key in mask:
                self.mask |= <unsigned int>int(key)
        else:
            self.mask = <unsigned int>int(mask)

    def __int__(self):
        return <int>self.mask

    def __contains__(self, key):
        return (self.mask & <unsigned int>int(key)) != 0

    cpdef bint has(self, key):
        return (self.mask & <unsigned int>int(key)) != 0

    cpdef FastDispatchKeySet add(self, key):
        self.mask |= <unsigned int>int(key)
        return self

    cpdef FastDispatchKeySet remove(self, key):
        self.mask &= ~(<unsigned int>int(key))
        return self

    def without(self, keys):
        cdef unsigned int new_mask = self.mask
        if isinstance(keys, (set, list, tuple)):
            for key in keys:
                new_mask &= ~(<unsigned int>int(key))
        else:
            new_mask &= ~(<unsigned int>int(keys))
        return FastDispatchKeySet(new_mask)

    def iter_keys(self):
        from candle._dispatch.keys import DISPATCH_KEY_PRIORITY
        cdef unsigned int m = self.mask
        for key in DISPATCH_KEY_PRIORITY:
            if m & <unsigned int>int(key):
                yield key

    @classmethod
    def from_mask(cls, mask):
        return cls(int(mask))

    @classmethod
    def from_tensors(cls, tensors, *, grad_enabled=False, pipeline_enabled=False,
                     functionalize_enabled=False, device=None, autocast_enabled=False):
        return _cy_from_tensors(tensors, grad_enabled, pipeline_enabled,
                                functionalize_enabled, device, autocast_enabled)


# Cached reference to base Tensor class
cdef object _BaseTensor = None

cdef FastDispatchKeySet _cy_from_tensors(list tensors, bint grad_enabled,
                                          bint pipeline_enabled,
                                          bint functionalize_enabled,
                                          object device, bint autocast_enabled):
    """Build keyset from tensors — C-speed bitmask construction."""
    # Import DispatchKey constants
    from candle._dispatch.keys import DispatchKey

    cdef bint has_meta = False, has_npu = False, has_cuda = False
    cdef bint has_mps = False, has_cpu = False
    cdef bint requires_grad = False, saw_device = False
    cdef bint has_dispatch_subclass = False
    cdef unsigned int mask = 0

    global _BaseTensor
    if _BaseTensor is None:
        from candle._tensor import Tensor
        _BaseTensor = Tensor

    for tensor in tensors:
        dev = getattr(tensor, "device", None)
        if dev is None:
            continue
        saw_device = True
        dev_type = getattr(dev, "type", dev)
        if dev_type == "meta":
            has_meta = True
        elif dev_type == "npu":
            has_npu = True
        elif dev_type == "cuda":
            has_cuda = True
        elif dev_type == "mps":
            has_mps = True
        else:
            has_cpu = True
        if getattr(tensor, "requires_grad", False):
            requires_grad = True
        if not has_dispatch_subclass:
            tensor_cls = type(tensor)
            if tensor_cls is not _BaseTensor:
                td = getattr(tensor_cls, "__torch_dispatch__", None)
                if td is not None:
                    base_td = _BaseTensor.__dict__.get("__torch_dispatch__")
                    actual_func = getattr(td, "__func__", td)
                    base_func = getattr(base_td, "__func__", base_td)
                    if actual_func is not base_func:
                        has_dispatch_subclass = True

    if (not saw_device) and device is not None:
        dev_type = getattr(device, "type", device)
        if dev_type == "meta":
            has_meta = True
        elif dev_type == "npu":
            has_npu = True
        elif dev_type == "cuda":
            has_cuda = True
        elif dev_type == "mps":
            has_mps = True
        else:
            has_cpu = True

    if has_meta:
        mask |= <unsigned int>int(DispatchKey.Meta)
    elif has_npu:
        mask |= <unsigned int>int(DispatchKey.NPU)
    elif has_cuda:
        mask |= <unsigned int>int(DispatchKey.CUDA)
    elif has_mps:
        mask |= <unsigned int>int(DispatchKey.PrivateUse2)
    else:
        mask |= <unsigned int>int(DispatchKey.CPU)

    if grad_enabled and requires_grad:
        mask |= <unsigned int>int(DispatchKey.ADInplaceOrView)
        mask |= <unsigned int>int(DispatchKey.Autograd)
        if has_meta:
            mask |= <unsigned int>int(DispatchKey.AutogradMeta)
        elif has_npu:
            mask |= <unsigned int>int(DispatchKey.AutogradNPU)
        elif has_cuda:
            mask |= <unsigned int>int(DispatchKey.AutogradCUDA)
        elif has_mps:
            mask |= <unsigned int>int(DispatchKey.PrivateUse3)
        else:
            mask |= <unsigned int>int(DispatchKey.AutogradCPU)

    if functionalize_enabled:
        mask |= <unsigned int>int(DispatchKey.Functionalize)
    if autocast_enabled:
        mask |= <unsigned int>int(DispatchKey.Autocast)
    if pipeline_enabled and not has_meta and not has_cuda:
        mask |= <unsigned int>int(DispatchKey.Pipeline)
    if has_dispatch_subclass:
        mask |= <unsigned int>int(DispatchKey.Python)

    return FastDispatchKeySet(mask)


# ---------------------------------------------------------------------------
# Fast tensor extraction
# ---------------------------------------------------------------------------

import numpy as np

def cy_extract_tensors(tuple args, dict kwargs):
    """Extract tensors from args/kwargs — typed C loop."""
    cdef list tensors = []
    cdef int n = len(args)

    # Fast path: binary op (2 tensor args, no kwargs)
    if n == 2 and not kwargs:
        a = args[0]
        b = args[1]
        if hasattr(a, "device") and hasattr(b, "device"):
            tensors.append(a)
            tensors.append(b)
            return tensors

    cdef int i
    for i in range(n):
        _cy_visit(args[i], tensors)
    for v in kwargs.values():
        _cy_visit(v, tensors)
    return tensors


cdef void _cy_visit(object value, list tensors):
    if hasattr(value, "device") and not isinstance(value, np.ndarray):
        tensors.append(value)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _cy_visit(item, tensors)


# ---------------------------------------------------------------------------
# Fast TLS mask application
# ---------------------------------------------------------------------------

def cy_apply_tls_masks(keyset):
    """Apply thread-local include/exclude masks — inline bitmask ops."""
    from candle._dispatch.keys import _tls_state
    cdef unsigned int base_mask
    if isinstance(keyset, FastDispatchKeySet):
        base_mask = (<FastDispatchKeySet>keyset).mask
    else:
        base_mask = <unsigned int>int(getattr(keyset, 'mask', keyset))

    state = _tls_state()
    cdef unsigned int include_mask = 0
    cdef unsigned int exclude_mask = 0
    for m in state["include"]:
        include_mask |= <unsigned int>m
    for m in state["exclude"]:
        exclude_mask |= <unsigned int>m

    return FastDispatchKeySet((base_mask | include_mask) & ~exclude_mask)


# ---------------------------------------------------------------------------
# Fast kernel lookup
# ---------------------------------------------------------------------------

def cy_kernel_for_entry(entry, keyset):
    """Find kernel for dispatch entry — direct iteration."""
    from candle._dispatch.keys import DISPATCH_KEY_PRIORITY
    from candle._dispatch.registry import registry

    cdef unsigned int m
    if isinstance(keyset, FastDispatchKeySet):
        m = (<FastDispatchKeySet>keyset).mask
    else:
        m = <unsigned int>int(getattr(keyset, 'mask', keyset))

    fallthrough = entry.fallthrough
    kernels = entry.kernels
    global_fallthrough = getattr(registry, "_global_fallthrough", set())

    for key in DISPATCH_KEY_PRIORITY:
        if not (m & <unsigned int>int(key)):
            continue
        if key in fallthrough:
            continue
        kernel = kernels.get(key)
        if kernel is not None:
            return kernel, key
        if key in global_fallthrough:
            continue
    return None, None


# ---------------------------------------------------------------------------
# Fast _prepare_kwargs with cached accepts_device
# ---------------------------------------------------------------------------

import inspect

_accepts_device_cache = {}

cdef bint _cy_accepts_device(object func):
    cached = _accepts_device_cache.get(func)
    if cached is not None:
        return cached
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        _accepts_device_cache[func] = False
        return False
    result = "device" in sig.parameters
    _accepts_device_cache[func] = result
    return result


def cy_prepare_kwargs(func, dict kwargs, device):
    """Prepare kwargs with device injection — cached signature check."""
    if not kwargs:
        if _cy_accepts_device(func):
            return {"device": device}
        return {}
    if "device" in kwargs:
        if _cy_accepts_device(func):
            return kwargs
        return {k: v for k, v in kwargs.items() if k != "device"}
    if _cy_accepts_device(func):
        merged = dict(kwargs)
        merged["device"] = device
        return merged
    return kwargs


# ---------------------------------------------------------------------------
# Public dispatch entry points (delegate to Python dispatcher with fast primitives)
# ---------------------------------------------------------------------------

def cy_dispatch(str name, dispatch_device, *args, **kwargs):
    """Cython-accelerated dispatch. Same semantics as Python dispatch()."""
    from candle._dispatch.dispatcher import dispatch_with_keyset
    from candle._dispatch.pipeline import current_pipeline
    from candle.autograd.grad_mode import is_grad_enabled
    from candle.amp.state import is_autocast_enabled
    from candle._dispatch.functionalize import is_functionalize_enabled

    tensors = cy_extract_tensors(args, kwargs)
    autocast_device_type = getattr(dispatch_device, "type", dispatch_device)
    if autocast_device_type is None and tensors:
        autocast_device_type = getattr(tensors[0].device, "type", None)
    pipe = current_pipeline()
    keyset = FastDispatchKeySet.from_tensors(
        tensors,
        grad_enabled=is_grad_enabled(),
        pipeline_enabled=pipe is not None,
        functionalize_enabled=is_functionalize_enabled(),
        device=dispatch_device,
        autocast_enabled=is_autocast_enabled(autocast_device_type),
    )
    return dispatch_with_keyset(name, keyset, dispatch_device, *args, **kwargs)


def cy_dispatch_with_keyset(str name, keyset, dispatch_device, *args, **kwargs):
    """Cython-accelerated dispatch_with_keyset."""
    from candle._dispatch.dispatcher import dispatch_with_keyset
    return dispatch_with_keyset(name, keyset, dispatch_device, *args, **kwargs)
