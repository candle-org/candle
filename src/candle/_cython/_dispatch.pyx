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

# Dispatch key bit constants (must match keys.py)
DEF _DK_CPU = 1 << 15
DEF _DK_NPU = 1 << 13
DEF _DK_CUDA = 1 << 14
DEF _DK_MPS = 1 << 21       # PrivateUse2
DEF _DK_META = 1 << 12
DEF _DK_AUTOGRAD_CPU = 1 << 6
DEF _DK_AUTOGRAD_NPU = 1 << 7
DEF _DK_AUTOGRAD_CUDA = 1 << 8
DEF _DK_AUTOGRAD_MPS = 1 << 22  # PrivateUse3
DEF _DK_AUTOGRAD_META = 1 << 10
DEF _DK_ADINPLACEORVIEW = 1 << 4
DEF _DK_AUTOGRAD = 1 << 11
DEF _DK_FUNCTIONALIZE = 1 << 3
DEF _DK_AUTOCAST = 1 << 19
DEF _DK_PIPELINE = 1 << 1
DEF _DK_PYTHON = 1 << 2

# Device type to backend key mapping (indexed by _device_type int)
# 0=cpu, 1=npu, 2=cuda, 3=mps, 4=meta
cdef unsigned int _DEVICE_BACKEND_KEY[5]
_DEVICE_BACKEND_KEY[0] = _DK_CPU
_DEVICE_BACKEND_KEY[1] = _DK_NPU
_DEVICE_BACKEND_KEY[2] = _DK_CUDA
_DEVICE_BACKEND_KEY[3] = _DK_MPS
_DEVICE_BACKEND_KEY[4] = _DK_META

# Device type to autograd key mapping
cdef unsigned int _DEVICE_AUTOGRAD_KEY[5]
_DEVICE_AUTOGRAD_KEY[0] = _DK_AUTOGRAD_CPU
_DEVICE_AUTOGRAD_KEY[1] = _DK_AUTOGRAD_NPU
_DEVICE_AUTOGRAD_KEY[2] = _DK_AUTOGRAD_CUDA
_DEVICE_AUTOGRAD_KEY[3] = _DK_AUTOGRAD_MPS
_DEVICE_AUTOGRAD_KEY[4] = _DK_AUTOGRAD_META

cdef FastDispatchKeySet _cy_from_tensors(list tensors, bint grad_enabled,
                                          bint pipeline_enabled,
                                          bint functionalize_enabled,
                                          object device, bint autocast_enabled):
    """Build keyset from tensors — C-speed bitmask construction.

    Step 4 optimization: reads _device_type and requires_grad directly from
    TensorImpl C fields when available, avoiding Python attribute access and
    string comparison.
    """
    from candle._dispatch.keys import DispatchKey

    cdef bint has_meta = False, has_npu = False, has_cuda = False
    cdef bint has_mps = False, has_cpu = False
    cdef bint requires_grad = False, saw_device = False
    cdef bint has_dispatch_subclass = False
    cdef unsigned int mask = 0
    cdef int dev_type_int

    global _BaseTensor
    if _BaseTensor is None:
        from candle._tensor import Tensor
        _BaseTensor = Tensor

    for tensor in tensors:
        # Fast path: read _device_type directly from TensorImpl
        dev_type_int = getattr(tensor, "_device_type", -1)
        if dev_type_int >= 0:
            saw_device = True
            if dev_type_int == 4:    # meta
                has_meta = True
            elif dev_type_int == 1:  # npu
                has_npu = True
            elif dev_type_int == 2:  # cuda
                has_cuda = True
            elif dev_type_int == 3:  # mps
                has_mps = True
            else:                    # cpu (0) or unknown
                has_cpu = True
        else:
            # Fallback for non-TensorImpl objects
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
        # Read requires_grad directly (C field access for TensorImpl)
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

    # Build backend key from device flags
    if has_meta:
        mask = _DK_META
    elif has_npu:
        mask = _DK_NPU
    elif has_cuda:
        mask = _DK_CUDA
    elif has_mps:
        mask = _DK_MPS
    else:
        mask = _DK_CPU

    if grad_enabled and requires_grad:
        mask |= _DK_ADINPLACEORVIEW | _DK_AUTOGRAD
        if has_meta:
            mask |= _DK_AUTOGRAD_META
        elif has_npu:
            mask |= _DK_AUTOGRAD_NPU
        elif has_cuda:
            mask |= _DK_AUTOGRAD_CUDA
        elif has_mps:
            mask |= _DK_AUTOGRAD_MPS
        else:
            mask |= _DK_AUTOGRAD_CPU

    if functionalize_enabled:
        mask |= _DK_FUNCTIONALIZE
    if autocast_enabled:
        mask |= _DK_AUTOCAST
    if pipeline_enabled and not has_meta and not has_cuda:
        mask |= _DK_PIPELINE
    if has_dispatch_subclass:
        mask |= _DK_PYTHON

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
# Public dispatch entry points
# ---------------------------------------------------------------------------
# cy_dispatch has been replaced by cy_dispatch_full in _dispatcher_core.pyx.
# cy_dispatch_with_keyset has been replaced by cy_dispatch_with_keyset_fast
# in _dispatcher_core.pyx.
# The utility functions (cy_extract_tensors, cy_prepare_kwargs,
# FastDispatchKeySet, cy_apply_tls_masks, cy_kernel_for_entry) remain here
# for backward compatibility and use by other modules.
