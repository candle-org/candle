# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython full dispatcher — single-function dispatch aligned with PyTorch Dispatcher::call.

Merges the entire dispatch call chain (keyset construction, tensor extraction,
device validation, TLS mask application, schema validation, kernel lookup,
kwargs preparation, context push/pop, version bumping, single-pass autograd)
into a single cdef _dispatch_core function with zero intermediate Python
function boundaries on the hot path.
"""

from libc.stdint cimport int64_t, uint32_t
import os
import inspect

# ---------------------------------------------------------------------------
# Schema validation fast-path: skip after N successful validations
# ---------------------------------------------------------------------------

cdef dict _validated_ops = {}
cdef int _VALIDATION_THRESHOLD = 100
cdef bint _FORCE_VALIDATE = os.environ.get("CANDLE_DEBUG_SCHEMA", "") == "1"

cdef inline bint _should_validate(str name):
    if _FORCE_VALIDATE:
        return True
    cdef int count = _validated_ops.get(name, 0)
    if count >= _VALIDATION_THRESHOLD:
        return False
    _validated_ops[name] = count + 1
    return True

# ---------------------------------------------------------------------------
# Cached TLS object references for fast access
# ---------------------------------------------------------------------------

cdef object _grad_tls = None

cdef inline bint _fast_grad_enabled():
    global _grad_tls
    if _grad_tls is None:
        from candle.autograd.grad_mode import _GRAD_MODE_STATE
        _grad_tls = _GRAD_MODE_STATE
    return getattr(_grad_tls, "enabled", True)

# Cached is_functionalize_enabled
cdef object _is_func_enabled_fn = None

cdef inline bint _fast_functionalize_enabled():
    global _is_func_enabled_fn
    if _is_func_enabled_fn is None:
        from candle._dispatch.functionalize import is_functionalize_enabled
        _is_func_enabled_fn = is_functionalize_enabled
    return _is_func_enabled_fn()

# ---------------------------------------------------------------------------
# Cached accepts_device check
# ---------------------------------------------------------------------------

cdef dict _accepts_device_cache = {}

cdef inline bint _cy_accepts_device(object func):
    cdef object cached = _accepts_device_cache.get(func)
    if cached is not None:
        return <bint>cached
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        _accepts_device_cache[func] = False
        return False
    cdef bint result = "device" in sig.parameters
    _accepts_device_cache[func] = result
    return result


cdef inline dict _fast_prepare_kwargs(object func, dict kwargs, object device):
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
# Dispatch context stack — MUST share with dispatcher.py's _DISPATCH_STATE
# ---------------------------------------------------------------------------

cdef object _DISPATCH_STATE = None

cdef inline object _get_dispatch_state():
    global _DISPATCH_STATE
    if _DISPATCH_STATE is None:
        from candle._dispatch.dispatcher import _DISPATCH_STATE as _ds
        _DISPATCH_STATE = _ds
    return _DISPATCH_STATE

cdef inline list _fast_state_stack():
    cdef object state = _get_dispatch_state()
    cdef object stack = getattr(state, "stack", None)
    if stack is None:
        stack = []
        state.stack = stack
    return <list>stack

cdef inline void _fast_push(object keyset, object key):
    _fast_state_stack().append((keyset, key))

cdef inline void _fast_pop():
    cdef list stack = _fast_state_stack()
    if stack:
        stack.pop()


# ---------------------------------------------------------------------------
# Fast version bumping (inlined _version_value access)
# ---------------------------------------------------------------------------

cdef void _fast_bump_versions(object schema_obj, tuple args, dict kwargs):
    if schema_obj is None:
        return
    cdef list params = schema_obj.params
    cdef list positional = [p for p in params if not p.kw_only]
    cdef dict bound = {}
    cdef int idx
    cdef set seen = set()
    cdef int64_t tid
    for idx in range(len(args)):
        if idx < len(positional):
            bound[positional[idx].name] = args[idx]
    if kwargs:
        for key, value in kwargs.items():
            bound[key] = value
    for param in params:
        if not param.mutates:
            continue
        alias = getattr(param, "alias_set", None)
        if alias is None or alias == "":
            continue
        target = bound.get(param.name)
        if target is None:
            continue
        base = getattr(target, "_base", None)
        if base is not None:
            target = base
        dev = getattr(target, "device", None)
        if dev is not None and getattr(dev, "type", None) == "meta":
            continue
        tid = id(target)
        if tid in seen:
            continue
        # Prefer direct _version_value (TensorImpl), fall back to _version_counter
        try:
            target._version_value += 1
        except AttributeError:
            counter = getattr(target, "_version_counter", None)
            if counter is not None:
                counter.bump()
        seen.add(tid)


# ---------------------------------------------------------------------------
# Core dispatch_with_keyset replacement
# ---------------------------------------------------------------------------

import numpy as np

# Cached module-level references (lazy-loaded)
cdef object _registry = None
cdef object _DispatchKey = None
cdef object _DISPATCH_KEY_PRIORITY = None

# Precomputed autograd key bitmask (set on first _ensure_imports)
cdef unsigned int _AUTOGRAD_MASK = 0
# Precomputed individual autograd key values
cdef unsigned int _KEY_ADInplaceOrView = 0
cdef unsigned int _KEY_Autograd = 0
cdef unsigned int _KEY_AutogradCPU = 0
cdef unsigned int _KEY_AutogradNPU = 0
cdef unsigned int _KEY_AutogradCUDA = 0
cdef unsigned int _KEY_AutogradXPU = 0
cdef unsigned int _KEY_AutogradMeta = 0
cdef unsigned int _KEY_AutogradOther = 0
cdef unsigned int _KEY_PrivateUse3 = 0
cdef unsigned int _KEY_Functionalize = 0
cdef unsigned int _KEY_Python = 0
cdef unsigned int _KEY_Autocast = 0
cdef unsigned int _KEY_Pipeline = 0
cdef unsigned int _KEY_Meta = 0
cdef unsigned int _KEY_NPU = 0
cdef unsigned int _KEY_CUDA = 0
cdef unsigned int _KEY_CPU = 0
cdef unsigned int _KEY_PrivateUse2 = 0

# Cached _strip_autograd_keys function
cdef object _strip_autograd_keys_fn = None
cdef object _FastDispatchKeySet = None

# Cached base Tensor class for __torch_dispatch__ subclass check
cdef object _BaseTensor = None

# Cached dispatch helper functions (lazy-loaded in _ensure_dispatch_helpers)
cdef object _apply_tls_masks_fn = None
cdef object _current_pipeline_fn = None
cdef object _should_functionalize_fn = None
cdef object _functionalize_op_fn = None
cdef object _is_autocast_enabled_fn = None
cdef object _apply_autocast_policy_fn = None
cdef object _forward_ad_mod = None
cdef object _is_profiler_enabled_fn = None
cdef object _dispatch_op_enter_fn = None
cdef object _dispatch_op_exit_fn = None
cdef object _infer_dispatch_device_fn = None
cdef object _wrap_dispatch_error_fn = None
cdef object _check_inplace_targets_fn = None
cdef object _PendingOp_cls = None
cdef object _pending_from_meta_fn = None
cdef object _dispatch_torch_dispatch_fn = None
cdef object _redispatch_fn = None

# Cached TLS state reference for fast apply_tls_masks
cdef object _tls_state_fn = None

cdef inline void _ensure_dispatch_helpers():
    global _apply_tls_masks_fn, _current_pipeline_fn
    global _should_functionalize_fn, _functionalize_op_fn
    global _is_autocast_enabled_fn, _apply_autocast_policy_fn
    global _forward_ad_mod, _is_profiler_enabled_fn
    global _dispatch_op_enter_fn, _dispatch_op_exit_fn
    global _infer_dispatch_device_fn, _wrap_dispatch_error_fn
    global _check_inplace_targets_fn, _PendingOp_cls
    global _pending_from_meta_fn, _dispatch_torch_dispatch_fn
    global _redispatch_fn
    global _tls_state_fn
    if _apply_tls_masks_fn is not None:
        return
    from candle._dispatch.keys import apply_tls_masks
    from candle._dispatch.pipeline import current_pipeline
    from candle._dispatch.functionalize import should_functionalize, functionalize_op
    from candle.amp.state import is_autocast_enabled
    from candle.amp.policy import apply_autocast_policy
    from candle.autograd import forward_ad
    from candle.profiler.profiler import is_profiler_enabled, dispatch_op_enter, dispatch_op_exit
    from candle._dispatch.dispatcher import (
        _infer_dispatch_device, _wrap_dispatch_error,
        _check_inplace_targets, _PendingOp,
        _pending_from_meta, _dispatch_torch_dispatch,
        redispatch,
    )
    _apply_tls_masks_fn = apply_tls_masks
    _current_pipeline_fn = current_pipeline
    _should_functionalize_fn = should_functionalize
    _functionalize_op_fn = functionalize_op
    _is_autocast_enabled_fn = is_autocast_enabled
    _apply_autocast_policy_fn = apply_autocast_policy
    _forward_ad_mod = forward_ad
    _is_profiler_enabled_fn = is_profiler_enabled
    _dispatch_op_enter_fn = dispatch_op_enter
    _dispatch_op_exit_fn = dispatch_op_exit
    _infer_dispatch_device_fn = _infer_dispatch_device
    _wrap_dispatch_error_fn = _wrap_dispatch_error
    _check_inplace_targets_fn = _check_inplace_targets
    _PendingOp_cls = _PendingOp
    _pending_from_meta_fn = _pending_from_meta
    _dispatch_torch_dispatch_fn = _dispatch_torch_dispatch
    _redispatch_fn = redispatch
    from candle._dispatch.keys import _tls_state
    _tls_state_fn = _tls_state

cdef inline void _ensure_imports():
    global _registry, _DispatchKey, _DISPATCH_KEY_PRIORITY
    global _AUTOGRAD_MASK
    global _KEY_ADInplaceOrView, _KEY_Autograd, _KEY_AutogradCPU
    global _KEY_AutogradNPU, _KEY_AutogradCUDA, _KEY_AutogradXPU
    global _KEY_AutogradMeta, _KEY_AutogradOther, _KEY_PrivateUse3
    global _KEY_Functionalize, _KEY_Python, _KEY_Autocast, _KEY_Pipeline
    global _KEY_Meta, _KEY_NPU, _KEY_CUDA, _KEY_CPU, _KEY_PrivateUse2
    global _strip_autograd_keys_fn
    global _FastDispatchKeySet
    global _BaseTensor
    if _registry is not None:
        return
    from candle._dispatch.registry import registry
    from candle._dispatch.keys import DispatchKey, DISPATCH_KEY_PRIORITY
    from candle._backends.autograd import _strip_autograd_keys
    from candle._cython._dispatch import FastDispatchKeySet
    from candle._tensor import Tensor
    _registry = registry
    _DispatchKey = DispatchKey
    _DISPATCH_KEY_PRIORITY = DISPATCH_KEY_PRIORITY
    _strip_autograd_keys_fn = _strip_autograd_keys
    _FastDispatchKeySet = FastDispatchKeySet
    _BaseTensor = Tensor
    # Cache individual key values as C unsigned ints
    _KEY_ADInplaceOrView = <unsigned int>int(DispatchKey.ADInplaceOrView)
    _KEY_Autograd = <unsigned int>int(DispatchKey.Autograd)
    _KEY_AutogradCPU = <unsigned int>int(DispatchKey.AutogradCPU)
    _KEY_AutogradNPU = <unsigned int>int(DispatchKey.AutogradNPU)
    _KEY_AutogradCUDA = <unsigned int>int(DispatchKey.AutogradCUDA)
    _KEY_AutogradXPU = <unsigned int>int(DispatchKey.AutogradXPU)
    _KEY_AutogradMeta = <unsigned int>int(DispatchKey.AutogradMeta)
    _KEY_AutogradOther = <unsigned int>int(DispatchKey.AutogradOther)
    _KEY_PrivateUse3 = <unsigned int>int(DispatchKey.PrivateUse3)
    _KEY_Functionalize = <unsigned int>int(DispatchKey.Functionalize)
    _KEY_Python = <unsigned int>int(DispatchKey.Python)
    _KEY_Autocast = <unsigned int>int(DispatchKey.Autocast)
    _KEY_Pipeline = <unsigned int>int(DispatchKey.Pipeline)
    _KEY_Meta = <unsigned int>int(DispatchKey.Meta)
    _KEY_NPU = <unsigned int>int(DispatchKey.NPU)
    _KEY_CUDA = <unsigned int>int(DispatchKey.CUDA)
    _KEY_CPU = <unsigned int>int(DispatchKey.CPU)
    _KEY_PrivateUse2 = <unsigned int>int(DispatchKey.PrivateUse2)
    # Autograd mask = all autograd keys OR'd together
    _AUTOGRAD_MASK = (_KEY_AutogradOther | _KEY_AutogradCPU | _KEY_AutogradNPU |
                      _KEY_AutogradCUDA | _KEY_AutogradXPU | _KEY_AutogradMeta |
                      _KEY_Autograd | _KEY_PrivateUse3)


cdef inline list _fast_extract_tensors(tuple args, dict kwargs):
    """Extract tensors from args/kwargs — typed C loop."""
    cdef list tensors = []
    cdef int n = len(args)
    # Fast path: binary op (2 tensor args, no kwargs with tensors)
    if n == 2 and not kwargs:
        a = args[0]
        b = args[1]
        if hasattr(a, "device") and hasattr(b, "device"):
            tensors.append(a)
            tensors.append(b)
            return tensors
    cdef int i
    for i in range(n):
        _fast_visit(args[i], tensors)
    if kwargs:
        for v in kwargs.values():
            _fast_visit(v, tensors)
    return tensors


cdef inline void _fast_visit(object value, list tensors):
    if hasattr(value, "device") and not isinstance(value, np.ndarray):
        tensors.append(value)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _fast_visit(item, tensors)


cdef inline void _fast_validate_devices(list tensors):
    """Validate all tensors are on the same device."""
    cdef int n = len(tensors)
    if n <= 1:
        return
    cdef object expected = tensors[0].device
    cdef str expected_type = getattr(expected, "type", expected)
    cdef object expected_index = getattr(expected, "index", None)
    cdef int i
    for i in range(1, n):
        dev = tensors[i].device
        dt = getattr(dev, "type", dev)
        di = getattr(dev, "index", None)
        if dt != expected_type or di != expected_index:
            el = expected_type if expected_index is None else f"{expected_type}:{expected_index}"
            dl = dt if di is None else f"{dt}:{di}"
            raise RuntimeError(
                f"Tensor on device {dl} is not on the expected device {el}!"
            )


cdef inline object _fast_kernel_for_entry(object entry, unsigned int m):
    """Find kernel for dispatch entry — direct bitmask iteration."""
    fallthrough = entry.fallthrough
    kernels = entry.kernels
    global_fallthrough = getattr(_registry, "_global_fallthrough", set())
    cdef unsigned int kv
    for key in _DISPATCH_KEY_PRIORITY:
        kv = <unsigned int>int(key)
        if not (m & kv):
            continue
        if key in fallthrough:
            continue
        kernel = kernels.get(key)
        if kernel is not None:
            return kernel, key
        if key in global_fallthrough:
            continue
    return None, None


cdef inline object _fast_kernel_for_entry_skip_autograd(object entry, unsigned int m):
    """Find backend kernel, skipping autograd keys entirely.

    Used by single-pass dispatch when autograd_post is available.
    Returns (kernel, key) for the first non-autograd, non-fallthrough key.
    """
    fallthrough = entry.fallthrough
    kernels = entry.kernels
    global_fallthrough = getattr(_registry, "_global_fallthrough", set())
    # Strip autograd + ADInplaceOrView from mask
    cdef unsigned int backend_mask = m & ~(_AUTOGRAD_MASK | _KEY_ADInplaceOrView)
    cdef unsigned int kv
    for key in _DISPATCH_KEY_PRIORITY:
        kv = <unsigned int>int(key)
        if not (backend_mask & kv):
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
# Inline keyset construction — replaces _dispatch.pyx::_cy_from_tensors
# ---------------------------------------------------------------------------

# Dispatch key bit constants (must match keys.py / _dispatch.pyx DEF values)
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

cdef inline object _build_keyset_inline(list tensors, object dispatch_device):
    """Build FastDispatchKeySet from tensors — fully inlined.

    Reads _device_type and requires_grad directly from TensorImpl C fields
    when available, checks grad_enabled/pipeline/functionalize/autocast state,
    and returns a FastDispatchKeySet. No cross-module Python calls.
    """
    cdef bint has_meta = False, has_npu = False, has_cuda = False
    cdef bint has_mps = False, has_cpu = False
    cdef bint requires_grad = False, saw_device = False
    cdef bint has_dispatch_subclass = False
    cdef unsigned int mask = 0
    cdef int dev_type_int

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

    if (not saw_device) and dispatch_device is not None:
        dev_type = getattr(dispatch_device, "type", dispatch_device)
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

    # Autograd keys — check grad_enabled inline
    if _fast_grad_enabled() and requires_grad:
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

    if _fast_functionalize_enabled():
        mask |= _DK_FUNCTIONALIZE
    # Autocast: infer device type for autocast check
    cdef object autocast_device_type = getattr(dispatch_device, "type", dispatch_device)
    if autocast_device_type is None and tensors:
        autocast_device_type = getattr(tensors[0].device, "type", None)
    if _is_autocast_enabled_fn is not None and _is_autocast_enabled_fn(autocast_device_type):
        mask |= _DK_AUTOCAST
    if _current_pipeline_fn is not None and _current_pipeline_fn() is not None and not has_meta and not has_cuda:
        mask |= _DK_PIPELINE
    if has_dispatch_subclass:
        mask |= _DK_PYTHON

    return _FastDispatchKeySet(mask)


# ---------------------------------------------------------------------------
# Inline TLS mask application
# ---------------------------------------------------------------------------

cdef inline object _apply_tls_masks_inline(object keyset):
    """Apply TLS include/exclude masks — inline bitmask ops.

    Avoids the cross-module Python call to apply_tls_masks.
    """
    cdef unsigned int base_mask
    if hasattr(keyset, "mask"):
        base_mask = <unsigned int>int(keyset.mask)
    else:
        base_mask = <unsigned int>int(keyset)

    state = _tls_state_fn()
    cdef unsigned int include_mask = 0
    cdef unsigned int exclude_mask = 0
    for m_val in state["include"]:
        include_mask |= <unsigned int>m_val
    for m_val in state["exclude"]:
        exclude_mask |= <unsigned int>m_val

    return _FastDispatchKeySet((base_mask | include_mask) & ~exclude_mask)


# ---------------------------------------------------------------------------
# _dispatch_core — the unified dispatch function (cdef, C-level)
# ---------------------------------------------------------------------------

cdef object _dispatch_core(str name, object dispatch_device,
                           tuple args, dict kwargs,
                           object keyset, list tensors):
    """Core dispatch — single C-level function, no intermediate Python boundaries.

    Parameters
    ----------
    name : str
        Op name (e.g. "add").
    dispatch_device : object
        Target device, or None to infer.
    args, kwargs : tuple, dict
        Op arguments.
    keyset : object or None
        Pre-built keyset (for redispatch), or None to build from tensors.
    tensors : list or None
        Pre-extracted tensors, or None to extract from args/kwargs.
    """
    # All cdef declarations at function top (Cython requirement)
    cdef str alias_name
    cdef object schema_obj
    cdef unsigned int m
    cdef bint has_autograd
    cdef object autograd_post_fn = None
    cdef object backend_kernel = None
    cdef object backend_key = None
    cdef dict impl_kwargs
    cdef object token = None

    # -- Tensor extraction (once) --
    if tensors is None:
        tensors = _fast_extract_tensors(args, kwargs)
        _fast_validate_devices(tensors)

    # -- Keyset construction or TLS application --
    if keyset is None:
        # Full dispatch: build keyset from tensors (inlined)
        keyset = _build_keyset_inline(tensors, dispatch_device)
    # Apply TLS masks (inlined)
    keyset = _apply_tls_masks_inline(keyset)

    # -- Pipeline (read once) --
    pipe = _current_pipeline_fn()

    # -- Infer dispatch device --
    dispatch_device = _infer_dispatch_device_fn(dispatch_device, tensors, keyset)

    # -- Registry lookup --
    alias_name = name
    try:
        name = _registry.resolve(name)
        entry = _registry.get(name)
    except Exception as exc:
        raise _wrap_dispatch_error_fn(exc, alias_name, dispatch_device) from exc

    # -- Schema validation --
    schema_obj = entry.schema_obj
    if schema_obj is not None:
        schema_obj.bind(args, kwargs, op_name=alias_name,
                        error_overrides=entry.error_overrides)
    if schema_obj is not None and keyset.has(_DispatchKey.ADInplaceOrView):
        _check_inplace_targets_fn(schema_obj, args, kwargs)

    # -- Functionalize (cold path) --
    if keyset.has(_DispatchKey.Functionalize) and _should_functionalize_fn(entry):
        if pipe is not None and keyset.has(_DispatchKey.Pipeline):
            return _functionalize_op_fn(name, alias_name, entry, keyset, args, kwargs,
                                    _redispatch_fn, pipeline=pipe, dispatch_device=dispatch_device)
        return _functionalize_op_fn(name, alias_name, entry, keyset, args, kwargs,
                                _redispatch_fn, dispatch_device=dispatch_device)

    # -- __torch_dispatch__ (cold path) --
    if keyset.has(_DispatchKey.Python):
        result = _dispatch_torch_dispatch_fn(alias_name, tensors, args, kwargs)
        if result is not NotImplemented:
            return result

    # -- Autocast (cold path) --
    if keyset.has(_DispatchKey.Autocast):
        device_type = getattr(dispatch_device, "type", dispatch_device)
        if device_type is None and tensors:
            device_type = getattr(tensors[0].device, "type", None)
        casted_args, casted_kwargs = _apply_autocast_policy_fn(alias_name, args, kwargs, device_type)
        return _redispatch_fn(alias_name, keyset.without(_DispatchKey.Autocast),
                          *casted_args, **casted_kwargs)

    # -- Pipeline deferred execution (cold path) --
    if pipe is not None and keyset.has(_DispatchKey.Pipeline):
        if not pipe.should_defer_next():
            # Run kernel immediately (fall through to hot path below)
            pass
        else:
            meta = entry.kernels.get(_DispatchKey.Meta)
            if meta is None:
                raise RuntimeError(f"pipeline requires meta kernel for op {name}")
            meta_kwargs = _fast_prepare_kwargs(meta, kwargs, dispatch_device)
            spec = meta(*args, **meta_kwargs)
            out = _pending_from_meta_fn(spec, dispatch_device)
            if isinstance(out, (tuple, list)):
                for item in out:
                    item._pending = True
            else:
                out._pending = True
            impl, impl_key = _fast_kernel_for_entry(entry, keyset.without(_DispatchKey.Pipeline).mask)
            if impl is None:
                raise RuntimeError(f"pipeline requires backend kernel for op {name}")
            impl_kwargs = _fast_prepare_kwargs(impl, kwargs, dispatch_device)
            pipe.record(
                _PendingOp_cls(impl, args, impl_kwargs, out,
                           keyset.without(_DispatchKey.Pipeline), impl_key,
                           schema_obj=schema_obj, op_name=alias_name),
                pending=out,
            )
            return out

    if pipe is not None and keyset.has(_DispatchKey.Pipeline):
        pipe.flush()

    # =====================================================================
    # HOT PATH: kernel execution (inlined from _run_kernel_fast)
    # =====================================================================

    if hasattr(keyset, "mask"):
        m = <unsigned int>int(keyset.mask)
    else:
        m = <unsigned int>int(keyset)

    # -- Single-pass autograd --
    has_autograd = (m & _AUTOGRAD_MASK) != 0

    if has_autograd:
        autograd_post_fn = getattr(entry, "autograd_post", None)

    if has_autograd and autograd_post_fn is not None:
        # Single-pass: find backend kernel directly, skip autograd wrapper
        backend_kernel, backend_key = _fast_kernel_for_entry_skip_autograd(entry, m)
        if backend_kernel is not None:
            active_keyset = keyset
            raw_keyset = _FastDispatchKeySet(m & ~(_AUTOGRAD_MASK | _KEY_ADInplaceOrView | _KEY_PrivateUse3))

            impl_kwargs = _fast_prepare_kwargs(backend_kernel, kwargs, dispatch_device)

            token = None
            if _is_profiler_enabled_fn():
                token = _dispatch_op_enter_fn(alias_name, dispatch_device, args, impl_kwargs)

            _fast_push(raw_keyset, backend_key)
            try:
                result = backend_kernel(*args, **impl_kwargs)
            except Exception as exc:
                raise _wrap_dispatch_error_fn(exc, alias_name, dispatch_device) from exc
            finally:
                _fast_pop()
                if token is not None:
                    _dispatch_op_exit_fn(token)

            _fast_bump_versions(schema_obj, args, impl_kwargs)

            # Apply autograd post-processing (attach grad_fn)
            result = autograd_post_fn(result, *args, raw_keyset=raw_keyset, active_keyset=active_keyset, **kwargs)

            # -- Forward AD (JVP) --
            _handle_forward_ad(_forward_ad_mod, alias_name, keyset, backend_key,
                               tensors, args, impl_kwargs, result)

            return result
        # If no backend kernel found via skip_autograd, fall through

    # -- Normal path (no single-pass optimization) --
    kernel, key = _fast_kernel_for_entry(entry, m)
    if kernel is None:
        key_names = [k.name for k in keyset.iter_keys()]
        exc = RuntimeError(
            f"could not find kernel for op {alias_name} with keys {key_names}"
        )
        raise _wrap_dispatch_error_fn(exc, alias_name, dispatch_device) from exc

    impl_kwargs = _fast_prepare_kwargs(kernel, kwargs, dispatch_device)

    token = None
    if _is_profiler_enabled_fn():
        token = _dispatch_op_enter_fn(alias_name, dispatch_device, args, impl_kwargs)

    _fast_push(keyset, key)
    try:
        result = kernel(*args, **impl_kwargs)
    except Exception as exc:
        raise _wrap_dispatch_error_fn(exc, alias_name, dispatch_device) from exc
    finally:
        _fast_pop()
        if token is not None:
            _dispatch_op_exit_fn(token)

    _fast_bump_versions(schema_obj, args, impl_kwargs)

    # -- Forward AD (JVP) --
    _handle_forward_ad(_forward_ad_mod, alias_name, keyset, key,
                       tensors, args, impl_kwargs, result)

    return result


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def cy_dispatch_full(str name, object dispatch_device, *args, **kwargs):
    """Full dispatch — single Python entry point replacing cy_dispatch.

    Builds keyset from scratch, extracts tensors, and runs the full dispatch
    pipeline in one C-level call chain.
    """
    _ensure_imports()
    _ensure_dispatch_helpers()
    return _dispatch_core(name, dispatch_device, args, kwargs, None, None)


def cy_dispatch_with_keyset_fast(str name, object keyset, object dispatch_device,
                                  *args, **kwargs):
    """Thin wrapper for redispatch callers — delegates to _dispatch_core.

    The keyset is pre-built by the caller; _dispatch_core applies TLS masks
    and skips keyset construction.
    """
    _ensure_imports()
    _ensure_dispatch_helpers()
    return _dispatch_core(name, dispatch_device, args, kwargs, keyset, None)


# ---------------------------------------------------------------------------
# Forward AD helper (shared by both paths)
# ---------------------------------------------------------------------------

cdef void _handle_forward_ad(object forward_ad, str alias_name, object keyset,
                              object key, list tensors, tuple args,
                              dict impl_kwargs, object result):
    """Handle forward-mode AD (JVP) after kernel execution."""
    cdef int level = forward_ad._current_level()
    if level < 0:
        return
    tangents = [forward_ad.get_tangent(t, level) for t in tensors]
    if not any(t is not None for t in tangents):
        return
    jvp = forward_ad.get_jvp(alias_name)
    if jvp is None:
        raise RuntimeError(
            f"no forward-mode rule registered for op {alias_name}"
        )
    inner_keyset = keyset.without({
        _DispatchKey.Autograd, _DispatchKey.AutogradCPU,
        _DispatchKey.AutogradCUDA, _DispatchKey.AutogradNPU,
        _DispatchKey.AutogradMeta, _DispatchKey.AutogradXPU,
        _DispatchKey.AutogradOther,
    })
    jvp_key = key
    if jvp_key in inner_keyset:
        inner_keyset = inner_keyset.without(jvp_key)

    def _eval_jvp():
        _fast_push(inner_keyset, jvp_key)
        try:
            return jvp(*args, **impl_kwargs, _tangents=tangents)
        finally:
            _fast_pop()

    try:
        with forward_ad.temporarily_disable(level):
            out_tangent = _eval_jvp()
    except Exception:
        out_tangent = _eval_jvp()

    if isinstance(result, (tuple, list)):
        for out, t in zip(result, out_tangent):
            if hasattr(out, "_fw_set"):
                out._fw_set(level, t)
    else:
        if hasattr(result, "_fw_set"):
            result._fw_set(level, out_tangent)
