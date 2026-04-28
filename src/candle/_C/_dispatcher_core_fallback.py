"""Pure-Python fallback for _dispatcher_core.pyx.

When Cython is not available, dispatcher.py keeps its original Python
definitions; these helpers mirror the compiled module surface for any direct
imports that expect dispatcher-core entry points.
"""


def cy_dispatch_full(name, dispatch_device, *args, **kwargs):
    """Fallback: delegate to the original Python dispatch."""
    from candle._dispatch.dispatcher import _py_dispatch_with_keyset
    from candle._dispatch.dispatcher import DispatchKeySet, _extract_tensors
    from candle._dispatch.dispatcher import current_pipeline
    from candle._dispatch.functionalize import is_functionalize_enabled
    from candle.autograd.grad_mode import is_grad_enabled
    from candle.amp.state import is_autocast_enabled

    tensors = _extract_tensors(args, kwargs)
    autocast_device_type = getattr(dispatch_device, "type", dispatch_device)
    if autocast_device_type is None and tensors:
        autocast_device_type = getattr(tensors[0].device, "type", None)
    pipe = current_pipeline()
    keyset = DispatchKeySet.from_tensors(
        tensors,
        grad_enabled=is_grad_enabled(),
        pipeline_enabled=pipe is not None,
        functionalize_enabled=is_functionalize_enabled(),
        device=dispatch_device,
        autocast_enabled=is_autocast_enabled(autocast_device_type),
    )
    return _py_dispatch_with_keyset(name, keyset, dispatch_device, *args, **kwargs)


def cy_dispatch_with_keyset_fast(name, keyset, dispatch_device, *args, **kwargs):
    """Fallback: delegate to the original Python dispatch_with_keyset."""
    from candle._dispatch.dispatcher import _py_dispatch_with_keyset
    return _py_dispatch_with_keyset(name, keyset, dispatch_device, *args, **kwargs)


def cy_prepare_dispatch_inputs(func, args, kwargs, device):
    """Fallback: mirror dispatcher-core argument preparation."""
    try:
        from inspect import signature
        accepts_device = "device" in signature(func).parameters
    except (TypeError, ValueError):
        accepts_device = False
    if not kwargs:
        if accepts_device:
            return args, {"device": device}
        return args, {}
    if "device" in kwargs:
        if accepts_device:
            return args, kwargs
        return args, {k: v for k, v in kwargs.items() if k != "device"}
    if accepts_device:
        merged = dict(kwargs)
        merged["device"] = device
        return args, merged
    return args, kwargs
