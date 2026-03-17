"""Pure-Python fallback for _dispatcher_core.pyx.

When Cython is not available, dispatch_with_keyset remains the original
Python implementation in dispatcher.py — this module is only imported
to satisfy the conditional import pattern.
"""


def cy_dispatch_with_keyset_fast(name, keyset, dispatch_device, *args, **kwargs):
    """Fallback: delegate to the original Python dispatch_with_keyset."""
    from candle._dispatch.dispatcher import _py_dispatch_with_keyset
    return _py_dispatch_with_keyset(name, keyset, dispatch_device, *args, **kwargs)
