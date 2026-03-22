"""Failure stubs for dispatcher_core when the compiled extension is missing.

The runtime-critical dispatcher path requires the compiled Cython core.
This module exists only to provide actionable errors if something imports the
fallback shim directly.
"""


def cy_dispatch_with_keyset_fast(name, keyset, dispatch_device, *args, **kwargs):
    raise ImportError(
        "Failed to import candle._cython._dispatcher_core. Build the required Cython "
        "runtime core with `python setup.py build_ext --inplace` or install with "
        "a build that includes the compiled extensions."
    )


def cy_dispatch_full(name, dispatch_device, *args, **kwargs):
    raise ImportError(
        "Failed to import candle._cython._dispatcher_core. Build the required Cython "
        "runtime core with `python setup.py build_ext --inplace` or install with "
        "a build that includes the compiled extensions."
    )
