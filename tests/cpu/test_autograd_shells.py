"""Compiled-boundary ownership tests for autograd public-shell stubs.

These modules — ``candle.autograd.version_counter`` and
``candle.autograd.profiler_util`` — only host torch-compatible compatibility
shims.  Per the repo-wide rule (Python public surface mirrors torch 1:1,
runtime/mechanism stays in compiled ``_C``), even the stubs must be owned
by a compiled extension, with the Python files acting as thin re-export
shells.
"""

import importlib.machinery


def test_version_counter_lives_in_compiled_c_boundary():
    """VersionCounter (standalone, non-TensorImpl path) should be owned by
    the compiled ``candle._C._tensor_impl`` extension that already hosts
    ``_VersionCounterProxy``."""
    from candle._C import _tensor_impl
    from candle.autograd import version_counter as version_counter_shell

    assert isinstance(_tensor_impl.__loader__, importlib.machinery.ExtensionFileLoader)

    assert version_counter_shell.VersionCounter.__module__ == "candle._C._tensor_impl"
    assert version_counter_shell.VersionCounter is _tensor_impl.VersionCounter


def test_profiler_util_shims_live_in_compiled_c_boundary():
    """The torch-compatible profiler_util stubs must be owned by the
    compiled ``candle._C._autograd_profiler`` extension, not defined in
    the Python public shell."""
    from candle._C import _autograd_profiler
    from candle.autograd import profiler_util as profiler_util_shell

    assert isinstance(_autograd_profiler.__loader__, importlib.machinery.ExtensionFileLoader)

    assert profiler_util_shell.EventList.__module__ == "candle._C._autograd_profiler"
    assert profiler_util_shell.FunctionEvent.__module__ == "candle._C._autograd_profiler"
    assert profiler_util_shell.FunctionEventAvg.__module__ == "candle._C._autograd_profiler"
    assert profiler_util_shell._format_time.__module__ == "candle._C._autograd_profiler"


def test_profiler_util_shims_are_reexports():
    """The Python shell candle.autograd.profiler_util must hand back the
    same compiled objects as ``candle._C._autograd_profiler``."""
    from candle._C import _autograd_profiler
    from candle.autograd import profiler_util as profiler_util_shell

    assert profiler_util_shell.EventList is _autograd_profiler.EventList
    assert profiler_util_shell.FunctionEvent is _autograd_profiler.FunctionEvent
    assert profiler_util_shell.FunctionEventAvg is _autograd_profiler.FunctionEventAvg
    assert profiler_util_shell._format_time is _autograd_profiler._format_time
