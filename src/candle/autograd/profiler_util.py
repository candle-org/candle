"""Public shell for ``candle.autograd.profiler_util``.

Mirrors torch.autograd.profiler_util's public surface; the runtime owner is
``candle._C._autograd_profiler`` (which already hosts the profiler context
managers).  This module is only a thin re-export so that
``candle.autograd.profiler_util.{EventList, FunctionEvent, FunctionEventAvg,
_format_time}`` keep working.
"""

from .._C._autograd_profiler import (  # pylint: disable=import-error,no-name-in-module
    EventList,
    FunctionEvent,
    FunctionEventAvg,
    _format_time,
)


__all__ = ["EventList", "FunctionEvent", "FunctionEventAvg", "_format_time"]
