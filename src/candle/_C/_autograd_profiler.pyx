# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython-owned autograd profiler compatibility shims."""

from contextlib import contextmanager


def emit_itt(*args, **kwargs):  # noqa: ARG001
    return None


def emit_nvtx(*args, **kwargs):  # noqa: ARG001
    return None


@contextmanager
def profile(*args, **kwargs):  # noqa: ARG001
    yield None


@contextmanager
def record_function(*args, **kwargs):  # noqa: ARG001
    yield None


# ---------------------------------------------------------------------------
# torch.autograd.profiler_util compatibility stubs
# ---------------------------------------------------------------------------
# These mirror torch.autograd.profiler_util's public surface for code that
# imports the names without doing real profiling.  Candle does not link
# kineto / itt / nvtx, so the implementations are intentionally minimal.


class EventList(list):
    """torch.autograd.profiler_util.EventList compatibility stub."""


class FunctionEvent:
    """torch.autograd.profiler_util.FunctionEvent compatibility stub."""


class FunctionEventAvg:
    """torch.autograd.profiler_util.FunctionEventAvg compatibility stub."""


def _format_time(*args, **kwargs):  # noqa: ARG001
    """torch.autograd.profiler_util._format_time compatibility stub."""
    return "0.0us"
