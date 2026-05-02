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
