"""Stub for torch.testing._internal.common_mkldnn — MKLDNN test helpers."""
import contextlib


@contextlib.contextmanager
def bf32_on_and_off():
    """No-op context manager — candle has no MKLDNN bf32 toggle."""
    yield
