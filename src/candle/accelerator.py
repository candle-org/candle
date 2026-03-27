"""Minimal torch.accelerator compatibility helpers.

This module provides the small compatibility surface needed by the current
PyTorch beginner tutorials without depending on torch at runtime.
"""

from ._device import device as Device, get_default_device

_ACCELERATOR_TYPES = ("npu", "cuda", "mps")


def _availability_checks():
    from . import npu, cuda, mps

    return (
        ("npu", npu.is_available),
        ("cuda", cuda.is_available),
        ("mps", mps.is_available),
    )


def _is_available(device_type):
    for name, check in _availability_checks():
        if name == device_type:
            return bool(check())
    return False


def is_available() -> bool:
    """Return True when any accelerator backend is available at runtime."""
    return current_accelerator() is not None


def current_accelerator(check_available: bool = False):
    """Return the currently usable accelerator device, or None if unavailable.

    Candle does not have PyTorch's compile-time accelerator selection concept, so
    this compatibility shim resolves to a runtime-available accelerator only.
    The optional ``check_available`` argument is accepted for API compatibility.
    """
    del check_available

    default_device = get_default_device()
    if default_device.type in _ACCELERATOR_TYPES and _is_available(default_device.type):
        return Device(default_device.type, index=default_device.index)

    for device_type, check in _availability_checks():
        if check():
            return Device(device_type)
    return None


__all__ = ["is_available", "current_accelerator"]
