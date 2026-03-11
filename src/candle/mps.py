"""Public API for the MPS (Metal Performance Shaders) backend.

Usage:
    import candle as torch
    torch.mps.is_available()
    torch.mps.synchronize()
"""
import sys


def is_available():
    """Return True if Metal/MPS GPU is available."""
    if sys.platform != "darwin":
        return False
    try:
        from ._backends.mps.runtime import is_available as _is_available
        return _is_available()
    except Exception:
        return False


def synchronize():
    """Wait for all MPS operations to complete."""
    from ._backends.mps.runtime import get_runtime
    get_runtime().synchronize()


def current_device():
    """Return the current MPS device index (always 0)."""
    return 0


def device_count():
    """Return the number of MPS devices (0 or 1)."""
    return 1 if is_available() else 0


def get_device_name(device=None):
    """Return the Metal device name."""
    if not is_available():
        return ""
    from ._backends.mps.runtime import get_runtime
    return get_runtime().device_name()


def get_device_properties(device=None):
    """Return a dict of Metal device properties."""
    if not is_available():
        return {}
    from ._backends.mps.runtime import get_runtime
    rt = get_runtime()
    return {
        "name": rt.device_name(),
        "type": "mps",
    }


def empty_cache():
    """No-op for MPS (shared memory, no separate cache to clear)."""


def memory_allocated(device=None):
    """Not tracked for MPS shared memory — returns 0."""
    return 0


# --- RNG ---

_default_generator = None


def _get_default_generator():
    """Get or create the default MPS generator."""
    global _default_generator
    if _default_generator is None:
        from ._random import Generator
        _default_generator = Generator('mps')
    return _default_generator


def manual_seed(seed: int):
    """Set the seed for generating random numbers on the MPS device."""
    gen = _get_default_generator()
    gen.manual_seed(seed)


def manual_seed_all(seed: int):
    """Set the seed for all MPS devices (only one exists)."""
    manual_seed(seed)


__all__ = [
    "is_available",
    "synchronize",
    "current_device",
    "device_count",
    "get_device_name",
    "get_device_properties",
    "empty_cache",
    "memory_allocated",
    "manual_seed",
    "manual_seed_all",
    "_get_default_generator",
]
