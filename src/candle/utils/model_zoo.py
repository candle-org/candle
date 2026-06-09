"""candle.utils.model_zoo — compatibility stub for torch.utils.model_zoo.

Some third-party libraries (for example older torchvision versions) import::

    from torch.utils.model_zoo import load_url

This module provides that symbol as an alias for
:func:`candle.hub.load_state_dict_from_url`.
"""

from candle.hub import load_state_dict_from_url as load_url


class _TqdmFallback:
    """Small tqdm-compatible fallback used by torchvision download helpers."""

    def __init__(self, iterable=None, *args, **kwargs):
        del args, kwargs
        self.iterable = iterable
        self.n = 0

    def __iter__(self):
        if self.iterable is None:
            return iter(())
        return iter(self.iterable)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        self.close()
        return False

    def update(self, n=1):
        self.n += n

    def close(self):
        return None


def tqdm(*args, **kwargs):
    """Return a tqdm progress object when available, otherwise a no-op shim."""
    try:
        from tqdm.auto import tqdm as _real_tqdm  # pylint: disable=import-outside-toplevel
    except Exception:  # pragma: no cover - exercised when tqdm is absent
        return _TqdmFallback(*args, **kwargs)
    return _real_tqdm(*args, **kwargs)


__all__ = ["load_url", "tqdm"]
