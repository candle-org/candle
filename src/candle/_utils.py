"""Minimal torch._utils compatibility layer for candle storage."""


def _element_size(dtype):
    return dtype.itemsize


def _to(storage, device, non_blocking=False):
    """Move storage to the given device."""
    if storage.device == device:
        return storage
    if device.type == "cpu":
        return storage.cpu()
    if device.type == "cuda":
        return storage.cuda(device.index, non_blocking)
    raise RuntimeError(f"Unsupported device: {device}")


def _type(storage, dtype, non_blocking=False):
    """Convert storage to a different dtype."""
    if dtype is None:
        from .storage import TypedStorage
        if isinstance(storage, TypedStorage):
            legacy = storage._get_legacy_storage_class()
            if legacy is not None:
                return f"{legacy.__module__}.{legacy.__name__}"
        return f"{type(storage).__module__}.{type(storage).__name__}"

    if isinstance(storage, _StorageBase):
        return storage._to(dtype)
    return storage._to(dtype)


# Lazy import to avoid circular dependency
def __getattr__(name):
    if name == "_StorageBase":
        from .storage import _StorageBase
        return _StorageBase
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
