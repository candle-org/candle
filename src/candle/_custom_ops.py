"""Minimal torch._custom_ops compatibility shim.

Provides the small surface torchvision imports at module import time.
This stub does not perform real custom-op registration.
"""


class _CustomOpContext:
    def create_unbacked_symint(self):
        return 0


_CTX = _CustomOpContext()


def get_ctx():
    return _CTX


def register(*_args, **_kwargs):
    def decorator(fn):
        return fn

    return decorator


__all__ = ["get_ctx", "register"]
