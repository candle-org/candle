"""Compatibility shim for torch.version.

Exposes the attributes that torchvision and other downstream
libraries check at import time.
"""
from . import __version__

__all__ = ["__version__", "cuda", "git_version", "hip", "debug"]

cuda = None          # Candle does not ship CUDA bindings
git_version = ""
hip = None
debug = False
