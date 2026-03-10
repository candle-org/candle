"""torch.testing — Utilities for testing PyTorch code."""

from ._testing import assert_close, assert_allclose, make_tensor
from . import _internal

__all__ = [
    "assert_close",
    "assert_allclose",
    "make_tensor",
    "_internal",
]
