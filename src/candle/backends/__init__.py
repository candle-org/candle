"""torch.backends compatibility stubs.

Provide minimal module structure used by PyTorch tests.
"""

from . import quantized  # noqa: F401
from . import cuda  # noqa: F401
from . import mkldnn  # noqa: F401

__all__ = [
    "quantized",
    "cuda",
    "mkldnn",
]
