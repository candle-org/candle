"""Public shell for built-in autograd Function classes.

The runtime owner is ``candle._C._autograd_functions``; this module is only a
thin re-export so that ``candle.autograd._functions.tensor.Resize`` (the public
name PyTorch users expect) keeps working.
"""

from ..._C._autograd_functions import Resize  # pylint: disable=import-error,no-name-in-module


__all__ = ["Resize"]
