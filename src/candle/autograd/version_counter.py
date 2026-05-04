"""Public shell for the standalone autograd ``VersionCounter``.

Mirrors torch.autograd's exposure of ``VersionCounter``; the runtime owner
is ``candle._C._tensor_impl`` (which already hosts ``_VersionCounterProxy``
for tensors backed by ``TensorImpl``).  This module is only a thin re-export
so that ``candle.autograd.version_counter.VersionCounter`` keeps working.
"""

from .._C._tensor_impl import VersionCounter  # pylint: disable=import-error,no-name-in-module


__all__ = ["VersionCounter"]
