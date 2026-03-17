"""Version counter for tensor mutation tracking.

When Cython TensorImpl is available, the version counter is inlined as
``TensorImpl._version_value`` (C int64) and accessed via a lightweight
``_VersionCounterProxy``.  This module provides the fallback
``VersionCounter`` class for use when Cython is not available, and
re-exports the proxy for compatibility.
"""


class VersionCounter:
    """Standalone version counter (used when TensorImpl is not the base)."""
    __slots__ = ("value",)

    def __init__(self, value=0):
        self.value = int(value)

    def bump(self):
        self.value += 1
