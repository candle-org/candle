"""Version counter for tensor mutation tracking.

This module provides ``VersionCounter`` for contexts that do not use
``TensorImpl`` as a base. The ``_VersionCounterProxy`` used by
``TensorImpl`` is provided by the compiled ``_tensor_impl`` extension.
"""


class VersionCounter:
    """Standalone version counter (used when TensorImpl is not the base)."""
    __slots__ = ("value",)

    def __init__(self, value=0):
        self.value = int(value)

    def bump(self):
        self.value += 1
