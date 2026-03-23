"""Stateful protocol for torch.distributed.checkpoint compatibility.

Mirrors ``torch.distributed.checkpoint.stateful.Stateful``.

Objects that implement this protocol can be passed as values inside the
state_dict mapping handed to ``dcp.save`` / ``dcp.load``.  The DCP
save/load machinery will call ``state_dict()`` / ``load_state_dict()``
on these objects rather than treating them as opaque bytes.
"""

from abc import ABC, abstractmethod


class Stateful(ABC):
    """Abstract base class for objects that participate in DCP save/load.

    Subclasses must implement ``state_dict`` and ``load_state_dict``.
    This mirrors ``torch.distributed.checkpoint.stateful.Stateful``.
    """

    @abstractmethod
    def state_dict(self):
        """Return a serialisable dict representing this object's state."""

    @abstractmethod
    def load_state_dict(self, state_dict):
        """Restore state from *state_dict* (as returned by ``state_dict()``)."""
