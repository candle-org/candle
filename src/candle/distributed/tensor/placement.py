"""Tensor placement types for distributed tensors."""


class Placement:
    """Base class for tensor placement strategies."""

    def __eq__(self, other):
        return type(self) is type(other) and self.__dict__ == other.__dict__

    def __hash__(self):
        return hash((type(self), tuple(sorted(self.__dict__.items()))))

    def __repr__(self):
        return f"{type(self).__name__}()"


class Shard(Placement):
    """Tensor is sharded along a dimension across the mesh."""

    def __init__(self, dim: int = 0):
        self.dim = dim

    def __repr__(self):
        return f"Shard(dim={self.dim})"

    def __hash__(self):
        return hash((type(self), self.dim))


class Replicate(Placement):
    """Tensor is replicated across all ranks in the mesh."""



class Partial(Placement):
    """Tensor has pending reduction (e.g., gradient before reduce-scatter)."""

    def __init__(self, reduce_op: str = "sum"):
        self.reduce_op = reduce_op

    def __repr__(self):
        return f"Partial(reduce_op={self.reduce_op!r})"

    def __hash__(self):
        return hash((type(self), self.reduce_op))
