from .._cython._autograd_graph import (  # pylint: disable=import-error,no-name-in-module
    GradientEdge,
    current_saved_tensors_hooks,
    get_gradient_edge,
    saved_tensors_hooks,
)


__all__ = ["saved_tensors_hooks", "GradientEdge", "get_gradient_edge"]


class _NodeMeta(type):
    def __instancecheck__(cls, instance):
        return hasattr(instance, "next_functions") and hasattr(instance, "name")

    def __subclasscheck__(cls, subclass):
        return hasattr(subclass, "__mro__") and any(
            hasattr(base, "next_functions") and hasattr(base, "name")
            for base in subclass.__mro__
        )


class Node(metaclass=_NodeMeta):
    """Virtual base class for autograd Nodes.

    Mirrors torch.autograd.graph.Node behavior where `isinstance` and
    `issubclass` are True for autograd nodes, but Node is not in the
    concrete class MRO.
    """


__all__.append("Node")
