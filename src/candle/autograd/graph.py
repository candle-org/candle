import contextlib
import threading


_STATE = threading.local()


class GradientEdge:
    def __init__(self, node, output_nr):
        self.node = node
        self.output_nr = output_nr


def get_gradient_edge(tensor):
    return GradientEdge(tensor.grad_fn, tensor.output_nr)


def _stack():
    stack = getattr(_STATE, "hooks", None)
    if stack is None:
        stack = []
        _STATE.hooks = stack
    return stack


def current_saved_tensors_hooks():
    stack = _stack()
    if not stack:
        return None
    return stack[-1]


@contextlib.contextmanager
def saved_tensors_hooks(pack_hook, unpack_hook):
    if not callable(pack_hook) or not callable(unpack_hook):
        raise TypeError("saved_tensors_hooks expects callable pack and unpack")
    stack = _stack()
    stack.append((pack_hook, unpack_hook))
    try:
        yield
    finally:
        stack.pop()


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
