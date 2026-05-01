# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython-owned autograd graph helpers."""

import contextlib
from candle._C._hooks_state import get_stack


cdef class GradientEdge:
    cdef public object node
    cdef public object output_nr

    def __init__(self, node, output_nr):
        self.node = node
        self.output_nr = output_nr


def get_gradient_edge(tensor):
    return GradientEdge(tensor.grad_fn, tensor.output_nr)


class _NodeMeta(type):
    def __instancecheck__(cls, instance):
        return hasattr(instance, "next_functions") and hasattr(instance, "name")

    def __subclasscheck__(cls, subclass):
        if not isinstance(subclass, type):
            raise TypeError("issubclass() arg 1 must be a class")
        return any(
            hasattr(base, "next_functions") and hasattr(base, "name")
            for base in subclass.__mro__
        )


class Node(metaclass=_NodeMeta):
    """Virtual base class for autograd Nodes.

    Mirrors torch.autograd.graph.Node behavior where `isinstance` and
    `issubclass` are True for autograd nodes, but Node is not in the
    concrete class MRO.
    """

def current_saved_tensors_hooks():
    stack = get_stack()
    if not stack:
        return None
    return stack[len(stack) - 1]


@contextlib.contextmanager
def saved_tensors_hooks(pack_hook, unpack_hook):
    if not callable(pack_hook) or not callable(unpack_hook):
        raise TypeError("saved_tensors_hooks expects callable pack and unpack")
    stack = get_stack()
    stack.append((pack_hook, unpack_hook))
    try:
        yield
    finally:
        stack.pop()
