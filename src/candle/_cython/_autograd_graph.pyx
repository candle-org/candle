# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython-owned autograd graph helpers."""

import contextlib
from candle._cython._hooks_state import get_stack


cdef class GradientEdge:
    cdef public object node
    cdef public object output_nr

    def __init__(self, node, output_nr):
        self.node = node
        self.output_nr = output_nr


def get_gradient_edge(tensor):
    return GradientEdge(tensor.grad_fn, tensor.output_nr)


def current_saved_tensors_hooks():
    stack = get_stack()
    if not stack:
        return None
    return stack[-1]


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
