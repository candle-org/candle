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
