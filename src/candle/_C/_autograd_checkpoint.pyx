# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython-owned checkpoint autograd runtime helpers."""

from candle._C._autograd_node import Node  # pylint: disable=import-error,no-name-in-module

_RELEASED_SAVED_TENSORS_ERROR = (
    "Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). "
    "Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). "
    "Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward."
)


class _CheckpointNode(Node):
    def __init__(self, backward, inputs, recompute_saved_result):
        super().__init__(backward, inputs)
        self._recompute_saved_result = recompute_saved_result
        self._checkpoint_released = False

    def release_saved_tensors(self):
        super().release_saved_tensors()
        self._checkpoint_released = True

    def __getattr__(self, name):
        if name == "_saved_result":
            if self._checkpoint_released:
                raise RuntimeError(_RELEASED_SAVED_TENSORS_ERROR)
            return self._recompute_saved_result()
        return super().__getattr__(name)
