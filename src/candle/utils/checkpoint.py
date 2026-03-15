from enum import Enum

from ..autograd.grad_mode import no_grad, enable_grad
from ..autograd.engine import _run_backward
from ..autograd.node import Node


class CheckpointPolicy(str, Enum):
    DEFAULT = "DEFAULT"


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
                raise RuntimeError(
                    "Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). "
                    "Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). "
                    "Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward."
                )
            return self._recompute_saved_result()
        return super().__getattr__(name)


def create_selective_checkpoint_contexts(*args, **kwargs):  # noqa: ARG001
    def _ctx():
        return None

    return _ctx, _ctx


def checkpoint(function, *args, use_reentrant=True, preserve_rng_state=True, **kwargs):
    """Checkpoint a function to trade compute for memory.

    Runs function(*args) without saving intermediates in forward,
    then recomputes them during backward.

    Args:
        function: The function to checkpoint
        *args: Arguments to pass to function
        use_reentrant: Whether to use reentrant checkpoint (for API compatibility)
        preserve_rng_state: Whether to preserve RNG state (not implemented, for API compatibility)
        **kwargs: Keyword arguments to pass to function
    """
    # Separate tensor and non-tensor args
    tensor_inputs = []
    tensor_indices = []
    for i, arg in enumerate(args):
        if hasattr(arg, 'requires_grad') and hasattr(arg, 'grad_fn'):
            tensor_inputs.append(arg)
            tensor_indices.append(i)

    # Forward: run without saving intermediates
    with no_grad():
        outputs = function(*args, **kwargs)

    # If no tensor input requires grad, just return
    if not any(t.requires_grad for t in tensor_inputs):
        return outputs

    is_tuple = isinstance(outputs, tuple)
    if not is_tuple:
        outputs = (outputs,)

    # Detach inputs for recomputation
    def make_recompute_inputs():
        new_args = list(args)
        detached = []
        for i, idx in enumerate(tensor_indices):
            d = tensor_inputs[i].detach()
            d.requires_grad_(tensor_inputs[i].requires_grad)
            new_args[idx] = d
            detached.append(d)
        return new_args, detached

    def _checkpoint_backward(grad):
        new_args, detached = make_recompute_inputs()
        with enable_grad():
            recomputed = function(*new_args, **kwargs)
        if not isinstance(recomputed, tuple):
            recomputed = (recomputed,)

        # Only backward through outputs that got gradients
        out_with_grad = []
        grad_outputs = []
        for r, o in zip(recomputed, outputs):
            out_with_grad.append(r)
            grad_outputs.append(grad if len(recomputed) == 1 else None)

        _run_backward(
            tuple(out_with_grad), tuple(grad_outputs),
            retain_graph=False, create_graph=False,
            accumulate_grad=True, inputs=None,
            allow_unused=True,
        )
        # Retrieve grads from the detached input copies
        all_grads = [d.grad for d in detached]
        return tuple(all_grads)

    def _recompute_saved_result():
        new_args, _ = make_recompute_inputs()
        with enable_grad():
            recomputed = function(*new_args, **kwargs)
        if isinstance(recomputed, tuple):
            return recomputed[0]
        return recomputed

    node = _CheckpointNode(_checkpoint_backward, tuple(tensor_inputs), _recompute_saved_result)
    # Attach grad_fn to outputs and mark as requiring grad
    for out in outputs:
        if hasattr(out, 'grad_fn'):
            out.grad_fn = node
            out.requires_grad = True

    if is_tuple:
        return outputs
    return outputs[0]


def checkpoint_sequential(functions, segments, input, **kwargs):
    """Checkpoint a sequential model by splitting into segments."""
    funcs = list(functions)
    segment_size = (len(funcs) + segments - 1) // segments

    def run_segment(start, end, inp):
        def segment_fn(x):
            for f in funcs[start:end]:
                x = f(x)
            return x
        return checkpoint(segment_fn, inp, **kwargs)

    x = input
    for start in range(0, len(funcs), segment_size):
        end = min(start + segment_size, len(funcs))
        x = run_segment(start, end, x)
    return x
