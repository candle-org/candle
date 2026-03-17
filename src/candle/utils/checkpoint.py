"""Minimal checkpointing utilities aligned with PyTorch semantics used in tests."""
from __future__ import annotations

import contextlib
from enum import Enum
import warnings

from ..autograd.grad_mode import no_grad, enable_grad
from ..autograd.engine import _run_backward
from ..autograd.node import Node
from . import checkpoint_state


class CheckpointError(RuntimeError):
    pass


_checkpoint_debug_enabled = None


@contextlib.contextmanager
def set_checkpoint_debug_enabled(enabled):
    """Context manager controlling checkpoint debug behavior."""
    global _checkpoint_debug_enabled
    try:
        prev = _checkpoint_debug_enabled
        _checkpoint_debug_enabled = enabled
        yield
    finally:
        _checkpoint_debug_enabled = prev


_DEFAULT_DETERMINISM_MODE = "default"


def _default_meta_extractor(x):
    return {"shape": x.shape, "dtype": x.dtype, "device": x.device}


_allowed_determinism_checks_to_fns = {
    _DEFAULT_DETERMINISM_MODE: _default_meta_extractor,
    "none": lambda _: None,
}


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


class _CheckpointEarlyStop:
    def __init__(self, enabled: bool):
        self._enabled = enabled
        self._prev = None

    def __enter__(self):
        self._prev = checkpoint_state.early_stop_enabled()
        checkpoint_state.set_early_stop_enabled(self._enabled)

    def __exit__(self, exc_type, exc, tb):
        checkpoint_state.set_early_stop_enabled(self._prev)
        return False


def set_checkpoint_early_stop(enabled: bool):
    return _CheckpointEarlyStop(enabled)


def create_selective_checkpoint_contexts(*args, **kwargs):  # noqa: ARG001
    def _ctx():
        return None

    return _ctx, _ctx


# ---------------------------------------------------------------------------
# Public checkpoint API
# ---------------------------------------------------------------------------

def _warn_if_use_reentrant_not_explicit(use_reentrant):
    if use_reentrant is None:
        warnings.warn(
            "the use_reentrant parameter should be passed explicitly", UserWarning
        )
        return True
    return use_reentrant


def checkpoint(function, *args, use_reentrant=None, preserve_rng_state=True, **kwargs):
    """Checkpoint a function to trade compute for memory."""
    use_reentrant = _warn_if_use_reentrant_not_explicit(use_reentrant)
    # Separate tensor and non-tensor args
    tensor_inputs = []
    tensor_indices = []
    for i, arg in enumerate(args):
        if hasattr(arg, "requires_grad") and hasattr(arg, "grad_fn"):
            tensor_inputs.append(arg)
            tensor_indices.append(i)

    # Forward: run without saving intermediates
    with no_grad():
        with checkpoint_state.checkpoint_context():
            outputs = function(*args, **kwargs)

    if not isinstance(outputs, tuple):
        outputs = (outputs,)

    checkpoint_state.save_expected_count(len(outputs))

    # If no tensor input requires grad, just return
    if not any(t.requires_grad for t in tensor_inputs):
        return outputs[0] if len(outputs) == 1 else outputs

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
            with checkpoint_state.checkpoint_context():
                recomputed = function(*new_args, **kwargs)
        if not isinstance(recomputed, tuple):
            recomputed = (recomputed,)

        expected = checkpoint_state.pop_expected_count()
        if expected is not None:
            if len(recomputed) < expected:
                raise RuntimeError("A different number of tensors was saved")
            if len(recomputed) > expected and not checkpoint_state.early_stop_enabled():
                raise RuntimeError("trying to save more tensors during recomputation")

        if len(recomputed) != len(outputs):
            raise RuntimeError("A different number of tensors was saved")
        for orig, new in zip(outputs, recomputed):
            if hasattr(orig, "shape") and hasattr(new, "shape"):
                if tuple(orig.shape) != tuple(new.shape):
                    raise RuntimeError("tensors have different metadata")
            if hasattr(orig, "dtype") and hasattr(new, "dtype"):
                if orig.dtype != new.dtype:
                    raise RuntimeError("tensors have different metadata")

        out_with_grad = []
        grad_outputs = []
        for r in recomputed:
            out_with_grad.append(r)
            grad_outputs.append(grad if len(recomputed) == 1 else None)

        _run_backward(
            tuple(out_with_grad), tuple(grad_outputs),
            retain_graph=False, create_graph=False,
            accumulate_grad=True, inputs=None,
            allow_unused=True,
        )
        all_grads = [d.grad for d in detached]
        return tuple(all_grads)

    def _recompute_saved_result():
        new_args, _ = make_recompute_inputs()
        with enable_grad():
            with checkpoint_state.checkpoint_context():
                recomputed = function(*new_args, **kwargs)
        if isinstance(recomputed, tuple):
            return recomputed[0]
        return recomputed

    node = _CheckpointNode(_checkpoint_backward, tuple(tensor_inputs), _recompute_saved_result)
    for out in outputs:
        if hasattr(out, "grad_fn"):
            out.grad_fn = node
            out.requires_grad = True

    return outputs[0] if len(outputs) == 1 else outputs


def checkpoint_sequential(functions, segments, input, use_reentrant=None, **kwargs):
    """Checkpoint a sequential model by splitting into segments."""
    use_reentrant = _warn_if_use_reentrant_not_explicit(use_reentrant)
    funcs = list(functions)
    segment_size = (len(funcs) + segments - 1) // segments

    def run_segment(start, end, inp):
        def segment_fn(x):
            for f in funcs[start:end]:
                x = f(x)
            return x
        return checkpoint(segment_fn, inp, use_reentrant=use_reentrant, **kwargs)

    x = input
    for start in range(0, len(funcs), segment_size):
        end = min(start + segment_size, len(funcs))
        x = run_segment(start, end, x)
    return x
