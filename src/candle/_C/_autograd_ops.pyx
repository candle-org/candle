# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython-accelerated autograd wrapper factory functions.

These factory functions create the forward wrappers that build the autograd
graph (Node, save_for_backward, etc.) for each dispatched op.  Moving them
into Cython reduces per-call overhead on the hot dispatch path.

The *backward implementations* stay in Python
(``candle._backends.autograd``).  They are passed in as ``backward_impl``
arguments to the factory functions here.
"""

import weakref
from contextlib import nullcontext


def _strip_autograd_keys(keyset):
    """Remove all autograd-related dispatch keys from *keyset*."""
    if keyset is None:
        return None
    from candle._dispatch.keys import DispatchKey
    return keyset.without(
        {
            DispatchKey.Autograd,
            DispatchKey.AutogradOther,
            DispatchKey.AutogradCPU,
            DispatchKey.AutogradNPU,
            DispatchKey.AutogradCUDA,
            DispatchKey.AutogradXPU,
            DispatchKey.AutogradMeta,
            DispatchKey.PrivateUse3,
        }
    )


def _grad_context(_keyset=None):
    """Return a no_grad context unless create_graph mode is active."""
    from candle.autograd.grad_mode import no_grad
    from candle._C._autograd_engine import is_create_graph_enabled

    if is_create_graph_enabled():
        return nullcontext()
    return no_grad()


def _backward_dispatch_keyset(raw_keyset, autograd_keyset):
    """Choose the dispatch keyset for a backward call."""
    from candle._C._autograd_engine import is_create_graph_enabled

    if is_create_graph_enabled() and autograd_keyset is not None:
        return autograd_keyset
    return raw_keyset


def _autograd_unary_passthrough(name):
    """Wrapper factory for ops that just pass through to device and record grad_fn."""
    from candle.autograd.grad_mode import GradMode
    from candle.autograd.anomaly_mode import annotate_node_creation
    from candle._dispatch.dispatcher import current_dispatch_keyset, redispatch
    from candle._C._autograd_node import Node

    def wrapper(a, *args, **kwargs):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, *args, **kwargs)
        if GradMode.enabled and getattr(a, "requires_grad", False):
            node_holder = {}

            def _backward(grad):
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return (redispatch("to", backward_keyset, grad, a.device, non_blocking=False),)

            node = Node(_backward, (a,), name=f"{name.capitalize()}Backward0")
            annotate_node_creation(node)
            node_holder["node"] = weakref.proxy(node)
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _autograd_binary(name, backward_impl, *, save_inputs=True):
    """Wrapper factory for binary ops (two tensor inputs, no extra args)."""
    from candle.autograd.grad_mode import GradMode
    from candle.autograd.anomaly_mode import annotate_node_creation
    from candle._dispatch.dispatcher import current_dispatch_keyset, redispatch
    from candle._C._autograd_node import Node

    def wrapper(a, b):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, b)
        a_requires_grad = getattr(a, "requires_grad", False)
        b_requires_grad = getattr(b, "requires_grad", False)
        if GradMode.enabled and (a_requires_grad or b_requires_grad):
            node_holder = {}

            def _backward(grad):
                if save_inputs:
                    saved_a, saved_b = node_holder["node"].saved_tensors()
                else:
                    saved_a, saved_b = a, b
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return backward_impl(grad, a, b, saved_a, saved_b, backward_keyset)

            node = Node(_backward, (a, b), name=f"{name.capitalize()}Backward0")
            annotate_node_creation(node)
            node_holder["node"] = weakref.proxy(node)
            if save_inputs:
                node.save_for_backward(a, b)
                node._saved_fields["self"] = node._saved_tensors_list[0]
                node._saved_fields["other"] = node._saved_tensors_list[1]
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _autograd_binary_args(name, backward_impl, *, save_inputs=True):
    """Wrapper factory for binary ops with extra positional/keyword args."""
    from candle.autograd.grad_mode import GradMode
    from candle.autograd.anomaly_mode import annotate_node_creation
    from candle._dispatch.dispatcher import current_dispatch_keyset, redispatch
    from candle._C._autograd_node import Node

    def wrapper(a, b, *args, **kwargs):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, b, *args, **kwargs)
        a_requires_grad = getattr(a, "requires_grad", False)
        b_requires_grad = getattr(b, "requires_grad", False)
        if GradMode.enabled and (a_requires_grad or b_requires_grad):
            node_holder = {}

            def _backward(grad):
                if save_inputs:
                    saved_a, saved_b = node_holder["node"].saved_tensors()
                else:
                    saved_a, saved_b = a, b
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return backward_impl(grad, a, b, saved_a, saved_b, backward_keyset, args, kwargs)

            node = Node(_backward, (a, b), name=f"{name.capitalize()}Backward0")
            annotate_node_creation(node)
            node_holder["node"] = weakref.proxy(node)
            if save_inputs:
                node.save_for_backward(a, b)
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _autograd_unary_args(name, backward_impl, *, cpu_only=False, save_input=True):
    """Wrapper factory for unary ops with extra positional/keyword args."""
    from candle.autograd.grad_mode import GradMode
    from candle.autograd.anomaly_mode import annotate_node_creation
    from candle._dispatch.dispatcher import current_dispatch_keyset, redispatch
    from candle._C._autograd_node import Node

    def wrapper(a, *args, **kwargs):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, *args, **kwargs)
        if cpu_only and a.device.type != "cpu":
            return out
        if GradMode.enabled and a.requires_grad:
            node_holder = {}

            def _backward(grad):
                if save_input:
                    saved_a = node_holder["node"].saved_tensors()[0]
                else:
                    saved_a = a
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                return backward_impl(grad, a, saved_a, backward_keyset, args, kwargs)

            node = Node(_backward, (a,), name=f"{name.capitalize()}Backward0")
            annotate_node_creation(node)
            node_holder["node"] = weakref.proxy(node)
            if save_input:
                node.save_for_backward(a)
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper


def _norm_extract_weight_bias(args, kwargs):
    """Extract weight and bias tensors from norm op args."""
    from candle._tensor import Tensor
    weight = args[1] if len(args) > 1 else kwargs.get("weight", None)
    bias = args[2] if len(args) > 2 else kwargs.get("bias", None)
    if weight is not None and not isinstance(weight, Tensor):
        weight = None
    if bias is not None and not isinstance(bias, Tensor):
        bias = None
    return weight, bias


def _autograd_norm(name, backward_impl):
    """Wrapper factory for normalization ops (layer_norm, batch_norm, rms_norm).

    Like ``_autograd_unary_args`` but also captures ``_backward_data`` from the
    forward output and passes it to *backward_impl* as the 5th positional arg.
    This allows NPU backward kernels to access saved intermediate data (mean,
    rstd, etc.) that the NPU forward op attached to its output tensor.

    Also tracks weight and bias as Node inputs so their gradients propagate.
    """
    from candle.autograd.grad_mode import GradMode
    from candle.autograd.anomaly_mode import annotate_node_creation
    from candle._dispatch.dispatcher import current_dispatch_keyset, redispatch
    from candle._C._autograd_node import Node

    def wrapper(a, *args, **kwargs):
        active_keyset = current_dispatch_keyset()
        raw_keyset = _strip_autograd_keys(active_keyset)
        out = redispatch(name, raw_keyset, a, *args, **kwargs)

        weight, bias = _norm_extract_weight_bias(args, kwargs)
        any_requires_grad = a.requires_grad
        if weight is not None and getattr(weight, "requires_grad", False):
            any_requires_grad = True
        if bias is not None and getattr(bias, "requires_grad", False):
            any_requires_grad = True

        if GradMode.enabled and any_requires_grad:
            backward_data = getattr(out, "_backward_data", None)
            node_holder = {}

            # Build inputs list: input + optional weight + optional bias
            inputs = [a]
            if weight is not None and getattr(weight, "requires_grad", False):
                inputs.append(weight)
            if bias is not None and getattr(bias, "requires_grad", False):
                inputs.append(bias)

            def _backward(grad):
                saved_a = node_holder["node"].saved_tensors()[0]
                backward_keyset = _backward_dispatch_keyset(raw_keyset, active_keyset)
                all_grads = backward_impl(grad, a, saved_a, backward_keyset, args, kwargs, backward_data)
                # all_grads = (grad_input, grad_weight, grad_bias)
                # Map to the inputs list we built
                result = []
                result.append(all_grads[0])  # grad_input
                idx = 1
                if weight is not None and getattr(weight, "requires_grad", False):
                    result.append(all_grads[idx] if idx < len(all_grads) else None)
                    idx += 1
                if bias is not None and getattr(bias, "requires_grad", False):
                    result.append(all_grads[idx] if idx < len(all_grads) else None)
                return tuple(result)

            node = Node(_backward, tuple(inputs), name=f"{name.capitalize()}Backward0")
            annotate_node_creation(node)
            node_holder["node"] = weakref.proxy(node)
            node.save_for_backward(a)
            out.grad_fn = node
            out.requires_grad = True
        return out

    return wrapper
