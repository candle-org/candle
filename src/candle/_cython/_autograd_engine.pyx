# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython-owned autograd backward engine and hot-path state helpers."""

from collections import deque
from contextlib import nullcontext
import threading

from candle.autograd.grad_mode import no_grad


cdef object _GRAPH_STATE = threading.local()
cdef object _ANOMALY_STATE = threading.local()


cdef list _graph_task_stack():
    cdef object stack = getattr(_GRAPH_STATE, "stack", None)
    if stack is None:
        stack = []
        _GRAPH_STATE.stack = stack
    return stack


def is_create_graph_enabled():
    cdef list stack = _graph_task_stack()
    if not stack:
        return False
    return bool(stack[len(stack) - 1].create_graph)


cdef list _anomaly_config_stack():
    cdef object stack = getattr(_ANOMALY_STATE, "config_stack", None)
    if stack is None:
        stack = []
        _ANOMALY_STATE.config_stack = stack
    return stack


cdef list _anomaly_node_stack():
    cdef object stack = getattr(_ANOMALY_STATE, "node_stack", None)
    if stack is None:
        stack = []
        _ANOMALY_STATE.node_stack = stack
    return stack


def push_anomaly_config(config):
    cdef list stack = _anomaly_config_stack()
    stack.append(config)


def pop_anomaly_config():
    cdef list stack = _anomaly_config_stack()
    return stack.pop()


def is_anomaly_enabled():
    cdef list stack = _anomaly_config_stack()
    return bool(stack and stack[len(stack) - 1].enabled)


def is_anomaly_check_nan_enabled():
    cdef list stack = _anomaly_config_stack()
    return bool(stack and stack[len(stack) - 1].enabled and stack[len(stack) - 1].check_nan)


def current_anomaly_parent():
    cdef list stack = _anomaly_node_stack()
    if not stack:
        return None
    return stack[len(stack) - 1]


def push_evaluating_node(node):
    cdef list stack = _anomaly_node_stack()
    stack.append(node)


def pop_evaluating_node():
    cdef list stack = _anomaly_node_stack()
    if not stack:
        return None
    return stack.pop()


def _grad_has_nan(grad):
    if grad is None:
        return False
    try:
        return bool(grad.isnan().any().item())
    except Exception:
        return False


cdef void _check_anomaly_nan(object node, object grads, bint check_nan):
    cdef Py_ssize_t idx
    cdef object grad
    if not check_nan:
        return
    for idx, grad in enumerate(grads):
        if _grad_has_nan(grad):
            from candle.autograd.anomaly_mode import report_anomaly
            report_anomaly(node)
            raise RuntimeError(
                f"Function '{node.name()}' returned nan values in its {idx}th output."
            )


cdef class _GraphTask:
    cdef public object dependencies
    cdef public dict received
    cdef public dict node_grads
    cdef public object ready
    cdef public bint retain_graph
    cdef public bint create_graph
    cdef public bint accumulate_grad
    cdef public dict grads_map
    cdef public object inputs
    cdef public object input_ids
    cdef public bint anomaly_enabled
    cdef public bint anomaly_check_nan

    def __init__(
        self,
        dependencies,
        *,
        retain_graph,
        create_graph,
        accumulate_grad,
        grads_map=None,
        inputs=None,
    ):
        self.dependencies = dependencies
        self.received = {}
        self.node_grads = {}
        self.ready = deque()
        self.retain_graph = bool(retain_graph)
        self.create_graph = bool(create_graph)
        self.accumulate_grad = bool(accumulate_grad)
        self.grads_map = grads_map if grads_map is not None else {}
        self.inputs = None if inputs is None else tuple(inputs)
        self.input_ids = None if inputs is None else {id(t) for t in inputs}
        self.anomaly_enabled = is_anomaly_enabled()
        self.anomaly_check_nan = is_anomaly_check_nan_enabled()

    def _grad_accum_context(self):
        if self.create_graph:
            return nullcontext()
        return no_grad()

    def _accumulate_tensor_grad(self, tensor, grad):
        cdef bint should_touch_leaf
        cdef object acc_node
        cdef bint should_accumulate_into_grad
        cdef object prev
        cdef bint preserve_reference

        should_touch_leaf = (
            self.inputs is None or tensor.grad_fn is not None or id(tensor) in self.input_ids
        )
        if not should_touch_leaf:
            return grad

        grad = _apply_hooks(tensor, grad)
        acc_node = getattr(tensor, "_accumulate_grad_node", None)
        if acc_node is not None:
            grad = acc_node.apply_prehooks(grad)
        if self.create_graph and grad is not None:
            grad.requires_grad = True
        should_accumulate_into_grad = (
            self.accumulate_grad and (self.inputs is None or id(tensor) in self.input_ids)
        )
        if should_accumulate_into_grad:
            if tensor.grad_fn is None or getattr(tensor, "_retain_grad", False):
                if tensor.grad is None:
                    tensor.grad = grad
                else:
                    preserve_reference = not self.create_graph
                    if getattr(tensor.grad, "is_sparse", False) and not getattr(grad, "is_sparse", False):
                        preserve_reference = False
                    if self.create_graph or not preserve_reference:
                        from candle._functional import add
                        with self._grad_accum_context():
                            new_grad = add(tensor.grad, grad)
                        if getattr(tensor.grad, "is_sparse", False) and getattr(grad, "is_sparse", False):
                            new_grad._is_sparse = True
                            new_grad.layout = tensor.grad.layout
                        tensor.grad = new_grad
                    else:
                        with self._grad_accum_context():
                            tensor.grad += grad
                if acc_node is not None:
                    acc_node.apply_posthooks(tensor.grad)
        else:
            prev = self.grads_map.get(tensor)
            if prev is None:
                self.grads_map[tensor] = grad
            else:
                from candle._functional import add
                with self._grad_accum_context():
                    self.grads_map[tensor] = add(prev, grad)
        return grad

    def _accumulate_node_grad(self, node, grad):
        cdef object bucket = self.node_grads.get(node)
        if bucket is None:
            bucket = []
            self.node_grads[node] = bucket
        bucket.append(grad)
        self.received[node] = self.received.get(node, 0) + 1
        if self.received[node] >= self.dependencies.get(node, 0):
            self.ready.append(node)

    def run(self):
        cdef object node
        cdef object grads
        cdef object grad
        cdef object extra
        cdef object grad_counts
        cdef object g
        cdef object t
        cdef object next_fn
        cdef object _output_nr
        cdef bint should_visit

        while self.ready:
            node = self.ready.popleft()
            grads = self.node_grads.pop(node, None)
            if not grads:
                continue
            grad = grads[0]
            if len(grads) > 1:
                from candle._functional import add
                with self._grad_accum_context():
                    for extra in grads[1:]:
                        grad = add(grad, extra)
            if self.anomaly_enabled:
                push_evaluating_node(node)
                try:
                    grads = node.backward(grad)
                except Exception:
                    from candle.autograd.anomaly_mode import report_anomaly
                    report_anomaly(node)
                    raise
                finally:
                    pop_evaluating_node()
            else:
                grads = node.backward(grad)
            if grads is None:
                grads = ()
            elif not isinstance(grads, (tuple, list)):
                grads = (grads,)
            _check_anomaly_nan(node, grads, self.anomaly_check_nan)
            grad_counts = None
            if grads:
                from candle._tensor import Tensor
                grad_counts = {}
                for g in grads:
                    if isinstance(g, Tensor):
                        grad_counts[id(g)] = grad_counts.get(id(g), 0) + 1
            for (t, g), (next_fn, _output_nr) in zip(zip(node.inputs, grads), node.next_functions):
                if g is None:
                    continue
                if grad_counts is not None and grad_counts.get(id(g), 0) > 1:
                    g = g.clone()
                should_visit = (
                    self.inputs is None or next_fn is not None or id(t) in self.input_ids
                )
                if not should_visit:
                    continue
                if not isinstance(t, Tensor):
                    if next_fn is not None and next_fn in self.dependencies:
                        self._accumulate_node_grad(next_fn, g)
                    continue
                g = self._accumulate_tensor_grad(t, g)
                if next_fn is not None and next_fn in self.dependencies:
                    self._accumulate_node_grad(next_fn, g)
            if not self.retain_graph:
                node.release_saved_tensors()
        if not self.retain_graph:
            for node in self.dependencies:
                node.release_saved_tensors()


def _apply_hooks(tensor, grad):
    if tensor._backward_hooks:
        for hook in tensor._backward_hooks.values():
            result = hook(grad)
            if result is not None:
                grad = result
    return grad


def _build_dependencies(outputs, inputs=None):
    cdef object target_input_ids = None if inputs is None else {id(inp) for inp in inputs}
    cdef set nodes = set()
    cdef list stack = [out.grad_fn for out in outputs if out.grad_fn is not None]
    cdef object reachable = {} if target_input_ids is not None else None
    cdef object node
    cdef object node_inputs
    cdef bint has_target
    cdef object fn
    cdef object _output_nr
    cdef object deps
    cdef set seen
    cdef bint changed

    while stack:
        node = stack.pop()
        if node is None:
            continue
        if node in nodes:
            continue
        nodes.add(node)
        if reachable is not None:
            node_inputs = getattr(node, "inputs", ())
            has_target = any(id(inp) in target_input_ids for inp in node_inputs)
            for fn, _output_nr in node.next_functions:
                if fn is not None and hasattr(fn, "backward"):
                    stack.append(fn)
            reachable[node] = has_target
        else:
            for fn, _output_nr in node.next_functions:
                if fn is not None and hasattr(fn, "backward"):
                    stack.append(fn)
    if reachable is not None:
        changed = True
        while changed:
            changed = False
            for node in list(nodes):
                has_target = reachable[node]
                if has_target:
                    continue
                for fn, _output_nr in node.next_functions:
                    if fn is not None and hasattr(fn, "backward") and reachable.get(fn, False):
                        reachable[node] = True
                        changed = True
                        break
        nodes = {node for node in nodes if reachable.get(node, False)}
    deps = {node: 0 for node in nodes}
    for node in nodes:
        seen = set()
        for fn, _output_nr in node.next_functions:
            if fn is None or not hasattr(fn, "backward"):
                continue
            if fn in seen:
                continue
            seen.add(fn)
            if fn in deps:
                deps[fn] += 1
    return deps


def _run_backward(
    outputs,
    grad_outputs,
    *,
    retain_graph,
    create_graph,
    accumulate_grad,
    inputs=None,
    allow_unused=False,
):
    cdef _GraphTask task = _GraphTask(
        _build_dependencies(outputs, inputs),
        retain_graph=retain_graph,
        create_graph=create_graph,
        accumulate_grad=accumulate_grad,
        inputs=inputs,
    )
    cdef list stack = _graph_task_stack()
    cdef object out
    cdef object grad
    cdef list results
    cdef object grad_val

    stack.append(task)
    try:
        for out, grad in zip(outputs, grad_outputs):
            if grad is None:
                continue
            grad = _apply_hooks(out, grad)
            if out.grad_fn is None:
                task._accumulate_tensor_grad(out, grad)
            else:
                task._accumulate_node_grad(out.grad_fn, grad)
        task.run()
    finally:
        stack.pop()

    if inputs is None:
        return None
    results = []
    for inp in inputs:
        grad_val = task.grads_map.get(inp)
        if grad_val is None and not allow_unused:
            raise RuntimeError(
                "One of the differentiated Tensors appears to not have been used in the graph."
            )
        results.append(grad_val)
    return tuple(results)


def backward(tensor, grad=None, retain_graph=False, create_graph=False, inputs=None):
    if grad is None:
        if tensor.numel() != 1:
            raise RuntimeError("grad can be implicitly created only for scalar outputs")
        grad = tensor._ones_like()
    if create_graph and not retain_graph:
        retain_graph = True
    _run_backward(
        (tensor,),
        (grad,),
        retain_graph=retain_graph,
        create_graph=create_graph,
        accumulate_grad=True,
        inputs=inputs,
        allow_unused=True,
    )


def grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False, allow_unused=False):
    cdef object outs
    cdef object ins
    if retain_graph is None:
        retain_graph = create_graph
    outs = outputs if isinstance(outputs, (tuple, list)) else (outputs,)
    ins = inputs if isinstance(inputs, (tuple, list)) else (inputs,)
    if all(out.grad_fn is None and not out.requires_grad for out in outs):
        if allow_unused:
            return tuple(None for _ in ins)
        raise RuntimeError(
            "element 0 of tensors does not require grad and does not have a grad_fn"
        )
    if grad_outputs is None:
        grad_outputs = []
        for out in outs:
            if out.numel() != 1:
                raise RuntimeError("grad can be implicitly created only for scalar outputs")
            grad_outputs.append(out._ones_like())
        grad_outputs = tuple(grad_outputs)
    else:
        grad_outputs = grad_outputs if isinstance(grad_outputs, (tuple, list)) else (grad_outputs,)
        if len(grad_outputs) != len(outs):
            raise RuntimeError("grad_outputs must be the same length as outputs")
    return _run_backward(
        outs,
        grad_outputs,
        retain_graph=retain_graph,
        create_graph=create_graph,
        accumulate_grad=False,
        inputs=ins,
        allow_unused=allow_unused,
    )
