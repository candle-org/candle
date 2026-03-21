# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython-owned autograd Function runtime.

Provides FunctionCtx (the ctx object passed to forward/backward),
and the core Function.apply() execution path.

FunctionMeta (the metaclass that detects old-style vs new-style forward)
stays in Python because inspect.signature() interop is simpler there.
"""


cdef class FunctionCtx:
    """Context object passed to Function.forward() for saving state needed by backward()."""
    cdef public object _to_save
    cdef public list _saved_tensors
    cdef public tuple _needs_input_grad
    cdef public set _non_differentiable
    cdef public set _dirty
    cdef public bint _materialize_grads
    cdef dict __dict__

    def __init__(self):
        self._to_save = None
        self._saved_tensors = []
        self._needs_input_grad = ()
        self._non_differentiable = set()
        self._dirty = set()
        self._materialize_grads = True

    def save_for_backward(self, *tensors):
        self._to_save = tensors

    @property
    def saved_tensors(self):
        cdef object saved
        for saved in self._saved_tensors:
            if getattr(saved, "_released", False):
                raise RuntimeError(
                    "Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). "
                    "Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). "
                    "Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward."
                )
        return tuple(saved.materialize() for saved in self._saved_tensors)

    @property
    def needs_input_grad(self):
        return self._needs_input_grad

    def mark_dirty(self, *tensors):
        self._dirty = {id(t) for t in tensors}

    def mark_non_differentiable(self, *tensors):
        self._non_differentiable = {id(t) for t in tensors}

    def set_materialize_grads(self, value):
        self._materialize_grads = bool(value)


def _function_apply(cls, args, kwargs):
    """Core apply() logic extracted for Cython acceleration.

    Called from Function.apply(cls, *args, **kwargs) in the Python layer.
    """
    from candle._tensor import Tensor
    from candle.autograd.grad_mode import is_grad_enabled
    from candle._cython._autograd_node import Node, InputMetadata
    from candle.autograd.anomaly_mode import annotate_node_creation

    cdef tuple needs_input_grad = tuple(
        isinstance(a, Tensor) and a.requires_grad for a in args
    )
    cdef bint any_grad_needed = any(needs_input_grad) and is_grad_enabled()

    # Create context
    cdef FunctionCtx ctx = FunctionCtx()
    ctx._needs_input_grad = needs_input_grad

    # Execute forward
    cdef object output
    if cls._new_style:
        output = cls.forward(*args, **kwargs)
        cls.setup_context(ctx, args, output)
    else:
        output = cls.forward(ctx, *args, **kwargs)

    if not any_grad_needed:
        return output

    # Build autograd graph
    cdef list input_tensors = [a for a in args if isinstance(a, Tensor) and a.requires_grad]

    # Capture materialize flag
    cdef bint materialize = ctx._materialize_grads

    def _backward(grad):
        if materialize and grad is None:
            from candle._functional import zeros_like
            grad = zeros_like(output)
        return cls.backward(ctx, grad)

    # Create Node
    cdef object node = Node(_backward, input_tensors, name=f"{cls.__name__}Backward")
    annotate_node_creation(node)
    node._input_metadata = [InputMetadata(t) for t in input_tensors]

    # Wire saved tensors through Node
    if ctx._to_save is not None:
        node.save_for_backward(*ctx._to_save)
        ctx._saved_tensors = node._saved_tensors_list

    # Set grad_fn on outputs
    cdef set non_diff = ctx._non_differentiable
    cdef object o
    cdef object view_meta

    if isinstance(output, Tensor):
        _mark_output(output, non_diff, node, Tensor)
    elif isinstance(output, tuple):
        for o in output:
            _mark_output(o, non_diff, node, Tensor)

    return output


cdef void _mark_output(object o, set non_diff, object node, object Tensor):
    """Mark a single output tensor with grad_fn / non-differentiable."""
    cdef object view_meta
    if not isinstance(o, Tensor):
        return
    if id(o) not in non_diff:
        o.grad_fn = node
        o.requires_grad = True
        if o._is_view():
            view_meta = dict(getattr(o, "_view_meta", None) or {})
            view_meta["creation_kind"] = "custom_function"
            o._view_meta = view_meta
    else:
        o.grad_fn = None
        o.requires_grad = False
