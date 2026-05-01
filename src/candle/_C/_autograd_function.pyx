# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython-owned autograd Function runtime.

Provides FunctionCtx (the ctx object passed to forward/backward),
and the core Function.apply() execution path.

FunctionMeta detects old-style vs new-style forward declarations at subclass creation time.
"""



class FunctionMeta(type):
    """Metaclass that detects old-style (ctx as first param) vs new-style forward."""

    def __init__(cls, name, bases, attrs):
        import inspect

        super().__init__(name, bases, attrs)
        if "forward" in attrs:
            sig = inspect.signature(attrs["forward"])
            params = list(sig.parameters.keys())
            cls._new_style = not (params and params[0] == "ctx")
        else:
            cls._new_style = False

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
        self._dirty.update(id(t) for t in tensors)

    def mark_non_differentiable(self, *tensors):
        self._non_differentiable = {id(t) for t in tensors}

    def set_materialize_grads(self, value):
        self._materialize_grads = bool(value)


def _function_apply(cls, args, kwargs):
    """Core apply() logic extracted for Cython acceleration.

    Called from Function.apply(cls, *args, **kwargs) in the Python layer.
    """
    import weakref

    from candle._tensor import Tensor
    from candle.autograd.grad_mode import is_grad_enabled
    from candle._C._autograd_node import Node, InputMetadata
    from candle.autograd.anomaly_mode import annotate_node_creation

    cdef tuple needs_input_grad = tuple(
        isinstance(a, Tensor) and a.requires_grad for a in args
    )
    cdef bint any_grad_needed = any(needs_input_grad) and is_grad_enabled()

    cdef FunctionCtx ctx = FunctionCtx()
    ctx._needs_input_grad = needs_input_grad

    cdef object output
    if cls._new_style:
        output = cls.forward(*args, **kwargs)
        cls.setup_context(ctx, args, output)
    else:
        output = cls.forward(ctx, *args, **kwargs)

    cdef set dirty = ctx._dirty
    cdef object dirty_obj
    cdef set seen_dirty_ids
    cdef object dirty_id
    if dirty:
        seen_dirty_ids = set()
        for dirty_obj in args:
            if isinstance(dirty_obj, Tensor) and id(dirty_obj) in dirty:
                dirty_id = id(dirty_obj)
                if dirty_id in seen_dirty_ids:
                    continue
                dirty_obj._bump_version()
                seen_dirty_ids.add(dirty_id)
        for dirty_obj in kwargs.values():
            if isinstance(dirty_obj, Tensor) and id(dirty_obj) in dirty:
                dirty_id = id(dirty_obj)
                if dirty_id in seen_dirty_ids:
                    continue
                dirty_obj._bump_version()
                seen_dirty_ids.add(dirty_id)

    if not any_grad_needed:
        return output

    cdef list input_tensors = [a for a in args if isinstance(a, Tensor) and a.requires_grad]
    cdef bint materialize = ctx._materialize_grads
    cdef set non_diff = ctx._non_differentiable
    cdef object o
    cdef object node
    cdef object node_holder
    cdef object first_saved_list = None
    cdef tuple outputs
    cdef list differentiable_outputs = []
    cdef int differentiable_count
    cdef int diff_index

    if isinstance(output, Tensor):
        def _backward(grad):
            if materialize and grad is None:
                from candle._functional import zeros_like
                grad = zeros_like(output)
            return cls.backward(ctx, grad)

        node = Node(_backward, input_tensors, name=f"{cls.__name__}Backward")
        annotate_node_creation(node)
        node._input_metadata = [InputMetadata(t) for t in input_tensors]

        if ctx._to_save is not None:
            node.save_for_backward(*ctx._to_save)
            ctx._saved_tensors = node._saved_tensors_list

        _mark_output(output, non_diff, node, Tensor, 0)
        return output

    if isinstance(output, tuple):
        outputs = output
        for o in outputs:
            if isinstance(o, Tensor) and id(o) not in non_diff:
                differentiable_outputs.append(o)

        differentiable_count = len(differentiable_outputs)
        for diff_index, o in enumerate(differentiable_outputs):
            node_holder = {}

            def _backward(grad, _output=o, _diff_index=diff_index, _diff_outputs=tuple(differentiable_outputs), _node_holder=node_holder):
                cdef list backward_grads
                cdef int i
                cdef object _node
                if materialize and grad is None:
                    from candle._functional import zeros_like
                    grad = zeros_like(_output)
                backward_grads = []
                for i, _out in enumerate(_diff_outputs):
                    if i == _diff_index:
                        backward_grads.append(grad)
                    elif materialize:
                        from candle._functional import zeros_like
                        backward_grads.append(zeros_like(_out))
                    else:
                        backward_grads.append(None)
                if ctx._to_save is not None:
                    _node = _node_holder["node"]
                    ctx._saved_tensors = _node._saved_tensors_list
                return cls.backward(ctx, *backward_grads)

            node = Node(_backward, input_tensors, name=f"{cls.__name__}Backward")
            annotate_node_creation(node)
            node._input_metadata = [InputMetadata(t) for t in input_tensors]

            if ctx._to_save is not None:
                node.save_for_backward(*ctx._to_save)
                if first_saved_list is None:
                    first_saved_list = node._saved_tensors_list

            node_holder["node"] = weakref.proxy(node)
            _mark_output(o, non_diff, node, Tensor, diff_index)

        for o in outputs:
            if not (isinstance(o, Tensor) and id(o) not in non_diff):
                _mark_output(o, non_diff, None, Tensor, 0)

        if first_saved_list is not None:
            ctx._saved_tensors = first_saved_list
        return output

    return output


cdef void _mark_output(object o, set non_diff, object node, object Tensor, int output_nr=0):
    """Mark a single output tensor with grad_fn / non-differentiable."""
    cdef object view_meta
    if not isinstance(o, Tensor):
        return
    if id(o) not in non_diff:
        o.grad_fn = node
        o.requires_grad = True
        o._output_nr = output_nr
        if o._is_view():
            view_meta = dict(getattr(o, "_view_meta", None) or {})
            view_meta["creation_kind"] = "custom_function"
            o._view_meta = view_meta
    else:
        o.grad_fn = None
        o.requires_grad = False
        o._output_nr = 0
