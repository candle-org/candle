import inspect

from .grad_mode import is_grad_enabled
from .node import Node, InputMetadata
from .anomaly_mode import annotate_node_creation


class FunctionCtx:
    """Context object passed to Function.forward() for saving state needed by backward()."""

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
        if any(getattr(saved, "_released", False) for saved in self._saved_tensors):
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
        self._materialize_grads = value


class FunctionMeta(type):
    """Metaclass that detects old-style (ctx as first param) vs new-style forward."""

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        if "forward" in attrs:
            sig = inspect.signature(attrs["forward"])
            params = list(sig.parameters.keys())
            # Old-style: forward(ctx, ...) where first param is named 'ctx'
            # New-style: forward(...) with no 'ctx' first param
            if params and params[0] == "ctx":
                cls._new_style = False
            else:
                cls._new_style = True
        else:
            cls._new_style = False


class Function(metaclass=FunctionMeta):
    """Base class for custom autograd operations.

    Subclass this and implement static forward() and backward() methods.
    """

    @staticmethod
    def forward(ctx, *args, **kwargs):
        raise NotImplementedError("You must implement the forward function for custom autograd.Function.")

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("You must implement the backward function for custom autograd.Function.")

    @classmethod
    def apply(cls, *args, **kwargs):
        # Build needs_input_grad tuple from tensor args
        from .._tensor import Tensor
        needs_input_grad = tuple(
            isinstance(a, Tensor) and a.requires_grad for a in args
        )
        any_grad_needed = any(needs_input_grad) and is_grad_enabled()

        # Create context
        ctx = FunctionCtx()
        ctx._needs_input_grad = needs_input_grad

        # Execute forward
        if cls._new_style:
            output = cls.forward(*args, **kwargs)
            cls.setup_context(ctx, args, output)
        else:
            output = cls.forward(ctx, *args, **kwargs)

        if not any_grad_needed:
            return output

        # Build autograd graph
        # Collect input tensors that require grad
        input_tensors = [a for a in args if isinstance(a, Tensor) and a.requires_grad]

        # Create backward closure
        materialize = ctx._materialize_grads

        def _backward(grad):
            if materialize and grad is None:
                from .._functional import zeros_like
                grad = zeros_like(output)
            return cls.backward(ctx, grad)

        # Create Node
        node = Node(_backward, input_tensors, name=f"{cls.__name__}Backward")
        annotate_node_creation(node)
        node._input_metadata = [InputMetadata(t) for t in input_tensors]

        # Wire saved tensors through Node
        if ctx._to_save is not None:
            node.save_for_backward(*ctx._to_save)
            ctx._saved_tensors = node._saved_tensors_list

        # Set grad_fn on outputs
        non_diff = ctx._non_differentiable
        def _mark_output(o):
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

        if isinstance(output, Tensor):
            _mark_output(output)
        elif isinstance(output, tuple):
            for o in output:
                _mark_output(o)

        return output


class InplaceFunction(Function):
    pass


def once_differentiable(fn):
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper
