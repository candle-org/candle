import inspect

from .._cython._autograd_function import FunctionCtx, _function_apply


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
        return _function_apply(cls, args, kwargs)


class InplaceFunction(Function):
    pass


def once_differentiable(fn):
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper
