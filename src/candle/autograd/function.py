from .._C._autograd_function import (  # pylint: disable=import-error,no-name-in-module
    FunctionCtx,
    FunctionMeta,
    _apply_custom_op_autograd,
    _function_apply,
    once_differentiable,
)


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
