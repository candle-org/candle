"""Tests for __torch_dispatch__ tensor subclass protocol."""
import candle as torch
from candle._tensor import Tensor


class DispatchTrackedTensor(Tensor):
    """Tensor subclass that tracks dispatch-level op calls."""
    _dispatch_ops = []

    def __init__(self, data):
        if isinstance(data, Tensor):
            super().__init__(data._storage, data.shape, data.stride, data.offset, data.requires_grad)
            self._inner = data
        else:
            raise TypeError("Requires a Tensor")

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        cls._dispatch_ops.append(func)
        def unwrap(x):
            if isinstance(x, DispatchTrackedTensor):
                return x._inner
            if isinstance(x, (list, tuple)):
                return type(x)(unwrap(i) for i in x)
            return x
        new_args = unwrap(args)
        new_kwargs = {k: unwrap(v) for k, v in (kwargs or {}).items()}
        from candle._dispatch.dispatcher import dispatch
        return dispatch(func, None, *new_args, **new_kwargs)

    @classmethod
    def reset(cls):
        cls._dispatch_ops = []


def test_torch_dispatch_intercepts():
    """__torch_dispatch__ should be called when subclass tensor enters dispatch."""
    DispatchTrackedTensor.reset()
    a = DispatchTrackedTensor(torch.randn(3))
    b = torch.randn(3)
    result = torch.add(a, b)
    assert len(DispatchTrackedTensor._dispatch_ops) > 0, "Expected dispatch interception"


def test_torch_dispatch_after_autograd():
    """__torch_dispatch__ should fire after autograd recording."""
    DispatchTrackedTensor.reset()
    inner = torch.randn(3)
    inner.requires_grad = True
    a = DispatchTrackedTensor(inner)
    b = torch.randn(3)
    result = torch.add(a, b)
    assert len(DispatchTrackedTensor._dispatch_ops) > 0


def test_plain_tensors_unaffected():
    """Plain tensors should not trigger __torch_dispatch__."""
    DispatchTrackedTensor.reset()
    a = torch.randn(3)
    b = torch.randn(3)
    result = torch.add(a, b)
    assert len(DispatchTrackedTensor._dispatch_ops) == 0
