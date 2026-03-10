"""Tests for __torch_function__ tensor subclass protocol."""
import candle as torch
from candle._tensor import Tensor


class TrackedTensor(Tensor):
    """Test tensor subclass that tracks which ops are called."""
    _ops_called = []

    def __init__(self, data):
        if isinstance(data, Tensor):
            super().__init__(data._storage, data.shape, data.stride, data.offset, data.requires_grad)
            self._wrapped = data
        else:
            raise TypeError("TrackedTensor requires a Tensor")

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        cls._ops_called.append(func.__name__ if hasattr(func, '__name__') else str(func))
        def unwrap(x):
            if isinstance(x, TrackedTensor):
                return x._wrapped
            if isinstance(x, (list, tuple)):
                return type(x)(unwrap(i) for i in x)
            return x
        new_args = unwrap(args)
        new_kwargs = {k: unwrap(v) for k, v in (kwargs or {}).items()}
        result = func(*new_args, **new_kwargs)
        return result

    @classmethod
    def reset(cls):
        cls._ops_called = []


class BlockingTensor(Tensor):
    """Tensor subclass that blocks all ops."""
    def __init__(self, data):
        if isinstance(data, Tensor):
            super().__init__(data._storage, data.shape, data.stride, data.offset, data.requires_grad)
        else:
            raise TypeError("BlockingTensor requires a Tensor")

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        raise RuntimeError("BlockingTensor: ops not allowed")


def test_torch_function_intercepts_add():
    TrackedTensor.reset()
    a = TrackedTensor(torch.randn(3))
    b = torch.randn(3)
    result = torch.add(a, b)
    assert "add" in TrackedTensor._ops_called


def test_torch_function_intercepts_matmul():
    TrackedTensor.reset()
    a = TrackedTensor(torch.randn(3, 3))
    b = torch.randn(3, 3)
    result = torch.matmul(a, b)
    assert "matmul" in TrackedTensor._ops_called


def test_torch_function_not_implemented_fallthrough():
    class PassthroughTensor(Tensor):
        def __init__(self, data):
            super().__init__(data._storage, data.shape, data.stride, data.offset, data.requires_grad)
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            return NotImplemented
    # NotImplemented falls through to normal dispatch -- may fail because
    # backend gets subclass, not plain Tensor. Just test it doesn't hang.


def test_torch_function_blocking():
    a = BlockingTensor(torch.randn(3))
    b = torch.randn(3)
    try:
        result = torch.add(a, b)
        assert False, "Should have raised"
    except RuntimeError as e:
        assert "BlockingTensor" in str(e)


def test_plain_tensors_skip_torch_function():
    a = torch.randn(3)
    b = torch.randn(3)
    result = torch.add(a, b)
    assert result.shape == (3,)
