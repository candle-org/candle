import pytest
import numpy as np
import candle as torch


def test_autograd_engine_accumulates_shared_subgraph():
    a = torch.ones((2, 2)).requires_grad_()
    b = a * a
    c = b + b
    c.sum().backward()
    assert a.grad is not None
    # b = a * a, c = b + b => dc/da = 4a
    assert (a.grad.numpy() == 4).all()


def test_autograd_engine_accumulates_reused_leaf():
    # Use fixed values to verify the direct x path and the reused-leaf branch path
    # both contribute to x.grad.
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = torch.tensor([[0.5, 2.0], [4.0, 8.0]], requires_grad=True)
    z = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    grad_output = torch.ones((2, 2))

    term3 = 4 * z**2 * x / y
    (x + term3).backward(grad_output)

    expected = torch.tensor([[9.0, 9.0], [10.0, 9.0]], dtype=x.grad.dtype)
    assert x.grad is not None
    torch.testing.assert_close(x.grad, expected)

def test_autograd_engine_reentrant_backward():
    # Backward inside backward hook should be supported.
    a = torch.ones((2, 2)).requires_grad_()
    b = a * a
    c = b.sum()
    d = (a * a).sum()
    called = {"ok": False}

    def hook(_grad):
        d.backward()
        called["ok"] = True

    c.register_hook(hook)
    c.backward()
    assert called["ok"]


def test_saved_tensors_hooks_unpacked_once_per_node():
    a = torch.ones((2, 2)).requires_grad_()
    counters = {"unpack": 0}

    def pack(t):
        return t

    def unpack(t):
        counters["unpack"] += 1
        return t

    from candle.autograd.graph import saved_tensors_hooks

    with saved_tensors_hooks(pack, unpack):
        b = a * a
        c = b + b
    c.sum().backward()
    # mul saves two tensors; add does not save inputs.
    assert counters["unpack"] == 2


def test_autograd_grad_allow_unused_false_raises_for_unused_input():
    a = torch.tensor([1.0], requires_grad=True)
    b = torch.tensor([2.0], requires_grad=True)
    out = (a * a).sum()

    with pytest.raises(RuntimeError, match="not have been used in the graph"):
        torch.autograd.grad(out, (a, b), allow_unused=False)


def test_autograd_grad_allow_unused_true_returns_none_for_unused_input():
    a = torch.tensor([1.0], requires_grad=True)
    b = torch.tensor([2.0], requires_grad=True)
    out = (a * a).sum()

    grad_a, grad_b = torch.autograd.grad(out, (a, b), allow_unused=True)

    assert grad_a is not None
    assert grad_b is None
