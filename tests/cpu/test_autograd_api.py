import pytest
import candle as torch


def test_backward_requires_grad_for_non_scalar():
    t = torch.ones((2,))
    with pytest.raises(RuntimeError):
        t.backward()


def test_backward_defaults_to_ones_for_scalar():
    t = torch.ones((2,))
    t.requires_grad = True
    y = t.sum()
    y.backward()
    assert t.grad is not None
    assert t.grad.numpy().tolist() == [1.0, 1.0]


def test_retain_graph_allows_double_backward():
    t = torch.ones((2,))
    t.requires_grad = True
    y = t.sum()
    y.backward(retain_graph=True)
    y.backward(retain_graph=True)
    assert t.grad is not None


def test_retain_grad_populates_non_leaf_grad():
    t = torch.ones((2,))
    y = t.sum()
    y.retain_grad()
    y.backward()
    assert y.grad is not None


def test_detach_breaks_grad_chain():
    t = torch.ones((2,))
    t.requires_grad_(True)
    y = t.detach()
    assert y.requires_grad is False


def test_detach_inplace():
    t = torch.ones((2,))
    t.requires_grad_(True)
    t.detach_()
    assert t.requires_grad is False


def test_register_hook_receives_grad():
    t = torch.ones((2,))
    t.requires_grad_(True)
    seen = []

    def hook(grad):
        seen.append(grad.numpy().tolist())
        return grad

    t.register_hook(hook)
    t.sum().backward()
    assert seen == [[1.0, 1.0]]


def test_backward_leaf_output_copies_input_grad_tensor():
    x = torch.tensor([2.0], requires_grad=True)
    grad = torch.tensor([3.0], requires_grad=True)

    x.backward(grad)

    assert x.grad.tolist() == [3.0]
    assert x.grad is not grad
    assert x.grad.requires_grad is False


def test_autograd_grad_basic():
    x = torch.ones((2,))
    x.requires_grad_(True)
    y = (x * x).sum()
    (gx,) = torch.autograd.grad(y, (x,))
    assert gx.numpy().tolist() == [2.0, 2.0]


def test_autograd_grad_leaf_identity_applies_hooks_once():
    x = torch.tensor([2.0], requires_grad=True)
    grad = torch.tensor([3.0])
    seen = []

    def hook(g):
        seen.append(g.tolist())
        return g

    x.register_hook(hook)

    (gx,) = torch.autograd.grad(x, (x,), grad_outputs=grad)

    assert gx.tolist() == [3.0]
    assert seen == [[3.0]]


def test_autograd_grad_leaf_identity_create_graph_keeps_grad_flag():
    x = torch.tensor([2.0], requires_grad=True)
    grad = torch.tensor([3.0])

    (gx,) = torch.autograd.grad(x, (x,), grad_outputs=grad, create_graph=True)

    assert gx.tolist() == [3.0]
    assert gx.requires_grad is False


def test_autograd_grad_allow_unused():
    x = torch.ones((2,))
    x.requires_grad_(True)
    unused = torch.ones((2,))
    unused.requires_grad_(True)
    y = x.sum()
    with pytest.raises(RuntimeError):
        torch.autograd.grad(y, (unused,))

    x = torch.ones((2,))
    x.requires_grad_(True)
    unused = torch.ones((2,))
    unused.requires_grad_(True)
    y = x.sum()
    gx = torch.autograd.grad(y, (unused,), allow_unused=True)[0]
    assert gx is None


def test_autograd_grad_allow_unused_still_rejects_outputs_without_grad():
    x = torch.ones((2,))
    x.requires_grad_(True)
    y = torch.ones((1,))

    with pytest.raises(RuntimeError, match="does not require grad"):
        torch.autograd.grad(y, (x,), allow_unused=True)


def test_autograd_grad_rejects_mixed_outputs_without_grad():
    x = torch.ones((1,))
    x.requires_grad_(True)
    y = x.sum()
    z = torch.ones((1,))

    with pytest.raises(RuntimeError, match="element 1 of tensors does not require grad"):
        torch.autograd.grad((y, z), (x,), grad_outputs=(torch.ones_like(y), torch.ones_like(z)))


def test_autograd_grad_materialize_grads_returns_zeros_for_unused_input():
    a = torch.tensor([2.0], requires_grad=True)
    b = torch.tensor([5.0], requires_grad=True)
    out = (a * a).sum()

    grad_a, grad_b = torch.autograd.grad(out, (a, b), materialize_grads=True)

    assert grad_a is not None
    assert grad_a.tolist() == [4.0]
    assert grad_b is not None
    assert grad_b.tolist() == [0.0]
    assert grad_b.requires_grad is False


def test_autograd_grad_materialize_grads_create_graph_requires_grad():
    a = torch.tensor([2.0], requires_grad=True)
    b = torch.tensor([5.0], requires_grad=True)
    out = (a * a).sum()

    grad_a, grad_b = torch.autograd.grad(out, (a, b), create_graph=True, materialize_grads=True)

    assert grad_a.requires_grad is True
    assert grad_b.tolist() == [0.0]
    assert grad_b.requires_grad is True


def test_autograd_grad_materialize_grads_rejects_allow_unused_false():
    a = torch.tensor([2.0], requires_grad=True)
    b = torch.tensor([5.0], requires_grad=True)
    out = (a * a).sum()

    with pytest.raises(ValueError, match="allow_unused"):
        torch.autograd.grad(out, (a, b), materialize_grads=True, allow_unused=False)
