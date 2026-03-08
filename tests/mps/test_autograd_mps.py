import numpy as np
import candle as torch


def test_backward_simple_mps():
    x = torch.tensor([1.0, 2.0, 3.0], device="mps")
    x.requires_grad = True
    y = x * 2.0
    y.sum().backward()
    assert x.grad is not None
    np.testing.assert_allclose(x.grad.cpu().numpy(), np.array([2.0, 2.0, 2.0]))


def test_backward_add_mul_mps():
    x = torch.tensor([1.0, 2.0], device="mps")
    y = torch.tensor([3.0, 4.0], device="mps")
    x.requires_grad = True
    y.requires_grad = True
    z = torch.mul(torch.add(x, y), x)
    z.sum().backward()
    assert x.grad is not None
    assert y.grad is not None


def test_grad_matches_cpu_mps():
    x_cpu = torch.tensor([1.0, 2.0, 3.0])
    x_cpu.requires_grad = True
    y_cpu = (x_cpu ** 2).sum()
    y_cpu.backward()

    x_mps = torch.tensor([1.0, 2.0, 3.0], device="mps")
    x_mps.requires_grad = True
    y_mps = (x_mps ** 2).sum()
    y_mps.backward()

    np.testing.assert_allclose(
        x_mps.grad.cpu().numpy(),
        x_cpu.grad.numpy(),
        rtol=1e-5,
    )


def test_matmul_backward_mps():
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="mps")
    b = torch.tensor([[2.0, 0.5], [1.0, -1.0]], device="mps")
    a.requires_grad = True
    b.requires_grad = True
    out = torch.matmul(a, b)
    out.sum().backward()
    assert a.grad is not None
    assert b.grad is not None


def test_linear_backward_mps():
    from candle import nn
    layer = nn.Linear(2, 3).to("mps")
    x = torch.tensor([[1.0, 2.0]], device="mps")
    y = layer(x)
    y.sum().backward()
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None
