import candle as torch
from candle import nn


def test_linear_forward_backward():
    layer = nn.Linear(2, 3)
    x = torch.tensor([[1.0, 2.0]])
    y = layer(x)
    y.sum().backward()
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None
