import candle as torch
from candle.autograd.graph import Node


def test_autograd_graph_node_virtual_base():
    x = torch.randn(2, 2, requires_grad=True)
    y = (x * x).sum()
    assert isinstance(y.grad_fn, Node)
    assert issubclass(type(y.grad_fn), Node)
