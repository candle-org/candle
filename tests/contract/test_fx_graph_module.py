"""Tests for candle.fx.GraphModule — executable Module wrapping a Graph."""
import operator
import candle as torch
from candle import nn


def test_graph_module_import():
    from candle.fx import GraphModule
    assert GraphModule is not None


def test_graph_module_is_nn_module():
    from candle.fx import Graph, GraphModule
    g = Graph()
    x = g.placeholder("x")
    out = g.output(x)
    gm = GraphModule(nn.Module(), g)
    assert isinstance(gm, nn.Module)


def test_graph_module_forward_identity():
    from candle.fx import Graph, GraphModule
    g = Graph()
    x = g.placeholder("x")
    out = g.output(x)
    gm = GraphModule(nn.Module(), g)
    t = torch.tensor([1.0, 2.0, 3.0])
    result = gm(t)
    assert result is t


def test_graph_module_forward_add():
    from candle.fx import Graph, GraphModule
    g = Graph()
    x = g.placeholder("x")
    y = g.placeholder("y")
    add = g.call_function(operator.add, (x, y))
    out = g.output(add)
    gm = GraphModule(nn.Module(), g)
    result = gm(3, 4)
    assert result == 7


def test_graph_module_with_submodule():
    from candle.fx import Graph, GraphModule
    class MyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 3, bias=False)
        def forward(self, x):
            return self.linear(x)
    root = MyModule()
    g = Graph()
    x = g.placeholder("x")
    linear = g.call_module("linear", (x,))
    out = g.output(linear)
    gm = GraphModule(root, g)
    assert hasattr(gm, "linear")
    t = torch.randn(1, 2)
    result = gm(t)
    assert result.shape == (1, 3)


def test_graph_module_get_attr():
    from candle.fx import Graph, GraphModule
    class MyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        def forward(self, x):
            return x
    root = MyModule()
    g = Graph()
    w = g.get_attr("weight")
    out = g.output(w)
    gm = GraphModule(root, g)
    result = gm()
    assert result.shape == (3,)


def test_graph_module_code_property():
    from candle.fx import Graph, GraphModule
    g = Graph()
    x = g.placeholder("x")
    out = g.output(x)
    gm = GraphModule(nn.Module(), g)
    assert "def forward" in gm.code


def test_graph_module_graph_property():
    from candle.fx import Graph, GraphModule
    g = Graph()
    x = g.placeholder("x")
    out = g.output(x)
    gm = GraphModule(nn.Module(), g)
    assert gm.graph is g


def test_graph_module_recompile():
    from candle.fx import Graph, GraphModule
    g = Graph()
    x = g.placeholder("x")
    y = g.placeholder("y")
    add = g.call_function(operator.add, (x, y))
    out = g.output(add)
    gm = GraphModule(nn.Module(), g)
    assert gm(2, 3) == 5
    # Modify graph: replace add with mul
    with g.inserting_before(out):
        mul = g.call_function(operator.mul, (x, y))
    out.args = (mul,)
    g.eliminate_dead_code()
    gm.recompile()
    assert gm(2, 3) == 6


def test_graph_module_parameters():
    from candle.fx import Graph, GraphModule
    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 5)
        def forward(self, x):
            return self.linear(x)
    root = nn.Linear(4, 5)
    wrapper = Wrapper()
    g = Graph()
    x = g.placeholder("x")
    linear = g.call_module("linear", (x,))
    out = g.output(linear)
    gm = GraphModule(wrapper, g)
    params = list(gm.parameters())
    assert len(params) == 2  # weight + bias


def test_graph_module_print_readable():
    from candle.fx import Graph, GraphModule
    g = Graph()
    x = g.placeholder("x")
    out = g.output(x)
    gm = GraphModule(nn.Module(), g)
    readable = gm.print_readable(print_output=False)
    assert "def forward" in readable