"""Integration tests: candle.fx with real candle tensors and nn modules."""
import operator
import candle as torch
from candle import nn


def test_fx_importable_from_candle():
    import candle.fx
    assert hasattr(candle.fx, "Node")
    assert hasattr(candle.fx, "Graph")
    assert hasattr(candle.fx, "GraphModule")
    assert hasattr(candle.fx, "Interpreter")


def test_fx_wrap_accepts_function_name_and_callable():
    import candle.fx

    assert candle.fx.wrap("drop_block2d") == "drop_block2d"

    def fn(x):
        return x

    assert candle.fx.wrap(fn) is fn


def test_fx_proxy_type_exists_for_isinstance_checks():
    import candle.fx

    assert isinstance(candle.fx.Proxy(), candle.fx.Proxy)


def test_fx_graph_with_candle_add():
    from candle.fx import Graph, GraphModule
    g = Graph()
    x = g.placeholder("x")
    y = g.placeholder("y")
    add = g.call_function(torch.add, (x, y))
    out = g.output(add)
    gm = GraphModule(nn.Module(), g)
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    result = gm(a, b)
    expected = torch.tensor([5.0, 7.0, 9.0])
    assert torch.allclose(result, expected)


def test_fx_interpreter_with_candle_ops():
    from candle.fx import Graph, GraphModule, Interpreter
    g = Graph()
    x = g.placeholder("x")
    neg = g.call_function(torch.neg, (x,))
    relu = g.call_function(torch.relu, (neg,))
    out = g.output(relu)
    gm = GraphModule(nn.Module(), g)
    interp = Interpreter(gm)
    t = torch.tensor([-1.0, 2.0, -3.0, 4.0])
    result = interp.run(t)
    expected = torch.tensor([1.0, 0.0, 3.0, 0.0])
    assert torch.allclose(result, expected)


def test_fx_graph_module_with_linear():
    from candle.fx import Graph, GraphModule
    torch.manual_seed(42)
    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 2)
        def forward(self, x):
            return self.linear(x)
    model = MyModel()
    g = Graph()
    x = g.placeholder("x")
    linear = g.call_module("linear", (x,))
    out = g.output(linear)
    gm = GraphModule(model, g)
    inp = torch.randn(3, 4)
    expected = model(inp)
    result = gm(inp)
    assert torch.allclose(result, expected)


def test_fx_graph_module_preserves_training_mode():
    from candle.fx import Graph, GraphModule
    root = nn.Module()
    g = Graph()
    x = g.placeholder("x")
    out = g.output(x)
    gm = GraphModule(root, g)
    assert gm.training is True
    gm.eval()
    assert gm.training is False
    gm.train()
    assert gm.training is True


def test_fx_graph_node_meta():
    from candle.fx import Graph
    g = Graph()
    x = g.placeholder("x")
    x.meta["tensor_meta"] = {"shape": (3, 4), "dtype": "float32"}
    assert x.meta["tensor_meta"]["shape"] == (3, 4)


def test_fx_graph_str_format():
    from candle.fx import Graph
    g = Graph()
    x = g.placeholder("x")
    neg = g.call_function(operator.neg, (x,))
    out = g.output(neg)
    s = str(g)
    assert "placeholder" in s
    assert "call_function" in s or "neg" in s
    assert "output" in s


def test_fx_eliminate_dead_code_preserves_live():
    from candle.fx import Graph
    g = Graph()
    x = g.placeholder("x")
    a = g.call_function(operator.neg, (x,))
    b = g.call_function(operator.neg, (x,))  # dead
    c = g.call_function(operator.add, (a, a))
    out = g.output(c)
    g.eliminate_dead_code()
    nodes = [n.name for n in g.nodes]
    assert "neg" in nodes      # a is alive
    assert "neg_1" not in nodes  # b is dead
    assert "add" in nodes      # c is alive


def test_fx_graph_module_callable_like_original():
    """GraphModule produces identical results to eager execution."""
    from candle.fx import Graph, GraphModule
    g = Graph()
    x = g.placeholder("x")
    matmul_node = g.call_function(torch.matmul, (x, x))
    sum_node = g.call_method("sum", (matmul_node,))
    out = g.output(sum_node)
    gm = GraphModule(nn.Module(), g)
    t = torch.randn(3, 3)
    result = gm(t)
    expected = torch.matmul(t, t).sum()
    assert torch.allclose(result, expected)
