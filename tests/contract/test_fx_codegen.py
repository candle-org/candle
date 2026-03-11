"""Tests for Graph.python_code() — source code generation from graph IR."""
import operator


def test_codegen_simple_add():
    """python_code generates valid Python for a simple add graph."""
    from candle.fx import Graph
    g = Graph()
    x = g.placeholder("x")
    y = g.placeholder("y")
    add = g.call_function(operator.add, (x, y))
    out = g.output(add)
    code = g.python_code()
    assert "def forward(self, x, y):" in code
    assert "return add" in code


def test_codegen_call_method():
    from candle.fx import Graph
    g = Graph()
    x = g.placeholder("x")
    view = g.call_method("view", (x, 2, 3))
    out = g.output(view)
    code = g.python_code()
    assert "x.view(2, 3)" in code
    assert "return view" in code


def test_codegen_call_module():
    from candle.fx import Graph
    g = Graph()
    x = g.placeholder("x")
    linear = g.call_module("linear", (x,))
    out = g.output(linear)
    code = g.python_code()
    assert "self.linear(x)" in code
    assert "return linear" in code


def test_codegen_get_attr():
    from candle.fx import Graph
    g = Graph()
    w = g.get_attr("linear.weight")
    out = g.output(w)
    code = g.python_code()
    assert "self.linear.weight" in code


def test_codegen_tuple_output():
    from candle.fx import Graph
    g = Graph()
    x = g.placeholder("x")
    y = g.placeholder("y")
    out = g.output((x, y))
    code = g.python_code()
    assert "return" in code
    # Should contain both x and y in the return
    assert "x" in code.split("return")[-1]
    assert "y" in code.split("return")[-1]


def test_codegen_is_executable():
    """Generated code can be compiled and executed."""
    from candle.fx import Graph
    g = Graph()
    x = g.placeholder("x")
    y = g.placeholder("y")
    add = g.call_function(operator.add, (x, y))
    out = g.output(add)
    code = g.python_code()
    # Need to provide the necessary globals for exec
    import _operator
    globs = {"_operator": _operator, "operator": operator}
    # Also add the operator module since codegen may use qualified names
    exec(compile(code, "<test>", "exec"), globs)
    forward = globs["forward"]
    # forward(self, x, y) — pass None as self since no module access needed
    assert forward(None, 3, 4) == 7