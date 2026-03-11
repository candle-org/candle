"""Tests for candle.fx.Node — the atomic unit of the FX graph IR."""
import operator


def test_node_import():
    from candle.fx import Node
    assert Node is not None


def test_node_attributes():
    from candle.fx.graph import Graph
    g = Graph()
    n = g.create_node("placeholder", "x")
    assert n.op == "placeholder"
    assert n.target == "x"
    assert n.name == "x"
    assert n.args == ()
    assert n.kwargs == {}
    assert n.meta == {}
    assert n.graph is g


def test_node_users_tracking():
    from candle.fx.graph import Graph
    g = Graph()
    x = g.placeholder("x")
    y = g.placeholder("y")
    add = g.call_function(operator.add, (x, y))
    assert x in add.all_input_nodes
    assert y in add.all_input_nodes
    assert add in x.users
    assert add in y.users


def test_node_users_auto_update_on_args_set():
    from candle.fx.graph import Graph
    g = Graph()
    x = g.placeholder("x")
    y = g.placeholder("y")
    z = g.placeholder("z")
    add = g.call_function(operator.add, (x, y))
    add.args = (x, z)
    assert add not in y.users
    assert add in z.users
    assert add in x.users


def test_node_replace_all_uses_with():
    from candle.fx.graph import Graph
    g = Graph()
    a = g.placeholder("a")
    b = g.call_function(operator.neg, (a,))
    c = g.call_function(operator.add, (b, b))
    new = g.placeholder("new")
    b.replace_all_uses_with(new)
    assert c.args == (new, new)
    assert c not in b.users
    assert c in new.users


def test_node_replace_input_with():
    from candle.fx.graph import Graph
    g = Graph()
    x = g.placeholder("x")
    y = g.placeholder("y")
    z = g.placeholder("z")
    add = g.call_function(operator.add, (x, y))
    add.replace_input_with(y, z)
    assert add.args == (x, z)
    assert add not in y.users
    assert add in z.users


def test_node_update_arg():
    from candle.fx.graph import Graph
    g = Graph()
    x = g.placeholder("x")
    y = g.placeholder("y")
    z = g.placeholder("z")
    add = g.call_function(operator.add, (x, y))
    add.update_arg(1, z)
    assert add.args == (x, z)
    assert add not in y.users
    assert add in z.users


def test_node_update_kwarg():
    from candle.fx.graph import Graph
    g = Graph()
    x = g.placeholder("x")
    y = g.placeholder("y")
    add = g.call_function(operator.add, (x,), {"other": y})
    z = g.placeholder("z")
    add.update_kwarg("other", z)
    assert add.kwargs["other"] is z
    assert add not in y.users
    assert add in z.users


def test_node_all_input_nodes():
    from candle.fx.graph import Graph
    g = Graph()
    x = g.placeholder("x")
    y = g.placeholder("y")
    add = g.call_function(operator.add, (x, x), {"bias": y})
    inputs = add.all_input_nodes
    assert x in inputs
    assert y in inputs
    assert len(inputs) == 2


def test_node_all_input_nodes_nested():
    from candle.fx.graph import Graph
    g = Graph()
    a = g.placeholder("a")
    b = g.placeholder("b")
    c = g.placeholder("c")
    cat = g.call_function(lambda *args: None, ([a, b], c))
    inputs = cat.all_input_nodes
    assert set(inputs) == {a, b, c}


def test_node_prev_next_linked_list():
    from candle.fx.graph import Graph
    g = Graph()
    x = g.placeholder("x")
    y = g.placeholder("y")
    assert x.next is y
    assert y.prev is x


def test_node_repr():
    from candle.fx.graph import Graph
    g = Graph()
    x = g.placeholder("x")
    assert repr(x) == "x"


def test_node_identity_equality():
    from candle.fx.graph import Graph
    g = Graph()
    a = g.placeholder("a")
    b = g.placeholder("a")
    assert a != b
    assert a is not b
    assert a == a
