import importlib.machinery
import weakref

import pytest
import candle as torch
from candle._C import _autograd_graph
from candle.autograd.graph import Node


def test_autograd_graph_node_lives_in_compiled_c_boundary():
    assert isinstance(_autograd_graph.__loader__, importlib.machinery.ExtensionFileLoader)
    assert Node.__module__ == "candle._C._autograd_graph"


def test_autograd_graph_node_virtual_base():
    x = torch.randn(2, 2, requires_grad=True)
    y = (x * x).sum()
    assert isinstance(y.grad_fn, Node)
    assert issubclass(type(y.grad_fn), Node)


def test_autograd_graph_node_subclasscheck_requires_type():
    with pytest.raises(TypeError):
        issubclass(object(), Node)


def test_autograd_graph_node_exposes_saved_tensors():
    x = torch.randn(2, 2, requires_grad=True)
    y = x * x
    saved = y.grad_fn._saved_tensors
    assert len(saved) == 2
    assert saved[0] is x
    assert saved[1] is x


def test_autograd_graph_node_raw_saved_tensors_is_distinct_from_materialized_saved_tensors():
    x = torch.randn(2, 2, requires_grad=True)
    y = x * x
    raw = y.grad_fn._raw_saved_tensors
    saved = y.grad_fn._saved_tensors
    assert isinstance(raw, tuple)
    assert len(raw) == len(saved) == 2
    assert raw != saved


def test_autograd_graph_node_saved_tensors_raise_after_release():
    x = torch.randn(2, 2, requires_grad=True)
    y = (x * x).sum()
    node = y.grad_fn
    node.release_saved_tensors()
    with pytest.raises(RuntimeError, match="backward"):
        _ = node._saved_tensors


def test_autograd_graph_node_metadata_is_persistent_and_name_is_stable():
    x = torch.randn(2, 2, requires_grad=True)
    y = (x * x).sum()
    node = y.grad_fn
    metadata = node.metadata
    assert metadata == {}
    metadata["k"] = "v"
    assert node.metadata["k"] == "v"
    name1 = node.name()
    name2 = node.name()
    assert isinstance(name1, str)
    assert name1
    assert name1 == name2


def test_autograd_graph_node_next_functions_is_tuple_of_pairs():
    x = torch.randn(2, 2, requires_grad=True)
    y = (x * x).sum()
    node = y.grad_fn
    nf = node.next_functions
    assert isinstance(nf, tuple)
    assert len(nf) > 0
    for pair in nf:
        assert isinstance(pair, tuple)
        assert len(pair) == 2
        fn, idx = pair
        assert fn is None or isinstance(fn, Node)
        assert isinstance(idx, int)
        assert idx >= 0


def test_autograd_graph_node_next_functions_expose_leaf_edges():
    x = torch.randn(2, 2, requires_grad=True)
    y = x * x
    node = y.grad_fn
    nf = node.next_functions
    assert len(nf) == 2
    assert nf[0][0] is not None
    assert nf[1][0] is not None
    assert isinstance(nf[0][0], Node)
    assert isinstance(nf[1][0], Node)


def test_autograd_graph_node_missing_saved_field_raises_attribute_error():
    x = torch.randn(2, 2, requires_grad=True)
    y = x * x
    node = y.grad_fn
    with pytest.raises(AttributeError):
        _ = node._saved_nonexistent_field


def test_autograd_graph_node_supports_weakref_proxy():
    x = torch.randn(2, 2, requires_grad=True)
    y = x * x
    node = y.grad_fn
    proxy = weakref.proxy(node)
    assert proxy.name() == node.name()
    assert len(proxy.next_functions) == len(node.next_functions)
