"""Pin the new view-tracking attributes on TensorImpl.

Sub-batch 1A only adds the storage; sub-batch 1B will populate it for the
real view ops.  Tests here only assert the attributes exist, default to
None, and accept assignment of a callable.
"""
import candle


def test_tensor_has_view_func_attribute_defaulting_to_none():
    t = candle.zeros((2, 3))
    assert hasattr(t, "_view_func")
    assert t._view_func is None


def test_tensor_has_rev_view_func_attribute_defaulting_to_none():
    t = candle.zeros((2, 3))
    assert hasattr(t, "_rev_view_func")
    assert t._rev_view_func is None


def test_view_func_attribute_accepts_callable_assignment():
    t = candle.zeros((2, 3))

    def f(base):
        return base

    t._view_func = f
    assert t._view_func is f


def test_rev_view_func_attribute_accepts_callable_assignment():
    t = candle.zeros((2, 3))

    def f(grad):
        return grad

    t._rev_view_func = f
    assert t._rev_view_func is f


def test_make_view_propagates_view_func_and_rev_view_func():
    from candle._backends.common.view import _make_view
    base = candle.zeros((2, 3))
    base.requires_grad = True

    def fwd(new_base):
        return new_base

    def rev(grad):
        return grad

    view = _make_view(
        base,
        shape=(2, 3),
        stride=(3, 1),
        offset=0,
        op="identity",
        view_func=fwd,
        rev_view_func=rev,
    )
    assert view._view_func is fwd
    assert view._rev_view_func is rev
    assert view._base is base


def test_make_view_default_view_func_and_rev_view_func_is_none():
    from candle._backends.common.view import _make_view
    base = candle.zeros((2, 3))
    view = _make_view(base, shape=(2, 3), stride=(3, 1), offset=0, op="identity")
    assert view._view_func is None
    assert view._rev_view_func is None


def test_grad_on_view_with_rev_view_func_rebases_onto_base():
    """If a view tensor with _rev_view_func receives gradient during backward,
    the engine must rebase that grad through _rev_view_func and accumulate it
    onto _base, leaving the view's own .grad as None.
    """
    import numpy as np
    from candle._backends.common.view import _make_view

    base = candle.zeros((4,))
    base.requires_grad = True

    def fwd(new_base):
        return new_base

    def rev(grad_view):  # pylint: disable=unused-argument
        return candle.zeros((4,))

    view = _make_view(
        base,
        shape=(2,),
        stride=(1,),
        offset=0,
        op="identity_truncate",
        view_func=fwd,
        rev_view_func=rev,
    )
    view.requires_grad = True

    g_view = candle.ones((2,))
    base.grad = None
    view._accumulate_grad_node = None  # avoid hook interference

    view.backward(g_view)

    assert view.grad is None
    assert base.grad is not None
    np.testing.assert_array_equal(base.grad.numpy(), np.zeros((4,)))
