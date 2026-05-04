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
