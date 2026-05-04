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
