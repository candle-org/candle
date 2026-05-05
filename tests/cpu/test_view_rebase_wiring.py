"""Pin engine view-rebase wiring for the 5 ops landed in 1B-B.

After 1B-B:
  - flatten/unflatten/squeeze/narrow/movedim views carry _view_func + _rev_view_func.
  - Backward through these ops flows via engine rebase, not via *Backward0 classes.
  - Chained views rebase recursively.
  - The 7 hand-added Backward0 classes are gone from _generated/functions.py.
"""
import numpy as np
import candle as torch


# --- Per-op rebase semantics ----------------------------------------------

def test_flatten_view_carries_view_func_and_rev_view_func():
    t = torch.randn(3, 4)
    u = t.flatten()
    assert callable(u._view_func)
    assert callable(u._rev_view_func)
    g = torch.ones(u.shape)
    assert u._rev_view_func(g).shape == t.shape


def test_squeeze_view_carries_view_func_and_rev_view_func():
    t = torch.randn(1, 3, 1, 4)
    u = t.squeeze()
    assert callable(u._view_func)
    assert callable(u._rev_view_func)
    assert u._rev_view_func(torch.ones(u.shape)).shape == t.shape


def test_narrow_rev_view_func_pads_with_zeros():
    t = torch.randn(5, 4)
    u = t.narrow(0, 1, 3)
    g = torch.ones(u.shape)
    g_base = u._rev_view_func(g)
    assert g_base.shape == t.shape
    np.testing.assert_array_equal(g_base.numpy()[0], np.zeros(4, dtype=np.float32))
    np.testing.assert_array_equal(g_base.numpy()[1], np.ones(4, dtype=np.float32))
    np.testing.assert_array_equal(g_base.numpy()[4], np.zeros(4, dtype=np.float32))


def test_movedim_rev_view_func_swaps_axes_back():
    t = torch.randn(2, 3, 4, requires_grad=True)
    u = t.movedim(0, 2)
    assert tuple(u.shape) == (3, 4, 2)
    g = torch.ones(u.shape)
    assert u._rev_view_func(g).shape == t.shape


def test_unflatten_rev_view_func_reshapes():
    t = torch.randn(2, 6)
    u = t.unflatten(1, (2, 3))
    g = torch.ones(u.shape)
    assert u._rev_view_func(g).shape == t.shape


# --- End-to-end backward semantics ----------------------------------------

def test_flatten_backward_via_rebase():
    t = torch.randn(3, 4, requires_grad=True)
    u = t.flatten()
    u.sum().backward()
    np.testing.assert_array_equal(t.grad.numpy(), np.ones((3, 4), dtype=np.float32))


def test_narrow_backward_via_rebase():
    t = torch.randn(5, 4, requires_grad=True)
    u = t.narrow(0, 1, 3)
    u.sum().backward()
    expected = np.zeros((5, 4), dtype=np.float32)
    expected[1:4] = 1.0
    np.testing.assert_array_equal(t.grad.numpy(), expected)


def test_squeeze_backward_via_rebase():
    t = torch.randn(1, 3, 1, 4, requires_grad=True)
    u = t.squeeze()
    u.sum().backward()
    np.testing.assert_array_equal(t.grad.numpy(), np.ones((1, 3, 1, 4), dtype=np.float32))


def test_movedim_backward_via_rebase():
    t = torch.randn(2, 3, 4, requires_grad=True)
    u = t.movedim(0, 2)
    u.sum().backward()
    np.testing.assert_array_equal(t.grad.numpy(), np.ones((2, 3, 4), dtype=np.float32))


def test_unflatten_backward_via_rebase():
    t = torch.randn(2, 6, requires_grad=True)
    u = t.unflatten(1, (2, 3))
    u.sum().backward()
    np.testing.assert_array_equal(t.grad.numpy(), np.ones((2, 6), dtype=np.float32))


# --- Chained views rebase recursively -------------------------------------

def test_chained_view_rebase_flatten_then_narrow():
    t = torch.randn(3, 4, requires_grad=True)
    u = t.flatten().narrow(0, 0, 6)  # 2 nested views
    u.sum().backward()
    expected = np.zeros((3, 4), dtype=np.float32).reshape(-1)
    expected[:6] = 1.0
    np.testing.assert_array_equal(t.grad.numpy().reshape(-1), expected)
