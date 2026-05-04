"""Pin conditional-view forward semantics for contiguous/flatten/unflatten.

After Sub-batch 1B-A:
- contiguous(a) returns `a` itself when a is already contiguous in default
  format. Non-contiguous input still copies.
- flatten(a, ...) and unflatten(a, ...) return a real view (storage shared,
  _base set, _view_meta["op"] populated) when a is contiguous. Non-contiguous
  input still copies.

Tests stay on CPU — MPS and NPU regression coverage rides on the cpu/contract
gate plus the existing per-backend test suites.
"""
import numpy as np
import candle as torch


# ---------------------------------------------------------------------------
# contiguous
# ---------------------------------------------------------------------------

def test_contiguous_returns_self_when_already_contiguous():
    t = torch.randn(3, 4)
    assert t.is_contiguous()
    u = t.contiguous()
    assert u is t


def test_contiguous_copies_when_input_is_non_contiguous():
    t = torch.randn(4, 4).t()
    assert not t.is_contiguous()
    u = t.contiguous()
    assert u is not t
    assert u.storage() is not t.storage()
    assert u.is_contiguous()


# ---------------------------------------------------------------------------
# flatten
# ---------------------------------------------------------------------------

def test_flatten_shares_storage_when_input_is_contiguous():
    t = torch.randn(3, 4)
    u = t.flatten()
    assert u.storage() is t.storage()
    assert u._base is not None
    assert u._view_meta is not None
    assert u._view_meta["op"] == "flatten"
    assert u._view_func is None
    assert u._rev_view_func is None


def test_flatten_view_mutation_propagates_to_input():
    t = torch.randn(3, 4)
    u = t.flatten()
    u[0] = 99.0
    np.testing.assert_array_equal(t.flatten().numpy()[0], 99.0)


def test_flatten_copies_when_input_is_non_contiguous():
    t = torch.randn(4, 4).t()
    assert not t.is_contiguous()
    u = t.flatten()
    assert u.storage() is not t.storage()


# ---------------------------------------------------------------------------
# unflatten
# ---------------------------------------------------------------------------

def test_unflatten_shares_storage_when_input_is_contiguous():
    t = torch.randn(2, 6)
    u = t.unflatten(1, (2, 3))
    assert u.storage() is t.storage()
    assert u._base is not None
    assert u._view_meta is not None
    assert u._view_meta["op"] == "unflatten"
    assert u._view_func is None
    assert u._rev_view_func is None


def test_unflatten_view_mutation_propagates_to_input():
    t = torch.randn(2, 6)
    u = t.unflatten(1, (2, 3))
    u[0, 0, 0] = 99.0
    np.testing.assert_array_equal(t.numpy()[0, 0], 99.0)


def test_unflatten_copies_when_input_is_non_contiguous():
    t = torch.randn(4, 4).t()
    assert not t.is_contiguous()
    u = t.unflatten(0, (2, 2))
    assert u.storage() is not t.storage()


# ---------------------------------------------------------------------------
# Backward through the fast-path is unchanged: hand-added *Backward0 classes
# in src/candle/_generated/functions.py still drive grad propagation in 1B-A.
# These tests guard against regressions in that wiring.
# ---------------------------------------------------------------------------

def test_flatten_backward_is_identity_after_fast_path():
    t = torch.randn(3, 4, requires_grad=True)
    u = t.flatten()
    u.sum().backward()
    np.testing.assert_array_equal(t.grad.numpy(), np.ones((3, 4), dtype=np.float32))


def test_unflatten_backward_is_identity_after_fast_path():
    t = torch.randn(2, 6, requires_grad=True)
    u = t.unflatten(1, (2, 3))
    u.sum().backward()
    np.testing.assert_array_equal(t.grad.numpy(), np.ones((2, 6), dtype=np.float32))


def test_contiguous_backward_is_identity_when_already_contiguous():
    t = torch.randn(3, 4, requires_grad=True)
    u = t.contiguous()
    assert u is t
    u.sum().backward()
    np.testing.assert_array_equal(t.grad.numpy(), np.ones((3, 4), dtype=np.float32))
