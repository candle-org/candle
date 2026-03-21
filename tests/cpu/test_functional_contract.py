"""Contract tests for _functional.py hot wrappers.

Locks current behavior of add / mul / matmul / relu / transpose / reshape / neg
including alpha= semantics and __torch_function__ override routing.
These must stay green before and after any Cython migration.
"""
import pytest
import candle as torch
from candle._tensor import Tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _allclose(a, b, atol=1e-5):
    a_list = a.flatten().tolist()
    b_list = list(b)
    assert len(a_list) == len(b_list), f"length mismatch: {len(a_list)} vs {len(b_list)}"
    for i, (x, y) in enumerate(zip(a_list, b_list)):
        assert abs(x - y) <= atol, f"index {i}: {x} vs {y} (atol={atol})"


# ===========================================================================
# 1. add wrapper
# ===========================================================================

class TestAddWrapper:

    def test_add_basic_values(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])
        out = torch.add(a, b)
        _allclose(out, [5.0, 7.0, 9.0])

    def test_add_returns_tensor(self):
        a = torch.tensor([1.0])
        b = torch.tensor([2.0])
        out = torch.add(a, b)
        assert isinstance(out, Tensor)

    def test_add_shape_preserved(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        out = torch.add(a, b)
        assert out.shape == (2, 2)

    def test_add_alpha_scales_second_arg(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.0, 1.0, 1.0])
        out = torch.add(a, b, alpha=3)
        # a + 3*b = [4, 5, 6]
        _allclose(out, [4.0, 5.0, 6.0])

    def test_add_alpha_one_same_as_no_alpha(self):
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0])
        out_default = torch.add(a, b)
        out_alpha1 = torch.add(a, b, alpha=1)
        _allclose(out_default, out_alpha1.flatten().tolist())

    def test_add_alpha_zero_ignores_b(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([99.0, 99.0, 99.0])
        out = torch.add(a, b, alpha=0)
        # a + 0*b = a
        _allclose(out, [1.0, 2.0, 3.0])

    def test_add_alpha_float(self):
        a = torch.tensor([0.0, 0.0])
        b = torch.tensor([2.0, 4.0])
        out = torch.add(a, b, alpha=0.5)
        _allclose(out, [1.0, 2.0])

    def test_add_out_none_accepted(self):
        # out= is accepted without error (popped internally)
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0])
        out = torch.add(a, b, out=None)
        _allclose(out, [4.0, 6.0])

    def test_add_scalar_rhs(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        out = torch.add(a, 2.0)
        _allclose(out, [3.0, 4.0, 5.0])

    def test_add_scalar_lhs_currently_errors(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        with pytest.raises(RuntimeError, match="_numpy_view"):
            torch.add(2.0, a)

    def test_add_broadcast(self):
        a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = torch.tensor([10.0, 20.0, 30.0])
        out = torch.add(a, b)
        assert out.shape == (2, 3)
        _allclose(out, [11.0, 22.0, 33.0, 14.0, 25.0, 36.0])

    def test_add_no_grad_fn_when_no_grad(self):
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0])
        out = torch.add(a, b)
        assert out.grad_fn is None


# ===========================================================================
# 2. mul wrapper
# ===========================================================================

class TestMulWrapper:

    def test_mul_basic_values(self):
        a = torch.tensor([2.0, 3.0, 4.0])
        b = torch.tensor([5.0, 6.0, 7.0])
        out = torch.mul(a, b)
        _allclose(out, [10.0, 18.0, 28.0])

    def test_mul_returns_tensor(self):
        a = torch.tensor([1.0])
        b = torch.tensor([2.0])
        assert isinstance(torch.mul(a, b), Tensor)

    def test_mul_shape_preserved(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = torch.tensor([[2.0, 2.0], [2.0, 2.0]])
        out = torch.mul(a, b)
        assert out.shape == (2, 2)

    def test_mul_scalar(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        out = torch.mul(a, 3.0)
        _allclose(out, [3.0, 6.0, 9.0])

    def test_mul_broadcast(self):
        a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = torch.tensor([2.0, 3.0, 4.0])
        out = torch.mul(a, b)
        assert out.shape == (2, 3)
        _allclose(out, [2.0, 6.0, 12.0, 8.0, 15.0, 24.0])

    def test_mul_backward_correct(self):
        a = torch.tensor([2.0, 3.0])
        b = torch.tensor([4.0, 5.0])
        a.requires_grad = True
        out = torch.mul(a, b)
        out.sum().backward()
        # d(a*b)/da = b
        _allclose(a.grad, [4.0, 5.0])


# ===========================================================================
# 3. matmul wrapper
# ===========================================================================

class TestMatmulWrapper:

    def test_matmul_2d_identity(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        out = torch.matmul(a, b)
        _allclose(out, [1.0, 2.0, 3.0, 4.0])

    def test_matmul_shape(self):
        a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = torch.tensor([[1.0], [2.0], [3.0]])
        out = torch.matmul(a, b)
        assert out.shape == (2, 1)

    def test_matmul_returns_tensor(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = torch.tensor([[1.0], [2.0]])
        assert isinstance(torch.matmul(a, b), Tensor)

    def test_matmul_values(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = torch.tensor([[1.0], [2.0]])
        out = torch.matmul(a, b)
        _allclose(out, [5.0, 11.0])

    def test_matmul_1d_dot_product(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])
        out = torch.matmul(a, b)
        assert out.shape == ()
        assert out.item() == 32.0


# ===========================================================================
# 4. relu wrapper
# ===========================================================================

class TestReluWrapper:

    def test_relu_zeros_negatives(self):
        a = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        out = torch.relu(a)
        _allclose(out, [0.0, 0.0, 0.0, 1.0, 2.0])

    def test_relu_positive_passthrough(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        out = torch.relu(a)
        _allclose(out, [1.0, 2.0, 3.0])

    def test_relu_returns_tensor(self):
        a = torch.tensor([1.0, -1.0])
        assert isinstance(torch.relu(a), Tensor)

    def test_relu_shape_preserved(self):
        a = torch.tensor([[1.0, -1.0], [-2.0, 3.0]])
        out = torch.relu(a)
        assert out.shape == (2, 2)

    def test_relu_preserves_grad_fn(self):
        a = torch.tensor([1.0, -1.0])
        a.requires_grad = True
        out = torch.relu(a)
        assert out.grad_fn is not None

    def test_relu_backward(self):
        a = torch.tensor([1.0, -1.0, 0.5])
        a.requires_grad = True
        out = torch.relu(a)
        out.sum().backward()
        _allclose(a.grad, [1.0, 0.0, 1.0])


# ===========================================================================
# 5. transpose wrapper
# ===========================================================================

class TestTransposeWrapper:

    def test_transpose_2d(self):
        a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = torch.transpose(a, 0, 1)
        assert out.shape == (3, 2)
        assert out[0].tolist() == [1.0, 4.0]
        assert out[1].tolist() == [2.0, 5.0]
        assert out[2].tolist() == [3.0, 6.0]

    def test_transpose_returns_tensor(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert isinstance(torch.transpose(a, 0, 1), Tensor)

    def test_transpose_dim_swap(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        t = torch.transpose(a, 0, 1)
        assert t[0, 0].item() == a[0, 0].item()
        assert t[1, 0].item() == a[0, 1].item()
        assert t[0, 1].item() == a[1, 0].item()

    def test_transpose_idempotent(self):
        a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = torch.transpose(torch.transpose(a, 0, 1), 0, 1)
        assert out.shape == a.shape
        _allclose(out, a.flatten().tolist())

    def test_transpose_negative_dim(self):
        a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = torch.transpose(a, 0, -1)
        assert out.shape == (3, 2)
        assert out[0].tolist() == [1.0, 4.0]
        assert out[2].tolist() == [3.0, 6.0]


# ===========================================================================
# 6. reshape wrapper
# ===========================================================================

class TestReshapeWrapper:

    def test_reshape_basic(self):
        a = torch.tensor([1.0, 2.0, 3.0, 4.0])
        out = torch.reshape(a, (2, 2))
        assert out.shape == (2, 2)

    def test_reshape_values_preserved(self):
        a = torch.tensor([1.0, 2.0, 3.0, 4.0])
        out = torch.reshape(a, (2, 2))
        _allclose(out, [1.0, 2.0, 3.0, 4.0])

    def test_reshape_returns_tensor(self):
        a = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert isinstance(torch.reshape(a, (4, 1)), Tensor)

    def test_reshape_to_1d(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        out = torch.reshape(a, (4,))
        assert out.shape == (4,)
        _allclose(out, [1.0, 2.0, 3.0, 4.0])

    def test_reshape_infers_negative_one_dimension(self):
        a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        out = torch.reshape(a, (-1, 2))
        assert out.shape == (3, 2)
        _allclose(out, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_reshape_numel_invariant(self):
        a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        out = torch.reshape(a, (2, 3))
        assert out.shape[0] * out.shape[1] == 6


# ===========================================================================
# 7. neg wrapper
# ===========================================================================

class TestNegWrapper:

    def test_neg_basic(self):
        a = torch.tensor([1.0, -2.0, 3.0])
        out = torch.neg(a)
        _allclose(out, [-1.0, 2.0, -3.0])

    def test_neg_double_negation(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        out = torch.neg(torch.neg(a))
        _allclose(out, [1.0, 2.0, 3.0])

    def test_neg_returns_tensor(self):
        a = torch.tensor([1.0, 2.0])
        assert isinstance(torch.neg(a), Tensor)

    def test_neg_shape_preserved(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        out = torch.neg(a)
        assert out.shape == (2, 2)

    def test_neg_preserves_grad_fn(self):
        a = torch.tensor([1.0, 2.0])
        a.requires_grad = True
        out = torch.neg(a)
        assert out.grad_fn is not None

    def test_neg_backward(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        a.requires_grad = True
        out = torch.neg(a)
        out.sum().backward()
        _allclose(a.grad, [-1.0, -1.0, -1.0])


# ===========================================================================
# 8. __torch_function__ / _has_torch_function / _handle_torch_function
# ===========================================================================

class _TrackedTensor(Tensor):
    """Subclass that records which func was dispatched and unwraps for compute."""
    _log = []

    def __init__(self, data):
        if isinstance(data, Tensor):
            super().__init__(
                data._storage, data.shape, data.stride, data.offset, data.requires_grad
            )
            self._wrapped = data
        else:
            raise TypeError("_TrackedTensor requires a Tensor")

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        cls._log.append(func)
        def _unwrap(x):
            if isinstance(x, _TrackedTensor):
                return x._wrapped
            if isinstance(x, (list, tuple)):
                return type(x)(_unwrap(i) for i in x)
            return x
        new_args = tuple(_unwrap(a) for a in args)
        new_kwargs = {k: _unwrap(v) for k, v in (kwargs or {}).items()}
        return func(*new_args, **new_kwargs)

    @classmethod
    def reset(cls):
        cls._log.clear()


# ---------------------------------------------------------------------------
# NOTE: When the Cython fast-path is active (post-autograd-cython build),
# torch.add/mul/matmul/relu/neg are replaced by cyfunction wrappers that
# call the Python _py_<op> function for __torch_function__ dispatch.
# Therefore __torch_function__ receives _py_<op>, NOT the cyfunction.
# Tests use _log_nonempty() to stay identity-agnostic and correct on both
# pure-Python and Cython-accelerated builds.
# ---------------------------------------------------------------------------

class TestTorchFunctionPath:

    def setup_method(self):
        _TrackedTensor.reset()

    def test_has_torch_function_detects_subclass(self):
        from candle._functional import _has_torch_function
        t = _TrackedTensor(torch.tensor([1.0, 2.0]))
        assert _has_torch_function((t,), {}) is True

    def test_has_torch_function_false_for_base_tensor(self):
        from candle._functional import _has_torch_function
        t = torch.tensor([1.0, 2.0])
        assert _has_torch_function((t,), {}) is False

    def test_handle_torch_function_calls_subclass(self):
        from candle._functional import _handle_torch_function, _py_add
        t = _TrackedTensor(torch.tensor([1.0, 2.0]))
        # Pass the Python-level add (_py_add) so the log check is stable
        # regardless of whether the Cython shim replaced torch.add.
        result = _handle_torch_function(_py_add, (t, torch.tensor([1.0, 2.0])), {})
        assert result is not NotImplemented
        assert _py_add in _TrackedTensor._log

    def test_handle_torch_function_returns_not_implemented_for_base(self):
        from candle._functional import _handle_torch_function, _py_add
        a = torch.tensor([1.0])
        b = torch.tensor([2.0])
        result = _handle_torch_function(_py_add, (a, b), {})
        assert result is NotImplemented

    def test_add_routes_through_torch_function(self):
        """torch.add with a subclass arg must invoke __torch_function__."""
        t = _TrackedTensor(torch.tensor([1.0, 2.0]))
        out = torch.add(t, torch.tensor([3.0, 4.0]))
        # The log must be non-empty: something was routed through __torch_function__.
        assert len(_TrackedTensor._log) > 0, "__torch_function__ was not called"
        _allclose(out, [4.0, 6.0])

    def test_mul_routes_through_torch_function(self):
        t = _TrackedTensor(torch.tensor([2.0, 3.0]))
        out = torch.mul(t, torch.tensor([4.0, 5.0]))
        assert len(_TrackedTensor._log) > 0, "__torch_function__ was not called"
        _allclose(out, [8.0, 15.0])

    def test_matmul_routes_through_torch_function(self):
        t = _TrackedTensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        out = torch.matmul(t, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        assert len(_TrackedTensor._log) > 0, "__torch_function__ was not called"
        assert out.shape == (2, 2)

    def test_relu_routes_through_torch_function(self):
        t = _TrackedTensor(torch.tensor([-1.0, 1.0]))
        out = torch.relu(t)
        assert len(_TrackedTensor._log) > 0, "__torch_function__ was not called"
        _allclose(out, [0.0, 1.0])

    def test_neg_routes_through_torch_function(self):
        t = _TrackedTensor(torch.tensor([1.0, -2.0]))
        out = torch.neg(t)
        assert len(_TrackedTensor._log) > 0, "__torch_function__ was not called"
        _allclose(out, [-1.0, 2.0])

    def test_add_alpha_still_routes_through_torch_function(self):
        """alpha= branch must not skip __torch_function__ check."""
        t = _TrackedTensor(torch.tensor([1.0, 2.0]))
        out = torch.add(t, torch.tensor([1.0, 1.0]), alpha=3)
        assert len(_TrackedTensor._log) > 0, "__torch_function__ was not called for alpha= path"
        # result must still be correct: 1+3*1=4, 2+3*1=5
        _allclose(out, [4.0, 5.0])

    def test_plain_add_skips_torch_function(self):
        """Plain base Tensor args must NOT invoke __torch_function__."""
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0])
        out = torch.add(a, b)
        # log still empty (reset in setup_method)
        assert _TrackedTensor._log == []
        _allclose(out, [4.0, 6.0])

    def test_subclass_in_kwargs_detected(self):
        from candle._functional import _has_torch_function
        t = _TrackedTensor(torch.tensor([1.0]))
        assert _has_torch_function((), {"x": t}) is True

    def test_subclass_in_nested_list_detected(self):
        from candle._functional import _has_torch_function
        t = _TrackedTensor(torch.tensor([1.0]))
        assert _has_torch_function(([t],), {}) is True
