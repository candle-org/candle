"""Contract tests for Tensor API forwarding methods.

Locks current behavior of the hottest Tensor methods before Cython migration.
Must stay green before and after any migration.

Coverage:
  1. __add__, __mul__, __matmul__
  2. __iadd__, __imul__
  3. clone, detach, to
  4. relu, reshape, transpose, view
  5. backward
  6. Property stability: .grad, .shape, .device, .dtype
"""
import pytest
import candle as torch
from candle._tensor import Tensor


def _allclose(t, expected, atol=1e-5):
    vals = t.flatten().tolist()
    exp = list(expected)
    assert len(vals) == len(exp), f"length mismatch: {len(vals)} vs {len(exp)}"
    for i, (x, y) in enumerate(zip(vals, exp)):
        assert abs(x - y) <= atol, f"index {i}: {x} vs {y} (atol={atol})"


# ===========================================================================
# 1. __add__
# ===========================================================================

class TestDunderAdd:

    def test_basic_values(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])
        _allclose(a + b, [5.0, 7.0, 9.0])

    def test_returns_tensor(self):
        a = torch.tensor([1.0])
        b = torch.tensor([2.0])
        assert isinstance(a + b, Tensor)

    def test_shape_preserved(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        assert (a + b).shape == (2, 2)

    def test_scalar_rhs(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        _allclose(a + 10.0, [11.0, 12.0, 13.0])

    def test_scalar_lhs_radd_not_supported(self):
        # Current behavior: float.__add__(Tensor) falls back to NotImplemented and
        # Tensor has no __radd__, so Python raises TypeError.
        # This test locks that behavior; if __radd__ is added later, update it.
        a = torch.tensor([1.0, 2.0, 3.0])
        with pytest.raises(TypeError):
            _ = 10.0 + a

    def test_does_not_modify_inputs(self):
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0])
        _ = a + b
        _allclose(a, [1.0, 2.0])
        _allclose(b, [3.0, 4.0])

    def test_grad_fn_when_requires_grad(self):
        a = torch.tensor([1.0, 2.0], requires_grad=True)
        b = torch.tensor([3.0, 4.0])
        assert (a + b).grad_fn is not None

    def test_no_grad_fn_without_requires_grad(self):
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0])
        assert (a + b).grad_fn is None


# ===========================================================================
# 2. __mul__
# ===========================================================================

class TestDunderMul:

    def test_basic_values(self):
        a = torch.tensor([2.0, 3.0, 4.0])
        b = torch.tensor([5.0, 6.0, 7.0])
        _allclose(a * b, [10.0, 18.0, 28.0])

    def test_returns_tensor(self):
        assert isinstance(torch.tensor([2.0]) * torch.tensor([3.0]), Tensor)

    def test_scalar_rhs(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        _allclose(a * 3.0, [3.0, 6.0, 9.0])

    def test_scalar_lhs_rmul(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        _allclose(3.0 * a, [3.0, 6.0, 9.0])

    def test_shape_preserved(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = torch.tensor([[2.0, 2.0], [2.0, 2.0]])
        assert (a * b).shape == (2, 2)

    def test_does_not_modify_inputs(self):
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0])
        _ = a * b
        _allclose(a, [1.0, 2.0])
        _allclose(b, [3.0, 4.0])

    def test_grad_fn_when_requires_grad(self):
        a = torch.tensor([2.0, 3.0], requires_grad=True)
        b = torch.tensor([4.0, 5.0])
        assert (a * b).grad_fn is not None


# ===========================================================================
# 3. __matmul__
# ===========================================================================

class TestDunderMatmul:

    def test_2d_identity(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        _allclose(a @ b, [1.0, 2.0, 3.0, 4.0])

    def test_returns_tensor(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        assert isinstance(a @ b, Tensor)

    def test_output_shape(self):
        a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2,3)
        b = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # (3,2)
        assert (a @ b).shape == (2, 2)

    def test_mv(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        v = torch.tensor([1.0, 1.0])
        _allclose(a @ v, [3.0, 7.0])

    def test_grad_fn_when_requires_grad(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        assert (a @ b).grad_fn is not None

    def test_rmatmul(self):
        # b.__rmatmul__(a) computes matmul(a, b) per Python's reflected op protocol
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        out = b.__rmatmul__(a)   # == matmul(a, b)
        assert isinstance(out, Tensor)
        assert out.shape == (2, 2)
        # a @ b = [[2, 1], [4, 3]], distinct from b @ a = [[3, 4], [1, 2]]
        _allclose(out, [2.0, 1.0, 4.0, 3.0])


# ===========================================================================
# 4. __iadd__
# ===========================================================================

class TestDunderIadd:

    def test_basic(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        a += torch.tensor([4.0, 5.0, 6.0])
        _allclose(a, [5.0, 7.0, 9.0])

    def test_returns_same_object(self):
        a = torch.tensor([1.0, 2.0])
        orig_id = id(a)
        a += torch.tensor([1.0, 1.0])
        assert id(a) == orig_id

    def test_scalar(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        a += 10.0
        _allclose(a, [11.0, 12.0, 13.0])

    def test_raises_for_leaf_with_requires_grad(self):
        a = torch.tensor([1.0, 2.0], requires_grad=True)
        with pytest.raises(RuntimeError):
            a += torch.tensor([1.0, 1.0])


# ===========================================================================
# 5. __imul__
# ===========================================================================

class TestDunderImul:

    def test_basic(self):
        a = torch.tensor([2.0, 3.0, 4.0])
        a *= torch.tensor([5.0, 6.0, 7.0])
        _allclose(a, [10.0, 18.0, 28.0])

    def test_returns_same_object(self):
        a = torch.tensor([1.0, 2.0])
        orig_id = id(a)
        a *= torch.tensor([2.0, 2.0])
        assert id(a) == orig_id

    def test_scalar(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        a *= 3.0
        _allclose(a, [3.0, 6.0, 9.0])

    def test_raises_for_leaf_with_requires_grad(self):
        a = torch.tensor([1.0, 2.0], requires_grad=True)
        with pytest.raises(RuntimeError):
            a *= torch.tensor([2.0, 2.0])


# ===========================================================================
# 6. clone
# ===========================================================================

class TestClone:

    def test_values_match(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        _allclose(a.clone(), [1.0, 2.0, 3.0])

    def test_returns_tensor(self):
        assert isinstance(torch.tensor([1.0, 2.0]).clone(), Tensor)

    def test_is_copy_not_alias(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = a.clone()
        b += torch.tensor([10.0, 10.0, 10.0])
        _allclose(a, [1.0, 2.0, 3.0])  # original unchanged

    def test_shape_preserved(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert a.clone().shape == (2, 2)

    def test_dtype_preserved(self):
        a = torch.tensor([1.0, 2.0])
        assert a.clone().dtype == a.dtype

    def test_device_preserved(self):
        a = torch.tensor([1.0, 2.0])
        assert a.clone().device.type == a.device.type

    def test_participates_in_autograd(self):
        a = torch.tensor([1.0, 2.0], requires_grad=True)
        b = a.clone()
        assert b.grad_fn is not None


# ===========================================================================
# 7. detach
# ===========================================================================

class TestDetach:

    def test_breaks_grad(self):
        a = torch.tensor([1.0, 2.0], requires_grad=True)
        assert a.detach().requires_grad is False

    def test_returns_tensor(self):
        assert isinstance(torch.tensor([1.0, 2.0]).detach(), Tensor)

    def test_values_match(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        _allclose(a.detach(), [1.0, 2.0, 3.0])

    def test_grad_fn_is_none(self):
        a = torch.tensor([1.0, 2.0], requires_grad=True)
        assert a.detach().grad_fn is None

    def test_shape_preserved(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert a.detach().shape == (2, 2)


# ===========================================================================
# 8. to
# ===========================================================================

class TestTo:

    def test_same_device_returns_tensor(self):
        a = torch.tensor([1.0, 2.0])
        assert isinstance(a.to("cpu"), Tensor)

    def test_same_device_values_preserved(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        _allclose(a.to("cpu"), [1.0, 2.0, 3.0])

    def test_dtype_float32_to_float64(self):
        a = torch.tensor([1.0, 2.0])
        assert a.to(torch.float64).dtype == torch.float64

    def test_dtype_values_preserved(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        _allclose(a.to(torch.float64), [1.0, 2.0, 3.0])

    def test_dtype_kwarg(self):
        a = torch.tensor([1.0, 2.0])
        assert a.to(dtype=torch.float64).dtype == torch.float64

    def test_device_kwarg_string(self):
        a = torch.tensor([1.0, 2.0])
        assert a.to(device="cpu").device.type == "cpu"

    def test_shape_preserved_after_to(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert a.to("cpu").shape == (2, 2)


# ===========================================================================
# 9. relu
# ===========================================================================

class TestRelu:

    def test_basic(self):
        a = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        _allclose(a.relu(), [0.0, 0.0, 0.0, 1.0, 2.0])

    def test_returns_tensor(self):
        assert isinstance(torch.tensor([-1.0, 1.0]).relu(), Tensor)

    def test_shape_preserved(self):
        a = torch.tensor([[1.0, -2.0], [-3.0, 4.0]])
        assert a.relu().shape == (2, 2)

    def test_nonneg_unchanged(self):
        a = torch.tensor([0.0, 1.0, 100.0])
        _allclose(a.relu(), [0.0, 1.0, 100.0])

    def test_all_negative_zeroed(self):
        a = torch.tensor([-5.0, -3.0, -1.0])
        _allclose(a.relu(), [0.0, 0.0, 0.0])

    def test_grad_fn_when_requires_grad(self):
        a = torch.tensor([-1.0, 1.0], requires_grad=True)
        assert a.relu().grad_fn is not None

    def test_does_not_modify_input(self):
        a = torch.tensor([-1.0, 2.0])
        _ = a.relu()
        _allclose(a, [-1.0, 2.0])


# ===========================================================================
# 10. reshape
# ===========================================================================

class TestReshape:

    def test_basic(self):
        a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        assert a.reshape(2, 3).shape == (2, 3)

    def test_returns_tensor(self):
        assert isinstance(torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape(2, 2), Tensor)

    def test_values_preserved(self):
        a = torch.tensor([1.0, 2.0, 3.0, 4.0])
        _allclose(a.reshape(2, 2), [1.0, 2.0, 3.0, 4.0])

    def test_tuple_arg(self):
        a = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert a.reshape((2, 2)).shape == (2, 2)

    def test_minus_one_infer(self):
        a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        assert a.reshape(2, -1).shape == (2, 3)

    def test_to_1d(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert a.reshape(4).shape == (4,)

    def test_empty_args_raises(self):
        a = torch.tensor([1.0, 2.0])
        with pytest.raises(TypeError):
            a.reshape()


# ===========================================================================
# 11. transpose
# ===========================================================================

class TestTranspose:

    def test_shape_2d(self):
        a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert a.transpose(0, 1).shape == (3, 2)

    def test_returns_tensor(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert isinstance(a.transpose(0, 1), Tensor)

    def test_values(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        out = a.transpose(0, 1)
        assert out.flatten().tolist() == [1.0, 3.0, 2.0, 4.0]

    def test_same_dims_noop_shape(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert a.transpose(0, 0).shape == a.shape

    def test_negative_dims(self):
        a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert a.transpose(-2, -1).shape == (3, 2)


# ===========================================================================
# 12. view
# ===========================================================================

class TestView:

    def test_basic(self):
        a = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert a.view(2, 2).shape == (2, 2)

    def test_returns_tensor(self):
        assert isinstance(torch.tensor([1.0, 2.0, 3.0, 4.0]).view(2, 2), Tensor)

    def test_values_preserved(self):
        a = torch.tensor([1.0, 2.0, 3.0, 4.0])
        _allclose(a.view(2, 2), [1.0, 2.0, 3.0, 4.0])

    def test_tuple_arg(self):
        a = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert a.view((2, 2)).shape == (2, 2)

    def test_minus_one_infer(self):
        a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        assert a.view(2, -1).shape == (2, 3)

    def test_empty_args_raises(self):
        a = torch.tensor([1.0, 2.0])
        with pytest.raises(TypeError):
            a.view()


# ===========================================================================
# 13. backward
# ===========================================================================

class TestBackward:

    def test_scalar_no_grad_arg(self):
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        (x * x).sum().backward()
        assert x.grad.flatten().tolist() == [2.0, 4.0]

    def test_accumulates_grad(self):
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        (x * x).sum().backward()
        first = x.grad.flatten().tolist()[:]
        assert first == [2.0, 4.0]
        (x * x).sum().backward()
        second = x.grad.flatten().tolist()
        # gradients accumulate exactly: second pass adds another [2.0, 4.0]
        assert second == [4.0, 8.0]
    def test_non_scalar_without_grad_raises(self):
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        with pytest.raises(RuntimeError):
            x.backward()

    def test_non_scalar_with_gradient_arg(self):
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        y = x * torch.tensor([3.0, 4.0])
        y.backward(torch.ones(2))
        assert x.grad.flatten().tolist() == [3.0, 4.0]

    def test_retain_graph(self):
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        y = (x * x).sum()
        y.backward(retain_graph=True)
        y.backward(retain_graph=True)  # must not raise
        assert x.grad is not None

    def test_grad_populated_on_leaf(self):
        x = torch.tensor([3.0], requires_grad=True)
        (x * x).sum().backward()
        assert x.grad is not None


# ===========================================================================
# 14. Property/attribute stability: .grad, .shape, .device, .dtype
# ===========================================================================

class TestPropertyStability:

    def test_shape_is_tuple(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert isinstance(a.shape, tuple)
        assert a.shape == (2, 2)

    def test_shape_1d(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        assert a.shape == (3,)

    def test_shape_scalar(self):
        a = torch.tensor(5.0)
        assert a.shape == ()

    def test_device_has_type(self):
        a = torch.tensor([1.0])
        assert hasattr(a.device, 'type')
        assert isinstance(a.device.type, str)

    def test_device_cpu(self):
        a = torch.tensor([1.0])
        assert a.device.type == 'cpu'

    def test_dtype_is_float32_by_default(self):
        a = torch.tensor([1.0, 2.0])
        assert a.dtype == torch.float32

    def test_dtype_int32(self):
        a = torch.tensor([1, 2, 3], dtype=torch.int32)
        assert a.dtype == torch.int32

    def test_grad_initially_none(self):
        a = torch.tensor([1.0, 2.0])
        assert a.grad is None

    def test_grad_set_after_backward(self):
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        (x * x).sum().backward()
        assert x.grad is not None

    def test_grad_delete_resets_to_none(self):
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        (x * x).sum().backward()
        del x.grad
        assert x.grad is None

    def test_requires_grad_false_by_default(self):
        a = torch.tensor([1.0, 2.0])
        assert a.requires_grad is False

    def test_requires_grad_true_when_set(self):
        a = torch.tensor([1.0, 2.0], requires_grad=True)
        assert a.requires_grad is True

    def test_ndim_matches_shape_len(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert a.ndim == len(a.shape)



class TestTensorMethodTorchFunctionPath:

    def test_subclass_add_uses_torch_function(self):
        class TrackedTensor(Tensor):
            _ops_called = []

            def __init__(self, data):
                super().__init__(data._storage, data.shape, data.stride, data.offset, data.requires_grad)
                self._wrapped = data

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls._ops_called.append(func.__name__ if hasattr(func, '__name__') else str(func))
                def unwrap(x):
                    if isinstance(x, TrackedTensor):
                        return x._wrapped
                    return x
                new_args = tuple(unwrap(a) for a in args)
                new_kwargs = {k: unwrap(v) for k, v in (kwargs or {}).items()}
                return func(*new_args, **new_kwargs)

        TrackedTensor._ops_called = []
        a = TrackedTensor(torch.tensor([1.0, 2.0]))
        b = torch.tensor([3.0, 4.0])
        out = a + b
        assert len(TrackedTensor._ops_called) > 0
        _allclose(out, [4.0, 6.0])


def test_size_and_dim_contracts():
    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert a.size() == (2, 3)
    assert a.size(0) == 2
    assert a.size(-1) == 3
    assert a.dim() == 2
    with pytest.raises(IndexError):
        a.size(2)
