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
import builtins

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
        a = torch.tensor([1.0, 2.0], device="cpu")
        assert a.clone().device.type == "cpu"

    def test_grad_fn_when_requires_grad(self):
        a = torch.tensor([1.0, 2.0], requires_grad=True)
        assert a.clone().grad_fn is not None


# ===========================================================================
# 7. detach
# ===========================================================================

class TestDetach:

    def test_detach_returns_tensor(self):
        a = torch.tensor([1.0, 2.0], requires_grad=True)
        out = a.detach()
        assert isinstance(out, Tensor)

    def test_detach_disables_requires_grad(self):
        a = torch.tensor([1.0, 2.0], requires_grad=True)
        assert a.detach().requires_grad is False

    def test_detach_preserves_values(self):
        a = torch.tensor([1.0, 2.0], requires_grad=True)
        _allclose(a.detach(), [1.0, 2.0])

    def test_detach_has_no_grad_fn(self):
        a = torch.tensor([1.0, 2.0], requires_grad=True)
        assert a.detach().grad_fn is None

    def test_detach_shape_preserved(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        assert a.detach().shape == (2, 2)


# ===========================================================================
# 8. to
# ===========================================================================

class TestTo:

    def test_returns_tensor(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        assert isinstance(a.to("cpu"), Tensor)

    def test_cpu_values_preserved(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        _allclose(a.to("cpu"), [1.0, 2.0, 3.0])

    def test_dtype_change(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        assert a.to(torch.float64).dtype == torch.float64

    def test_dtype_change_values(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        _allclose(a.to(torch.float64), [1.0, 2.0, 3.0])

    def test_dtype_kwarg(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        assert a.to(dtype=torch.float64).dtype == torch.float64

    def test_device_kwarg(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        assert a.to(device="cpu").device.type == "cpu"

    def test_shape_preserved_after_to(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert a.to("cpu").shape == (2, 2)

    def test_channels_last_non_4d_is_false_for_is_contiguous(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        assert a.is_contiguous(memory_format=torch.channels_last) is False

    def test_channels_last_non_4d_contiguous_raises(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        with pytest.raises(RuntimeError, match="required rank 4 tensor"):
            a.contiguous(memory_format=torch.channels_last)

    def test_channels_last_non_4d_to_raises(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        with pytest.raises(RuntimeError, match="required rank 4 tensor"):
            a.to(memory_format=torch.channels_last)

    def test_channels_last_4d_contiguous_and_stride(self):
        a = torch.randn(2, 3, 4, 5)
        assert a.is_contiguous() is True
        assert a.is_contiguous(memory_format=torch.channels_last) is False
        b = a.contiguous(memory_format=torch.channels_last)
        assert b.shape == (2, 3, 4, 5)
        assert b.stride() == (60, 1, 15, 3)
        assert b.is_contiguous() is False
        assert b.is_contiguous(memory_format=torch.channels_last) is True
        _allclose(b.contiguous(), a.flatten().tolist())

    def test_to_channels_last_4d_contiguous_and_stride(self):
        a = torch.randn(2, 3, 4, 5)
        b = a.to(memory_format=torch.channels_last)
        assert b.shape == (2, 3, 4, 5)
        assert b.stride() == (60, 1, 15, 3)
        assert b.is_contiguous(memory_format=torch.channels_last) is True


class TestCreationMemoryFormat:

    def test_empty_supports_channels_last(self):
        x = torch.empty((2, 3, 4, 5), memory_format=torch.channels_last)
        assert x.stride() == (60, 1, 15, 3)
        assert x.is_contiguous() is False
        assert x.is_contiguous(memory_format=torch.channels_last) is True

    def test_empty_rejects_preserve_format(self):
        with pytest.raises(TypeError, match="memory_format"):
            torch.empty((2, 3, 4, 5), memory_format=torch.preserve_format)

    def test_creation_wrappers_do_not_define_python_memory_format_validator(self):
        assert hasattr(torch._functional, "_unsupported_memory_format") is False

        out = torch.empty((2, 3, 4, 5), memory_format=torch.channels_last)
        assert out.stride() == (60, 1, 15, 3)
        assert out.is_contiguous(memory_format=torch.channels_last) is True

    def test_empty_like_preserves_channels_last_by_default(self):
        base = torch.empty((2, 3, 4, 5), memory_format=torch.channels_last)
        out = torch.empty_like(base)
        assert out.stride() == (60, 1, 15, 3)
        assert out.is_contiguous(memory_format=torch.channels_last) is True

    def test_full_like_preserves_channels_last_by_default(self):
        base = torch.empty((2, 3, 4, 5), memory_format=torch.channels_last)
        out = torch.full_like(base, 1.0)
        assert out.stride() == (60, 1, 15, 3)
        assert out.is_contiguous(memory_format=torch.channels_last) is True

    def test_empty_like_contiguous_format_from_channels_last(self):
        base = torch.empty((2, 3, 4, 5), memory_format=torch.channels_last)
        out = torch.empty_like(base, memory_format=torch.contiguous_format)
        assert out.stride() == (60, 20, 5, 1)
        assert out.is_contiguous() is True
        assert out.is_contiguous(memory_format=torch.channels_last) is False

    def test_zeros_supports_channels_last(self):
        x = torch.zeros((2, 3, 4, 5), memory_format=torch.channels_last)
        assert x.stride() == (60, 1, 15, 3)
        assert x.is_contiguous(memory_format=torch.channels_last) is True

    def test_ones_supports_channels_last(self):
        x = torch.ones((2, 3, 4, 5), memory_format=torch.channels_last)
        assert x.stride() == (60, 1, 15, 3)
        assert x.is_contiguous(memory_format=torch.channels_last) is True

    def test_randn_supports_channels_last(self):
        x = torch.randn((2, 3, 4, 5), memory_format=torch.channels_last)
        assert x.stride() == (60, 1, 15, 3)
        assert x.is_contiguous(memory_format=torch.channels_last) is True

    def test_rand_supports_channels_last(self):
        x = torch.rand((2, 3, 4, 5), memory_format=torch.channels_last)
        assert x.stride() == (60, 1, 15, 3)
        assert x.is_contiguous(memory_format=torch.channels_last) is True

    def test_randint_supports_channels_last(self):
        x = torch.randint(10, (2, 3, 4, 5), memory_format=torch.channels_last)
        assert x.stride() == (60, 1, 15, 3)
        assert x.is_contiguous(memory_format=torch.channels_last) is True

    def test_full_supports_channels_last(self):
        x = torch.full((2, 3, 4, 5), 1.0, memory_format=torch.channels_last)
        assert x.stride() == (60, 1, 15, 3)
        assert x.is_contiguous(memory_format=torch.channels_last) is True

    def test_meta_full_supports_channels_last(self):
        x = torch.full((2, 3, 4, 5), 1.0, device="meta", memory_format=torch.channels_last)
        assert x.stride() == (60, 1, 15, 3)
        assert x.is_contiguous(memory_format=torch.channels_last) is True

    def test_ones_like_preserves_channels_last_by_default(self):
        base = torch.empty((2, 3, 4, 5), memory_format=torch.channels_last)
        out = torch.ones_like(base)
        assert out.stride() == (60, 1, 15, 3)
        assert out.is_contiguous(memory_format=torch.channels_last) is True

    def test_randn_like_preserves_channels_last_by_default(self):
        base = torch.empty((2, 3, 4, 5), memory_format=torch.channels_last)
        out = torch.randn_like(base)
        assert out.stride() == (60, 1, 15, 3)
        assert out.is_contiguous(memory_format=torch.channels_last) is True

    def test_ones_like_contiguous_format_from_channels_last(self):
        base = torch.empty((2, 3, 4, 5), memory_format=torch.channels_last)
        out = torch.ones_like(base, memory_format=torch.contiguous_format)
        assert out.stride() == (60, 20, 5, 1)
        assert out.is_contiguous() is True

    def test_rand_like_contiguous_format_from_channels_last(self):
        base = torch.empty((2, 3, 4, 5), memory_format=torch.channels_last)
        out = torch.rand_like(base, memory_format=torch.contiguous_format)
        assert out.stride() == (60, 20, 5, 1)
        assert out.is_contiguous() is True

    def test_randint_like_preserves_channels_last_by_default(self):
        base = torch.empty((2, 3, 4, 5), memory_format=torch.channels_last)
        out = torch.randint_like(base, high=10)
        assert out.stride() == (60, 1, 15, 3)
        assert out.is_contiguous(memory_format=torch.channels_last) is True

    def test_like_creation_memory_format_is_applied_during_creation(self, monkeypatch):
        base = torch.empty((2, 3, 4, 5), memory_format=torch.channels_last)

        def _fail_clone(*args, **kwargs):
            raise AssertionError("like creation should not relayout by cloning after allocation")

        monkeypatch.setattr(type(base), "clone", _fail_clone)

        out = torch.zeros_like(base)
        assert out.stride() == (60, 1, 15, 3)
        assert out.is_contiguous(memory_format=torch.channels_last) is True

    def test_like_creation_does_not_resolve_preserve_format_in_python(self, monkeypatch):
        base = torch.empty((2, 3, 4, 5), memory_format=torch.channels_last)

        def _fail_is_contiguous(*args, **kwargs):
            raise AssertionError("like creation should leave memory_format policy to creation backends")

        monkeypatch.setattr(type(base), "is_contiguous", _fail_is_contiguous)

        out = torch.zeros_like(base)
        assert out.stride() == (60, 1, 15, 3)

    def test_meta_random_like_preserves_channels_last_by_default(self):
        base = torch.empty((2, 3, 4, 5), device="meta", memory_format=torch.channels_last)
        for create in (torch.rand_like, torch.randn_like, lambda x: torch.randint_like(x, high=10)):
            out = create(base)
            assert out.stride() == (60, 1, 15, 3)
            assert out.is_contiguous(memory_format=torch.channels_last) is True

    def test_clone_channels_last_from_contiguous(self):
        base = torch.empty((2, 3, 4, 5))
        out = base.clone(memory_format=torch.channels_last)
        assert out.stride() == (60, 1, 15, 3)
        assert out.is_contiguous() is False
        assert out.is_contiguous(memory_format=torch.channels_last) is True

    def test_clone_preserve_format_from_channels_last(self):
        base = torch.empty((2, 3, 4, 5), memory_format=torch.channels_last)
        out = base.clone(memory_format=torch.preserve_format)
        assert out.stride() == (60, 1, 15, 3)
        assert out.is_contiguous(memory_format=torch.channels_last) is True

    def test_clone_contiguous_format_from_channels_last(self):
        base = torch.empty((2, 3, 4, 5), memory_format=torch.channels_last)
        out = base.clone(memory_format=torch.contiguous_format)
        assert out.stride() == (60, 20, 5, 1)
        assert out.is_contiguous() is True
        assert out.is_contiguous(memory_format=torch.channels_last) is False

    def test_basic(self):
        a = torch.tensor([-1.0, 0.0, 2.0])
        _allclose(a.relu(), [0.0, 0.0, 2.0])

    def test_returns_tensor(self):
        assert isinstance(torch.tensor([-1.0]).relu(), Tensor)

    def test_shape_preserved(self):
        a = torch.tensor([[-1.0, 2.0], [3.0, -4.0]])
        assert a.relu().shape == (2, 2)

    def test_does_not_modify_original(self):
        a = torch.tensor([-1.0, 2.0])
        _ = a.relu()
        _allclose(a, [-1.0, 2.0])

    def test_grad_fn_when_requires_grad(self):
        a = torch.tensor([-1.0, 2.0], requires_grad=True)
        assert a.relu().grad_fn is not None


# ===========================================================================
# 10. reshape
# ===========================================================================

class TestReshape:

    def test_basic_shape(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert a.reshape(4).shape == (4,)

    def test_values_preserved(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        _allclose(a.reshape(4), [1.0, 2.0, 3.0, 4.0])

    def test_returns_tensor(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert isinstance(a.reshape(4), Tensor)

    def test_infer_dimension(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert a.reshape(-1).shape == (4,)

    def test_grad_fn_when_requires_grad(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        assert a.reshape(4).grad_fn is not None


# ===========================================================================
# 11. transpose
# ===========================================================================

class TestTranspose:

    def test_shape_swapped(self):
        a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert a.transpose(0, 1).shape == (3, 2)

    def test_values_transposed(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        _allclose(a.transpose(0, 1), [1.0, 3.0, 2.0, 4.0])

    def test_returns_tensor(self):
        a = torch.tensor([[1.0, 2.0]])
        assert isinstance(a.transpose(0, 1), Tensor)

    def test_no_copy_semantics_not_required_but_shape_correct(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = a.transpose(0, 1)
        assert b.shape == (2, 2)

    def test_grad_fn_when_requires_grad(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        assert a.transpose(0, 1).grad_fn is not None


# ===========================================================================
# 12. view
# ===========================================================================

class TestView:

    def test_basic_shape(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert a.view(4).shape == (4,)

    def test_values_preserved(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        _allclose(a.view(4), [1.0, 2.0, 3.0, 4.0])

    def test_returns_tensor(self):
        a = torch.tensor([[1.0, 2.0]])
        assert isinstance(a.view(2), Tensor)

    def test_requires_contiguous_input(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = a.transpose(0, 1)
        with pytest.raises(RuntimeError):
            b.view(4)

    def test_grad_fn_when_requires_grad(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        assert a.view(4).grad_fn is not None


# ===========================================================================
# 13. backward
# ===========================================================================

class TestBackward:

    def test_scalar_backward_populates_grad(self):
        x = torch.tensor([3.0], requires_grad=True)
        y = x * x
        y.backward()
        assert x.grad is not None
        _allclose(x.grad, [6.0])

    def test_backward_with_external_gradient(self):
        x = torch.tensor([2.0, 3.0], requires_grad=True)
        y = x * 2.0
        y.backward(torch.tensor([1.0, 1.0]))
        _allclose(x.grad, [2.0, 2.0])

    def test_backward_returns_none(self):
        x = torch.tensor([3.0], requires_grad=True)
        y = x * x
        assert y.backward() is None


# ===========================================================================
# 14. property stability
# ===========================================================================

class TestPropertyStability:

    def test_shape_device_dtype_exist(self):
        x = torch.tensor([[1.0, 2.0]])
        assert x.shape == (1, 2)
        assert x.device.type == "cpu"
        assert x.dtype == torch.float32

    def test_grad_default_none(self):
        x = torch.tensor([1.0, 2.0])
        assert x.grad is None

    def test_grad_after_backward(self):
        x = torch.tensor([2.0], requires_grad=True)
        y = x * x
        y.backward()
        assert x.grad is not None

    def test_requires_grad_roundtrip(self):
        x = torch.tensor([1.0], requires_grad=True)
        assert x.requires_grad is True


# ===========================================================================
# 15. __torch_function__ forwarding path
# ===========================================================================

class TestTensorMethodTorchFunctionPath:

    class Wrapper(Tensor):
        called = None

        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            cls.called = func.__name__
            return torch.tensor([123.0])

    def setup_method(self):
        self.Wrapper.called = None

    def test___add___uses_torch_function(self):
        x = torch.tensor([1.0]).as_subclass(self.Wrapper)
        out = x + torch.tensor([2.0])
        assert self.Wrapper.called == "__add__"
        _allclose(out, [123.0])

    def test_clone_uses_torch_function(self):
        x = torch.tensor([1.0]).as_subclass(self.Wrapper)
        out = x.clone()
        assert self.Wrapper.called == "clone"
        _allclose(out, [123.0])

    def test_relu_uses_torch_function(self):
        x = torch.tensor([1.0]).as_subclass(self.Wrapper)
        out = x.relu()
        assert self.Wrapper.called == "relu"
        _allclose(out, [123.0])

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


class _MemoryFormatRecorder:
    seen = object()

    def clone(self, *, memory_format):
        self.seen = memory_format
        return self

    def contiguous(self, memory_format):
        self.seen = memory_format
        return self


def test_top_level_clone_passes_none_memory_format():
    x = _MemoryFormatRecorder()
    assert torch.clone(x) is x
    assert x.seen is None


def test_top_level_contiguous_passes_none_memory_format():
    x = _MemoryFormatRecorder()
    assert torch.contiguous(x) is x
    assert x.seen is None


def test_dim_order_uses_candle_prims_common(monkeypatch):
    real_import = builtins.__import__

    def reject_external_torch_prims(name, *args, **kwargs):
        if name == "torch._prims_common":
            raise AssertionError("dim_order should use candle._prims_common")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_external_torch_prims)

    assert torch.empty((2, 3, 4, 5)).dim_order() == (0, 1, 2, 3)


def test_dim_order_ambiguity_check_does_not_import_torch_fx(monkeypatch):
    real_import = builtins.__import__

    def reject_external_torch_fx(name, *args, **kwargs):
        if name == "torch.fx.experimental.symbolic_shapes":
            raise AssertionError("dim_order should not import torch.fx")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_external_torch_fx)

    assert torch.empty((2, 3, 4, 5)).dim_order(ambiguity_check=True) == (0, 1, 2, 3)


def test_register_post_accumulate_grad_hook_uses_candle_hooks(monkeypatch):
    real_import = builtins.__import__

    def reject_external_torch_hooks(name, *args, **kwargs):
        if name == "torch.utils.hooks":
            raise AssertionError("register_post_accumulate_grad_hook should use candle.utils.hooks")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_external_torch_hooks)

    x = torch.tensor([1.0], requires_grad=True)
    x._post_accumulate_grad_hooks = None
    handle = x.register_post_accumulate_grad_hook(lambda grad: None)
    assert handle.id in x._post_accumulate_grad_hooks
    handle.remove()
    assert handle.id not in x._post_accumulate_grad_hooks


def test_reduce_ex_uses_candle_hook_warning(monkeypatch):
    real_import = builtins.__import__

    def reject_external_torch_hooks(name, *args, **kwargs):
        if name == "torch.utils.hooks":
            raise AssertionError("Tensor.__reduce_ex__ should use candle.utils.hooks")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_external_torch_hooks)

    x = torch.tensor([1.0])
    with pytest.raises(AttributeError, match="_serialization_tls"):
        x._reduce_ex_internal(4)


def test_dlpack_device_uses_candle_enum(monkeypatch):
    real_import = builtins.__import__

    def reject_external_torch_dlpack(name, *args, **kwargs):
        if name == "torch.utils.dlpack":
            raise AssertionError("Tensor.__dlpack_device__ should not import torch.utils.dlpack")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_external_torch_dlpack)

    device_type, index = torch.tensor([1.0]).__dlpack_device__()
    assert int(device_type) == 1
    assert index == 0


def test_legacy_tensor_linalg_methods_use_candle_linalg(monkeypatch):
    real_import = builtins.__import__

    def reject_external_torch_linalg_utils(name, *args, **kwargs):
        if name == "torch._linalg_utils":
            raise AssertionError("legacy Tensor linalg methods should not import torch._linalg_utils")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_external_torch_linalg_utils)

    a = torch.tensor([[3.0, 1.0], [1.0, 2.0]])
    b = torch.tensor([[9.0], [8.0]])
    assert b.solve(a).shape == (2, 1)
    assert b.lstsq(a).solution.shape == (2, 1)
    assert a.eig(eigenvectors=True)[0].shape == (2,)
    assert a.symeig(eigenvectors=True)[0].shape == (2,)


def test_tensor_resize_uses_candle_resize_helper(monkeypatch):
    real_import = builtins.__import__

    def reject_external_torch_resize(name, *args, **kwargs):
        if name == "torch.autograd._functions":
            raise AssertionError("Tensor.resize should not import torch.autograd._functions")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_external_torch_resize)

    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = x.resize(2, 2)
    assert out.shape == (2, 2)
    assert out.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert out is not x


def test_tensor_resize_as_uses_candle_resize_helper(monkeypatch):
    real_import = builtins.__import__

    def reject_external_torch_resize(name, *args, **kwargs):
        if name == "torch.autograd._functions":
            raise AssertionError("Tensor.resize_as should not import torch.autograd._functions")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_external_torch_resize)

    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = x.resize_as(torch.empty((2, 2)))
    assert out.shape == (2, 2)
    assert out.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert out is not x


# ===========================================================================
# data_ptr
# ===========================================================================
def test_data_ptr_returns_int():
    t = torch.tensor([1.0, 2.0, 3.0])
    ptr = t.data_ptr()
    assert isinstance(ptr, int)
    assert ptr != 0


def test_data_ptr_different_tensors():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    assert a.data_ptr() != b.data_ptr()


def test_data_ptr_view_shares_base():
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = a.view(4)
    # view should share memory, so data_ptr could be the same
    # (not strictly guaranteed to be equal if offset differs, but
    # for a zero-offset view it should match)
    assert b.data_ptr() == a.data_ptr()


def test_set_rebinds_typed_storage_metadata():
    source = torch.arange(6, dtype=torch.float32)
    rebound = torch.tensor([], dtype=torch.float32)

    out = rebound.set_(source.storage(), 1, (2,), (2,))

    assert out is rebound
    assert rebound.shape == (2,)
    assert rebound.stride == (2,)
    assert rebound.storage_offset() == 1


def test_set_reads_rebound_logical_elements():
    source = torch.arange(6, dtype=torch.float32)
    rebound = torch.tensor([], dtype=torch.float32)

    rebound.set_(source.storage(), 1, (2,), (2,))

    assert rebound.tolist() == [1.0, 3.0]


def test_set_rebound_tensor_aliases_underlying_storage():
    source = torch.arange(6, dtype=torch.float32)
    rebound = torch.tensor([], dtype=torch.float32)

    rebound.set_(source.storage(), 1, (2,), (2,))
    source[3] = 99.0

    assert rebound.tolist() == [1.0, 99.0]


def test_set_rejects_out_of_bounds_view():
    source = torch.arange(6, dtype=torch.float32)
    rebound = torch.tensor([], dtype=torch.float32)

    with pytest.raises(RuntimeError):
        rebound.set_(source.storage(), 5, (2,), (1,))


def test_set_allows_empty_tensor_view_with_large_storage_offset():
    source = torch.arange(6, dtype=torch.float32)
    rebound = torch.tensor([], dtype=torch.float32)

    rebound.set_(source.storage(), 1000, (0,), (1,))

    assert rebound.shape == (0,)
    assert rebound.storage_offset() == 1000
    assert rebound.tolist() == []
