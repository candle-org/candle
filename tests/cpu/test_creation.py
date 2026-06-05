import numpy as np
import pytest
import candle as torch


def test_tensor_bfloat16_scalar_stores_numeric_value_cpu():
    x = torch.tensor(1.0, dtype=torch.bfloat16)

    assert x.dtype == torch.bfloat16
    assert x.to(torch.float32).item() == 1.0


def test_factory_bfloat16_stores_numeric_values_cpu():
    ones = torch.ones((2,), dtype=torch.bfloat16)
    full = torch.full((2,), 2.0, dtype=torch.bfloat16)

    np.testing.assert_allclose(ones.to(torch.float32).numpy(), [1.0, 1.0])
    np.testing.assert_allclose(full.to(torch.float32).numpy(), [2.0, 2.0])


def test_creation_ops():
    x = torch.zeros((2, 3))
    y = torch.ones((2, 3))
    assert x.shape == (2, 3)
    assert y.shape == (2, 3)
    assert x.storage().data.sum() == 0
    assert y.storage().data.sum() == 6


def test_creation_device_index_cpu_meta():
    cpu_tensor = torch.ones((1,), device="cpu:1")
    assert cpu_tensor.device.type == "cpu"
    assert cpu_tensor.device.index == 1

    meta_tensor = torch.ones((1,), device="meta:1")
    assert meta_tensor.device.type == "meta"
    assert meta_tensor.device.index == 1


def test_eye_cpu():
    x = torch.eye(3, 2)
    expected = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
    assert x.shape == (3, 2)
    assert x.numpy().tolist() == expected


def test_range_cpu():
    x = torch.range(0.0, 2.0, 0.5)
    expected = np.arange(0.0, 2.0 + 0.5, 0.5)
    np.testing.assert_allclose(x.numpy(), expected)


def test_arange_cpu():
    x = torch.arange(0, 5)
    assert x.shape == (5,)
    assert x.numpy().tolist() == [0, 1, 2, 3, 4]


def test_arange_rejects_non_finite_ranges_cpu():
    with pytest.raises(RuntimeError, match="unsupported range"):
        torch.arange(-5, float("nan"))
    with pytest.raises(RuntimeError, match="unsupported range"):
        torch.arange(0, float("inf"))
    with pytest.raises(RuntimeError, match="unsupported range"):
        torch.arange(float("nan"))


def test_arange_rejects_zero_step_cpu():
    with pytest.raises(RuntimeError, match="step must be nonzero"):
        torch.arange(0, 1, 0)


def test_linspace_cpu():
    x = torch.linspace(0.0, 1.0, 5)
    assert x.shape == (5,)
    assert x.numpy().tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_linspace_rejects_negative_steps_cpu():
    with pytest.raises(RuntimeError, match="number of steps must be non-negative"):
        torch.linspace(0.0, 1.0, -1)


def test_logspace_rejects_negative_steps_cpu():
    with pytest.raises(RuntimeError, match="number of steps must be non-negative"):
        torch.logspace(0.0, 1.0, -1)


def test_linspace_infers_complex_dtype_from_complex_endpoints_cpu():
    # PyTorch parity: complex start/end with dtype=None deduces complex64.
    for start, end in [(1j, 2j), (0.0, 2j), (1j, 2)]:
        assert torch.linspace(start, end, 100).dtype == torch.complex64


def test_logspace_infers_complex_dtype_from_complex_endpoints_cpu():
    for start, end in [(1j, 2j), (0.0, 2j), (1j, 2)]:
        assert torch.logspace(start, end, 100).dtype == torch.complex64


def test_linspace_rejects_complex_endpoints_with_real_dtype_cpu():
    with pytest.raises(RuntimeError, match=r"torch.linspace\(\): inferred dtype"):
        torch.linspace(0, 1j, 5, dtype=torch.float32)
    with pytest.raises(RuntimeError, match=r"torch.linspace\(\): inferred dtype"):
        torch.linspace(0j, 1, 5, dtype=torch.float32)
    with pytest.raises(RuntimeError, match=r"torch.linspace\(\): inferred dtype"):
        torch.linspace(0j, 1j, 5, dtype=torch.float32)


def test_logspace_rejects_complex_endpoints_with_real_dtype_cpu():
    with pytest.raises(RuntimeError, match=r"torch.logspace\(\): inferred dtype"):
        torch.logspace(0, 1j, 5, dtype=torch.float32)


def test_full_infers_bool_dtype_from_bool_fill_cpu():
    with torch.testing._internal.common_utils.set_default_dtype(torch.float16):
        assert torch.full((2, 2), True).dtype == torch.bool


def test_full_infers_int64_dtype_from_int_fill_cpu():
    with torch.testing._internal.common_utils.set_default_dtype(torch.float16):
        assert torch.full((2, 2), 1).dtype == torch.int64


def test_full_infers_complex_dtype_from_complex_fill_cpu():
    with torch.testing._internal.common_utils.set_default_dtype(torch.float16):
        assert torch.full((2, 2), 1 + 1j).dtype == torch.complex64
    with torch.testing._internal.common_utils.set_default_dtype(torch.float64):
        assert torch.full((2, 2), 1 + 1j).dtype == torch.complex128


def test_full_out_dtype_overrides_inference_cpu():
    out = torch.empty((5,), dtype=torch.int64)
    assert torch.full((5,), 1.0, out=out).dtype == torch.int64
    assert torch.full((5,), 1, out=out).dtype == torch.int64


def test_full_rejects_dtype_out_conflict_cpu():
    out = torch.empty((5,), dtype=torch.int64)
    with pytest.raises(RuntimeError):
        torch.full((5,), 1.0, dtype=torch.float32, out=out)


def test_full_cpu():
    x = torch.full((2, 3), 1.5)
    assert x.shape == (2, 3)
    assert x.numpy().tolist() == [[1.5, 1.5, 1.5], [1.5, 1.5, 1.5]]


def test_creation_requires_grad_float_dtype_cpu():
    x = torch.zeros((2, 3), requires_grad=True)
    assert x.requires_grad is True


def test_creation_requires_grad_complex_dtype_cpu():
    x = torch.full((2,), 1.0 + 2.0j, dtype=torch.complex64, requires_grad=True)
    assert x.requires_grad is True


def test_creation_requires_grad_rejects_integer_dtype_cpu():
    with pytest.raises(RuntimeError, match="Only Tensors of floating point and complex dtype can require gradients"):
        torch.arange(0, 5, requires_grad=True)


def test_creation_requires_grad_rejects_bool_dtype_cpu():
    with pytest.raises(RuntimeError, match="Only Tensors of floating point and complex dtype can require gradients"):
        torch.zeros((2, 3), dtype=torch.bool, requires_grad=True)


def test_rand_creation_requires_grad_float_dtype_cpu():
    x = torch.rand((2, 3), requires_grad=True)
    assert x.requires_grad is True


def test_randn_creation_requires_grad_float_dtype_cpu():
    x = torch.randn((2, 3), requires_grad=True)
    assert x.requires_grad is True


def test_rand_creation_requires_grad_rejects_integer_dtype_cpu():
    with pytest.raises(RuntimeError, match="Only Tensors of floating point and complex dtype can require gradients"):
        torch.rand((2, 3), dtype=torch.int64, requires_grad=True)


def test_randn_creation_requires_grad_rejects_bool_dtype_cpu():
    with pytest.raises(RuntimeError, match="Only Tensors of floating point and complex dtype can require gradients"):
        torch.randn((2, 3), dtype=torch.bool, requires_grad=True)


# ---------------------------------------------------------------------------
# layout= and out= kwargs (PyTorch parity)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory", [torch.zeros, torch.ones, torch.empty])
def test_factory_accepts_size_kwarg_cpu(factory):
    out = factory(size=(1, 3), dtype=torch.float32)
    assert out.shape == (1, 3)


@pytest.mark.parametrize("factory", [torch.zeros, torch.ones, torch.empty])
def test_factory_rejects_size_kwarg_with_positional_shape_cpu(factory):
    with pytest.raises(TypeError, match="received an invalid combination"):
        factory((1, 3), size=(1, 3), dtype=torch.float32)


@pytest.mark.parametrize("factory", [torch.zeros, torch.ones, torch.empty])
def test_factory_accepts_layout_strided_cpu(factory):
    t = factory((2, 3), dtype=torch.float32, layout=torch.strided)
    assert t.shape == (2, 3)
    assert t.layout is torch.strided


@pytest.mark.parametrize("factory", [torch.zeros, torch.ones, torch.empty])
def test_factory_rejects_non_strided_layout_cpu(factory):
    with pytest.raises(TypeError, match="layout"):
        factory((2, 3), layout="not-a-layout")


@pytest.mark.parametrize("factory_name", ["zeros", "ones", "empty"])
def test_factory_rejects_sparse_layout_cpu(factory_name):
    factory = getattr(torch, factory_name)
    with pytest.raises(RuntimeError, match="strided layout"):
        factory((2, 3), layout=torch.sparse_coo)


@pytest.mark.parametrize("factory_name", ["zeros", "ones", "empty"])
def test_factory_rejects_dtype_out_conflict_cpu(factory_name):
    factory = getattr(torch, factory_name)
    out = torch.empty((3, 4), dtype=torch.float32)
    with pytest.raises(RuntimeError):
        factory((3, 4), dtype=torch.int64, out=out)


def test_linspace_out_aliases_output_storage_cpu():
    out = torch.empty((4,), dtype=torch.float32)
    res = torch.linspace(0.0, 1.0, 4, out=out)
    assert res.data_ptr() == out.data_ptr()
    assert res.shape == (4,)
    np.testing.assert_allclose(res.numpy(), [0.0, 1 / 3, 2 / 3, 1.0])


def test_logspace_out_aliases_output_storage_cpu():
    out = torch.empty((4,), dtype=torch.float32)
    res = torch.logspace(0.0, 1.0, 4, out=out)
    assert res.data_ptr() == out.data_ptr()
    assert res.shape == (4,)


def test_linspace_out_rejects_memory_overlap_cpu():
    x = torch.rand((1,)).expand((10,))
    with pytest.raises(RuntimeError, match="unsupported operation"):
        torch.linspace(1, 10, 10, out=x)


def test_logspace_out_rejects_memory_overlap_cpu():
    x = torch.rand((1,)).expand((10,))
    with pytest.raises(RuntimeError, match="unsupported operation"):
        torch.logspace(1, 10, 10, out=x)


def test_ones_out_resizes_empty_output_cpu():
    out = torch.empty((0,), dtype=torch.float32)
    res = torch.ones((2, 3), out=out)
    assert res is out
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out.numpy(), np.ones((2, 3), dtype=np.float32))


def test_linspace_out_resizes_empty_output_cpu():
    out = torch.empty((0,), dtype=torch.float32)
    res = torch.linspace(0.0, 1.0, 4, out=out)
    assert res is out
    assert out.shape == (4,)
    np.testing.assert_allclose(out.numpy(), [0.0, 1 / 3, 2 / 3, 1.0])


def test_arange_out_warns_when_resizing_non_empty_output_cpu():
    out = torch.zeros(size=(1, 3), dtype=torch.float32)
    with pytest.warns(UserWarning, match="The out tensor will be resized"):
        res = torch.arange(0, 4, out=out)
    assert res is out
    assert out.shape == (4,)
    np.testing.assert_allclose(out.numpy(), [0.0, 1.0, 2.0, 3.0])


def test_arange_out_writes_non_contiguous_view_cpu():
    x = torch.zeros((2, 3), dtype=torch.float32)
    out = x.narrow(1, 1, 2)
    res = torch.arange(0, 4, out=out)
    assert res is out
    assert out.shape == (2, 2)
    np.testing.assert_allclose(x.numpy(), [[0.0, 0.0, 1.0], [0.0, 2.0, 3.0]])


def test_linspace_out_writes_non_contiguous_view_cpu():
    x = torch.zeros((2, 3), dtype=torch.float32)
    out = x.narrow(1, 1, 2)
    res = torch.linspace(0, 3, 4, out=out)
    assert res is out
    assert out.shape == (2, 2)
    np.testing.assert_allclose(x.numpy(), [[0.0, 0.0, 1.0], [0.0, 2.0, 3.0]])


def test_logspace_out_writes_non_contiguous_view_cpu():
    x = torch.zeros((2, 3), dtype=torch.float32)
    out = x.narrow(1, 1, 2)
    res = torch.logspace(0, 3, 4, base=2, out=out)
    assert res is out
    assert out.shape == (2, 2)
    np.testing.assert_allclose(x.numpy(), [[0.0, 1.0, 2.0], [0.0, 4.0, 8.0]])


@pytest.mark.parametrize("factory", [torch.rand, torch.randn])
def test_random_factory_out_aliases_output_cpu(factory):
    out = torch.empty((2, 3), dtype=torch.float32)
    res = factory((2, 3), out=out)
    assert res is out
    assert out.shape == (2, 3)


def test_randint_out_aliases_output_cpu():
    out = torch.empty((2, 3), dtype=torch.int64)
    res = torch.randint(0, 10, (2, 3), out=out)
    assert res is out
    assert out.shape == (2, 3)
    assert out.dtype == torch.int64


def test_randperm_out_aliases_output_cpu():
    out = torch.empty((0,), dtype=torch.int64)
    res = torch.randperm(5, out=out)
    assert res is out
    assert out.shape == (5,)
    assert out.dtype == torch.int64


# ---------------------------------------------------------------------------
# stack helpers with scalar inputs (PyTorch parity)
# ---------------------------------------------------------------------------


def test_hstack_accepts_scalar_inputs_cpu():
    out = torch.hstack((torch.tensor(1.0), torch.tensor(2.0)))
    assert out.shape == (2,)
    np.testing.assert_allclose(out.numpy(), [1.0, 2.0])


def test_vstack_accepts_scalar_inputs_cpu():
    out = torch.vstack((torch.tensor(1.0), torch.tensor(2.0)))
    assert out.shape == (2, 1)
    np.testing.assert_allclose(out.numpy(), [[1.0], [2.0]])


def test_dstack_accepts_scalar_inputs_cpu():
    out = torch.dstack((torch.tensor(1.0), torch.tensor(2.0)))
    assert out.shape == (1, 1, 2)
    np.testing.assert_allclose(out.numpy(), [[[1.0, 2.0]]])


def test_column_stack_accepts_scalar_inputs_cpu():
    out = torch.column_stack((torch.tensor(1.0), torch.tensor(2.0)))
    assert out.shape == (1, 2)
    np.testing.assert_allclose(out.numpy(), [[1.0, 2.0]])


def test_column_stack_accepts_mixed_1d_and_2d_cpu():
    one_dim = torch.arange(0, 3)
    two_dim = torch.arange(0, 9).reshape(3, 3)
    out = torch.column_stack((two_dim, one_dim, two_dim, one_dim))
    assert out.shape == (3, 8)
    np.testing.assert_allclose(
        out.numpy(),
        np.column_stack((two_dim.numpy(), one_dim.numpy(), two_dim.numpy(), one_dim.numpy())),
    )


@pytest.mark.parametrize("factory", [torch.hstack, torch.vstack, torch.dstack, torch.column_stack, torch.row_stack])
def test_stack_helpers_reject_single_tensor_input_cpu(factory):
    with pytest.raises(TypeError, match="must be tuple of Tensors, not Tensor"):
        factory(torch.tensor([1.0, 2.0]))


# ---------------------------------------------------------------------------
# random_ bounds (PyTorch parity)
# ---------------------------------------------------------------------------


def test_random_single_arg_uses_zero_as_lower_bound_cpu():
    t = torch.empty((200,), dtype=torch.int64)
    t.random_(4)
    assert t.min().item() == 0
    assert t.max().item() == 3


def test_random_bool_generates_false_and_true_cpu():
    t = torch.empty((2000,), dtype=torch.bool)
    t.random_()
    assert t.min().item() is False
    assert t.max().item() is True


def test_random_bool_rejects_out_of_bounds_cpu():
    t = torch.empty((8,), dtype=torch.bool)
    with pytest.raises(RuntimeError, match="from is out of bounds"):
        t.random_(-1, 1)
    with pytest.raises(RuntimeError, match="to - 1 is out of bounds"):
        t.random_(0, 3)


# ---------------------------------------------------------------------------
# missing creation APIs (PyTorch parity)
# ---------------------------------------------------------------------------


def test_vander_cpu():
    x = torch.tensor([1, 2, 3])
    out = torch.vander(x, 3)
    np.testing.assert_allclose(out.numpy(), [[1, 1, 1], [4, 2, 1], [9, 3, 1]])


def test_empty_strided_cpu():
    out = torch.empty_strided((2, 3), (1, 2))
    assert out.shape == (2, 3)
    assert out.stride() == (1, 2)


def test_chalf_alias_cpu():
    assert torch.chalf is torch.complex32


def test_kaiser_window_cpu():
    out = torch.kaiser_window(5, False, 2.0)
    np.testing.assert_allclose(out.numpy(), np.kaiser(5, 2.0), rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# asarray requires_grad propagation (PyTorch parity)
# ---------------------------------------------------------------------------


def test_asarray_clears_requires_grad_on_alias_cpu():
    t = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    r = torch.asarray(t, requires_grad=False)
    assert r.requires_grad is False
    assert r.data_ptr() == t.data_ptr()


def test_asarray_default_clears_requires_grad_on_alias_cpu():
    t = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    r = torch.asarray(t)
    assert r.requires_grad is False


def test_asarray_propagates_requires_grad_true_on_alias_cpu():
    t = torch.tensor([1.0, 2.0, 3.0], requires_grad=False)
    r = torch.asarray(t, requires_grad=True)
    assert r.requires_grad is True


def test_asarray_default_keeps_requires_grad_false_cpu():
    t = torch.tensor([1.0, 2.0, 3.0])
    r = torch.asarray(t)
    assert r.requires_grad is False
