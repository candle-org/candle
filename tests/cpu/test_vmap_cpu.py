"""CPU tests for torch.vmap compatibility."""

import numpy as np
import pytest
import candle as torch


def test_vmap_maps_single_tensor_arg_cpu():
    x = torch.tensor([1.0, 2.0, 3.0])

    out = torch.vmap(lambda value: value + 1.0)(x)

    assert out.shape == x.shape
    np.testing.assert_allclose(out.numpy(), np.array([2.0, 3.0, 4.0], dtype=np.float32))


def test_vmap_nested_broadcasts_none_inputs_cpu():
    def causal_mask(_batch_idx, _head_idx, q_idx, kv_idx):
        return kv_idx <= q_idx

    mapped = causal_mask
    for dims in (
        (None, None, None, 0),
        (None, None, 0, None),
        (None, 0, None, None),
        (0, None, None, None),
    ):
        mapped = torch.vmap(mapped, in_dims=dims, out_dims=0)

    out = mapped(torch.arange(1), torch.arange(1), torch.arange(2), torch.arange(4))

    expected = np.array([[[[True, False, False, False], [True, True, False, False]]]])
    assert out.shape == expected.shape
    np.testing.assert_array_equal(out.numpy(), expected)


def test_vmap_tuple_outputs_accept_tuple_out_dims_cpu():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    first, second = torch.vmap(lambda row: (row, row + 1), out_dims=(0, 1))(x)

    assert first.shape == x.shape
    assert second.shape == (2, 2)
    np.testing.assert_allclose(first.numpy(), x.numpy())
    np.testing.assert_allclose(second.numpy(), np.array([[2.0, 4.0], [3.0, 5.0]], dtype=np.float32))


def test_vmap_zero_size_mapped_dim_cpu():
    x = torch.empty((0, 3), dtype=torch.float32)

    out = torch.vmap(lambda row: row + 1)(x)

    assert out.shape == (0, 3)
    assert out.dtype == torch.float32


def test_vmap_rejects_mismatched_output_structure_cpu():
    x = torch.tensor([0, 1])

    def fn(value):
        if value.item() == 0:
            return (value, value)
        return (value,)

    with pytest.raises(RuntimeError, match="vmap output structure"):
        torch.vmap(fn)(x)
