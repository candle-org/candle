import numpy as np
import pytest

import candle as torch
from candle._dispatch.dispatcher import dispatch


def test_detach_shares_version_counter_with_source():
    base = torch.tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
    detached = base.detach()

    before_detached = detached._version_counter.value
    base._version_counter.bump()
    assert detached._version_counter.value == before_detached + 1

    before_base = base._version_counter.value
    detached._version_counter.bump()
    assert base._version_counter.value == before_base + 1


def test_as_strided_view_shares_version_counter_with_source():
    base = torch.tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
    view = base.as_strided((2, 2), (2, 1))

    before_view = view._version_counter.value
    base._version_counter.bump()
    assert view._version_counter.value == before_view + 1

    before_base = base._version_counter.value
    view._version_counter.bump()
    assert base._version_counter.value == before_base + 1


def test_view_runtime_truth_sets_base_to_source_tensor():
    base = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    view = base.view((4,))
    assert view._base is base


def test_as_strided_runtime_truth_sets_base_to_source_tensor():
    base = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    view = base.as_strided((2, 2), (2, 1))
    assert view._base is base


def test_view_runtime_truth_records_view_meta():
    base = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    view = base.view((4,))
    assert view._view_meta is not None
    assert view._view_meta["op"] == "view"


def test_unary_inplace_preserves_view_aliasing():
    x = torch.tensor([[-1.0, 2.0], [-3.0, 4.0]])
    v = x.view((4,))
    out = x.abs_()
    assert out is x
    assert v.tolist() == [1.0, 2.0, 3.0, 4.0]


@pytest.mark.parametrize(
    ("method_name", "values"),
    [
        ("neg_", [-1.0, 2.0, -3.0, 4.0]),
        ("exp_", [0.0, 1.0, 2.0, 3.0]),
        ("log_", [1.0, 2.0, 4.0, 8.0]),
        ("sqrt_", [1.0, 4.0, 9.0, 16.0]),
        ("sin_", [0.0, 0.5, 1.0, 1.5]),
        ("cos_", [0.0, 0.5, 1.0, 1.5]),
        ("tan_", [0.0, 0.5, 1.0, 1.5]),
        ("tanh_", [-2.0, -1.0, 1.0, 2.0]),
        ("sigmoid_", [-2.0, -1.0, 1.0, 2.0]),
        ("log2_", [1.0, 2.0, 4.0, 8.0]),
        ("log10_", [1.0, 10.0, 100.0, 1000.0]),
        ("floor_", [1.2, -1.2, 3.9, -3.9]),
        ("ceil_", [1.2, -1.2, 3.1, -3.1]),
        ("round_", [1.2, -1.2, 3.6, -3.6]),
        ("trunc_", [1.2, -1.2, 3.6, -3.6]),
        ("reciprocal_", [1.0, 2.0, 4.0, 8.0]),
    ],
)
def test_cpu_unary_inplace_supported_ops_preserve_view_aliasing(method_name, values):
    x = torch.tensor([values[:2], values[2:]], dtype=torch.float32)
    v = x.view((4,))

    out = getattr(x, method_name)()
    expected = getattr(torch, method_name[:-1])(torch.tensor(values, dtype=torch.float32))

    assert out is x
    np.testing.assert_allclose(v.numpy(), expected.numpy(), atol=1e-6, rtol=1e-6)


def test_pow_inplace_preserves_view_aliasing():
    x = torch.tensor([[1.0, 2.0], [4.0, 8.0]], dtype=torch.float32)
    v = x.view((4,))
    out = x.pow_(2.0)
    expected = torch.pow(torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float32), 2.0)
    assert out is x
    np.testing.assert_allclose(v.numpy(), expected.numpy(), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    ("method_name", "lhs", "rhs", "expected"),
    [
        ("bitwise_and_", [1, 3, 7, 15], [1, 1, 3, 7], [1, 1, 3, 7]),
        ("bitwise_or_", [1, 2, 4, 8], [1, 1, 2, 4], [1, 3, 6, 12]),
        ("bitwise_xor_", [1, 3, 7, 15], [1, 1, 3, 7], [0, 2, 4, 8]),
    ],
)
def test_bitwise_inplace_preserves_view_aliasing(method_name, lhs, rhs, expected):
    x = torch.tensor([lhs[:2], lhs[2:]], dtype=torch.int64)
    other = torch.tensor([rhs[:2], rhs[2:]], dtype=torch.int64)
    v = x.view((4,))
    out = getattr(x, method_name)(other)
    assert out is x
    assert v.tolist() == expected



def test_diagonal_view_shares_version_counter_with_source():
    base = torch.tensor([[0.0, 1.0], [2.0, 3.0]], requires_grad=True)
    view = base.diagonal()

    before_view = view._version_counter.value
    base._version_counter.bump()
    assert view._version_counter.value == before_view + 1

    before_base = base._version_counter.value
    view._version_counter.bump()
    assert base._version_counter.value == before_base + 1


def test_movedim_view_shares_version_counter_with_source():
    base = torch.tensor([[0.0, 1.0], [2.0, 3.0]], requires_grad=True)
    view = base.movedim(0, 1)

    before_view = view._version_counter.value
    base._version_counter.bump()
    assert view._version_counter.value == before_view + 1

    before_base = base._version_counter.value
    view._version_counter.bump()
    assert base._version_counter.value == before_base + 1


def test_moveaxis_view_shares_version_counter_with_source():
    base = torch.tensor([[0.0, 1.0], [2.0, 3.0]], requires_grad=True)
    view = base.moveaxis(0, 1)

    before_view = view._version_counter.value
    base._version_counter.bump()
    assert view._version_counter.value == before_view + 1

    before_base = base._version_counter.value
    view._version_counter.bump()
    assert base._version_counter.value == before_base + 1


def test_expand_view_shares_version_counter_with_source():
    base = torch.tensor([[0.0, 1.0]], requires_grad=True)
    view = base.expand((3, 2))

    before_view = view._version_counter.value
    base._version_counter.bump()
    assert view._version_counter.value == before_view + 1

    before_base = base._version_counter.value
    view._version_counter.bump()
    assert base._version_counter.value == before_base + 1


def test_broadcast_to_view_shares_version_counter_with_source():
    base = torch.tensor([[0.0, 1.0]], requires_grad=True)
    view = torch.broadcast_to(base, (3, 2))

    before_view = view._version_counter.value
    base._version_counter.bump()
    assert view._version_counter.value == before_view + 1

    before_base = base._version_counter.value
    view._version_counter.bump()
    assert base._version_counter.value == before_base + 1


def test_split_view_shares_version_counter_with_source():
    base = torch.tensor([[0.0, 1.0], [2.0, 3.0]], requires_grad=True)
    view = base.split(1, dim=0)[0]

    before_view = view._version_counter.value
    base._version_counter.bump()
    assert view._version_counter.value == before_view + 1

    before_base = base._version_counter.value
    view._version_counter.bump()
    assert base._version_counter.value == before_base + 1


def test_chunk_view_shares_version_counter_with_source():
    base = torch.tensor([[0.0, 1.0], [2.0, 3.0]], requires_grad=True)
    view = base.chunk(2, dim=0)[0]

    before_view = view._version_counter.value
    base._version_counter.bump()
    assert view._version_counter.value == before_view + 1

    before_base = base._version_counter.value
    view._version_counter.bump()
    assert base._version_counter.value == before_base + 1

# Guards the Python __setitem__ entry point; the direct dispatch path is covered
# by test_dispatch_setitem_bumps_version_counter_exactly_once below.
def test_setitem_bumps_version_counter():
    x = torch.tensor([1.0, 2.0, 3.0])
    before = x._version_counter.value
    x[0] = torch.tensor(9.0)
    assert x._version_counter.value == before + 1


def test_data_setter_bumps_version_counter_once():
    x = torch.tensor([1.0, 2.0])
    before = x._version_counter.value

    x.data = torch.tensor([3.0, 4.0])

    assert x._version_counter.value == before + 1
    assert x.tolist() == [3.0, 4.0]


def test_dispatch_setitem_bumps_version_counter_exactly_once():
    x = torch.tensor([1.0, 2.0, 3.0])
    before = x._version_counter.value
    dispatch("setitem", x.device.type, x, 0, torch.tensor(9.0))
    assert x._version_counter.value == before + 1


def test_view_setitem_bumps_shared_version_counter_once():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    v = x.view((4,))
    before_base = x._version_counter.value
    before_view = v._version_counter.value

    v[0] = torch.tensor(10.0)

    assert x._version_counter.value == before_base + 1
    assert v._version_counter.value == before_view + 1
    assert x.tolist() == [[10.0, 2.0], [3.0, 4.0]]
    assert v.tolist() == [10.0, 2.0, 3.0, 4.0]


def test_set_bumps_version_counter_once():
    x = torch.tensor([0.0, 1.0, 2.0, 3.0])
    before = x._version_counter.value

    out = x.set_(x.storage(), 1, (2,), (1,))

    assert out is x
    assert x._version_counter.value == before + 1
    assert x.tolist() == [1.0, 2.0]


def test_set_on_view_bumps_shared_version_counter_once():
    x = torch.tensor([0.0, 1.0, 2.0, 3.0])
    v = x[1:]
    before_base = x._version_counter.value
    before_view = v._version_counter.value

    out = v.set_(x.storage(), 0, (2,), (1,))

    assert out is v
    assert x._version_counter.value == before_base + 1
    assert v._version_counter.value == before_view + 1
    assert v.tolist() == [0.0, 1.0]
    assert v._base is x


def test_detach_preserves_storage_shape_stride_offset_runtime_truth():
    base = torch.tensor([[0.0, 1.0], [2.0, 3.0]], requires_grad=True)
    view = base.view((4,))
    detached = view.detach()
    assert detached.storage().data_ptr() == view.storage().data_ptr()
    assert detached.shape == view.shape
    assert detached.stride == view.stride
    assert detached.offset == view.offset
    assert detached.requires_grad is False
    assert detached.grad_fn is None


def test_detach_of_view_preserves_base_version_sharing_truth():
    base = torch.tensor([[0.0, 1.0], [2.0, 3.0]], requires_grad=True)
    view = base.view((4,))
    detached = view.detach()
    before = detached._version_counter.value
    base._version_counter.bump()
    assert detached._version_counter.value == before + 1


def test_set_preserves_device_dtype_runtime_truth():
    x = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)
    out = x.set_(x.storage(), 1, (2,), (1,))
    assert out is x
    assert x.device.type == "cpu"
    assert x.dtype == torch.float32
    assert x.tolist() == [1.0, 2.0]
