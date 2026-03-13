"""310B-specific op tests — ops that use composite fallback on Ascend 310B."""
import numpy as np
import pytest
import candle as torch
from candle.nn import functional as F

NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()
pytestmark = pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")


def test_310b_flip():
    x = torch.tensor([[1, 2], [3, 4]], device="npu")
    out = torch.flip(x, dims=(0,))
    np.testing.assert_array_equal(out.to("cpu").numpy(), np.flip(x.to("cpu").numpy(), axis=0))


def test_310b_argsort():
    x = torch.tensor([[3.0, 1.0, 2.0], [0.0, -1.0, 5.0]], device="npu")
    out = torch.argsort(x, dim=1)
    expected = np.argsort(x.to("cpu").numpy(), axis=1)
    np.testing.assert_array_equal(out.to("cpu").numpy(), expected)


def test_310b_sort():
    x = torch.tensor([[3.0, 1.0, 2.0], [0.0, -1.0, 5.0]], device="npu")
    values, indices = torch.sort(x, dim=1)
    expected_indices = np.argsort(x.to("cpu").numpy(), axis=1)
    expected_values = np.take_along_axis(x.to("cpu").numpy(), expected_indices, axis=1)
    np.testing.assert_allclose(values.to("cpu").numpy(), expected_values, atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(indices.to("cpu").numpy(), expected_indices)


def test_310b_topk():
    x = torch.tensor([[3.0, 1.0, 2.0], [0.0, -1.0, 5.0]], device="npu")
    values, indices = torch.topk(x, k=2, dim=1, largest=True, sorted=True)
    expected_indices = np.argsort(-x.to("cpu").numpy(), axis=1)[:, :2]
    expected_values = np.take_along_axis(x.to("cpu").numpy(), expected_indices, axis=1)
    np.testing.assert_allclose(values.to("cpu").numpy(), expected_values, atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(indices.to("cpu").numpy(), expected_indices)


def test_310b_diag():
    x = torch.tensor([1.0, 2.0, 3.0], device="npu")
    out = torch.diag(x)
    np.testing.assert_allclose(out.to("cpu").numpy(), np.diag(x.to("cpu").numpy()), atol=1e-6, rtol=1e-6)


def test_310b_take_along_dim():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="npu")
    indices = torch.tensor([[0, 2, 1], [2, 0, 1]], dtype=torch.int64, device="npu")
    expected = np.take_along_axis(
        x.to("cpu").numpy(),
        indices.to("cpu").numpy().astype(np.int64),
        axis=1,
    )
    np.testing.assert_allclose(
        torch.take_along_dim(x, indices, dim=1).to("cpu").numpy(),
        expected,
    )
    neg_indices = torch.tensor([[-1, 0, 1], [1, -2, 0]], dtype=torch.int64, device="npu")
    expected_neg = np.take_along_axis(
        x.to("cpu").numpy(),
        neg_indices.to("cpu").numpy().astype(np.int64),
        axis=1,
    )
    np.testing.assert_allclose(
        torch.take_along_dim(x, neg_indices, dim=1).to("cpu").numpy(),
        expected_neg,
    )
    out_of_range = torch.tensor([[3, 0, 1], [1, 2, 0]], dtype=torch.int64, device="npu")
    with pytest.raises(IndexError):
        torch.take_along_dim(x, out_of_range, dim=1)


def test_310b_gather():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="npu")
    index = torch.tensor([[0, 2], [1, 0]], dtype=torch.int64, device="npu")
    expected = np.take_along_axis(
        x.to("cpu").numpy(),
        index.to("cpu").numpy().astype(np.int64),
        axis=1,
    )
    np.testing.assert_allclose(
        torch.gather(x, dim=1, index=index).to("cpu").numpy(),
        expected,
    )
    neg_index = torch.tensor([[0, -1], [1, 0]], dtype=torch.int64, device="npu")
    with pytest.raises(IndexError):
        torch.gather(x, dim=1, index=neg_index)
    out_of_range = torch.tensor([[3, 0], [1, 0]], dtype=torch.int64, device="npu")
    with pytest.raises(IndexError):
        torch.gather(x, dim=1, index=out_of_range)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_310b_layer_norm(dtype):

    # Test with 1-row input (works)
    data_1row = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    x_1row = torch.tensor(data_1row, device="npu", dtype=dtype)

    out_1row = F.layer_norm(x_1row, (3,))

    mean_val = np.mean(data_1row, axis=-1, keepdims=True)
    var_val = np.var(data_1row, axis=-1, keepdims=True)
    expected = ((data_1row - mean_val) / np.sqrt(var_val + 1e-5)).astype(np.float32)
    assert np.allclose(out_1row.to("cpu").numpy().astype(np.float32), expected, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_310b_layer_norm_multirow(dtype):
    """Test layer_norm with multi-row input (batch_size > 1)."""
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    x = torch.tensor(data, device="npu", dtype=dtype)

    out = F.layer_norm(x, (3,))

    # Compute expected manually
    mean_val = np.mean(data, axis=-1, keepdims=True)
    var_val = np.var(data, axis=-1, keepdims=True)
    expected = ((data - mean_val) / np.sqrt(var_val + 1e-5)).astype(np.float32)
    assert np.allclose(out.to("cpu").numpy().astype(np.float32), expected, atol=1e-2, rtol=1e-2)


def test_310b_layer_norm_does_not_break_bool_any_cast_path():

    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="npu", dtype=torch.float16)
    _ = F.layer_norm(x, (3,))

    mask = torch.tensor([True, False, True], dtype=torch.bool, device="npu")
    assert bool(torch.any(mask).to("cpu").item()) is True


def test_310b_mish():
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device="npu")
    out = F.mish(x)

    x_cpu = x.to("cpu")
    expected = F.mish(x_cpu)

    np.testing.assert_allclose(
        out.to("cpu").numpy(), expected.numpy(), rtol=1e-3, atol=1e-4
    )


def test_310b_batch_norm():
    from candle import nn

    # Input: (N=2, C=3, H=4, W=4)
    x = torch.randn(2, 3, 4, 4, device="npu")
    running_mean = torch.randn(3, device="npu")
    running_var = torch.rand(3, device="npu") + 0.1  # Ensure positive
    weight = torch.randn(3, device="npu")
    bias = torch.randn(3, device="npu")

    out = F.batch_norm(
        x, running_mean, running_var, weight, bias, training=False
    )

    # Compare with CPU
    x_cpu = x.to("cpu")
    running_mean_cpu = running_mean.to("cpu")
    running_var_cpu = running_var.to("cpu")
    weight_cpu = weight.to("cpu")
    bias_cpu = bias.to("cpu")

    expected = F.batch_norm(
        x_cpu, running_mean_cpu, running_var_cpu, weight_cpu, bias_cpu, training=False
    )

    np.testing.assert_allclose(
        out.to("cpu").numpy(), expected.numpy(), rtol=1e-3, atol=1e-4
    )


def test_310b_mish_module():
    from candle import nn

    layer = nn.Mish()
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device="npu")
    out = layer(x)

    x_cpu = x.to("cpu")
    expected = layer(x_cpu)

    np.testing.assert_allclose(
        out.to("cpu").numpy(), expected.numpy(), rtol=1e-3, atol=1e-4
    )


def test_310b_batch_norm_module():
    from candle import nn

    layer = nn.BatchNorm2d(3)
    layer = layer.to("npu")
    layer.eval()  # Use eval mode to use running stats

    x = torch.randn(2, 3, 4, 4, device="npu")
    out = layer(x)

    assert out.shape == x.shape
    assert out.device.type == "npu"

