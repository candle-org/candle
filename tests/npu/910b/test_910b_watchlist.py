"""910B watchlist — exercises every op registered in ops_soc._FALLBACK_OPS["910b"].

Each test validates that the composite workaround produces correct results
on 910B hardware.  Tests are skipped when NPU is unavailable.
"""
import math

import numpy as np
import pytest

import candle as torch

NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()


# ---------------------------------------------------------------------------
# std (dim=None) — aclnnVar all-reduce 161002
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
def test_910b_std_scalar():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="npu", dtype=torch.float32)
    got = torch.std(x).to("cpu").numpy()
    expected = np.std([1, 2, 3, 4, 5], ddof=1, dtype=np.float32)
    np.testing.assert_allclose(got, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
def test_910b_std_dim():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="npu", dtype=torch.float32)
    got = torch.std(x, dim=1).to("cpu").numpy()
    expected = np.std([[1, 2], [3, 4]], axis=1, ddof=1, dtype=np.float32)
    np.testing.assert_allclose(got, expected, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# nansum — aclnnReduceNansum 161002
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
def test_910b_nansum():
    x_np = np.array([1.0, float("nan"), 3.0, float("nan"), 5.0], dtype=np.float32)
    x = torch.tensor(x_np, device="npu", dtype=torch.float32)
    got = float(torch.nansum(x).to("cpu").numpy())
    np.testing.assert_allclose(got, np.nansum(x_np), atol=1e-4)


@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
def test_910b_nansum_dim():
    x_np = np.array([[1.0, float("nan")], [float("nan"), 4.0]], dtype=np.float32)
    x = torch.tensor(x_np, device="npu", dtype=torch.float32)
    got = torch.nansum(x, dim=1).to("cpu").numpy()
    np.testing.assert_allclose(got, np.nansum(x_np, axis=1), atol=1e-4)


# ---------------------------------------------------------------------------
# instance_norm — aclnnInstanceNorm 161002
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
def test_910b_instance_norm():
    x_np = np.random.randn(2, 3, 4, 4).astype(np.float32)
    x = torch.tensor(x_np, device="npu", dtype=torch.float32)
    got = torch.nn.functional.instance_norm(x).to("cpu").numpy()
    # Manual reference: normalize per (N, C) instance over spatial dims
    for n in range(2):
        for c in range(3):
            patch = x_np[n, c]
            ref = (patch - patch.mean()) / (patch.std() + 1e-5)
            np.testing.assert_allclose(got[n, c], ref, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# avg_pool2d — aclnnAvgPool2d 161002
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
def test_910b_avg_pool2d():
    x_np = np.random.randn(1, 3, 4, 4).astype(np.float32)
    x = torch.tensor(x_np, device="npu", dtype=torch.float32)
    got = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2).to("cpu").numpy()
    # Reference: each 2x2 block averaged
    for c in range(3):
        for i in range(2):
            for j in range(2):
                expected = x_np[0, c, i*2:i*2+2, j*2:j*2+2].mean()
                np.testing.assert_allclose(got[0, c, i, j], expected, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# adaptive_avg_pool2d — cubeMathType contamination
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
def test_910b_adaptive_avg_pool2d():
    x_np = np.random.randn(1, 1, 4, 4).astype(np.float32)
    x = torch.tensor(x_np, device="npu", dtype=torch.float32)
    got = torch.nn.functional.adaptive_avg_pool2d(x, (2, 2)).to("cpu").numpy()
    # Reference: 4x4 → 2x2 means each 2x2 block is averaged
    expected = np.array([[
        [[x_np[0, 0, 0:2, 0:2].mean(), x_np[0, 0, 0:2, 2:4].mean()],
         [x_np[0, 0, 2:4, 0:2].mean(), x_np[0, 0, 2:4, 2:4].mean()]]
    ]], dtype=np.float32)
    np.testing.assert_allclose(got, expected, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# upsample_nearest1d — broken on 910B
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
def test_910b_upsample_nearest1d():
    x_np = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)
    x = torch.tensor(x_np, device="npu", dtype=torch.float32)
    got = torch.nn.functional.interpolate(x, size=6, mode="nearest").to("cpu").numpy()
    expected = np.array([[[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]]], dtype=np.float32)
    np.testing.assert_allclose(got, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# einsum — aclnnEinsum 161002
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
def test_910b_einsum_matmul():
    a_np = np.random.randn(2, 3).astype(np.float32)
    b_np = np.random.randn(3, 4).astype(np.float32)
    a = torch.tensor(a_np, device="npu", dtype=torch.float32)
    b = torch.tensor(b_np, device="npu", dtype=torch.float32)
    got = torch.einsum("ij,jk->ik", a, b).to("cpu").numpy()
    expected = a_np @ b_np
    np.testing.assert_allclose(got, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
def test_910b_einsum_trace():
    a_np = np.random.randn(3, 3).astype(np.float32)
    a = torch.tensor(a_np, device="npu", dtype=torch.float32)
    got = float(torch.einsum("ii->", a).to("cpu").numpy())
    expected = float(np.trace(a_np))
    np.testing.assert_allclose(got, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# isinf — aclnnIsInf 161001
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
def test_910b_isinf():
    x_np = np.array([1.0, float("inf"), -float("inf"), 0.0, float("nan")], dtype=np.float32)
    x = torch.tensor(x_np, device="npu", dtype=torch.float32)
    got = torch.isinf(x).to("cpu").numpy()
    expected = np.isinf(x_np)
    np.testing.assert_array_equal(got, expected)


# ---------------------------------------------------------------------------
# im2col — aclnnIm2col 561103
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
def test_910b_im2col():
    # Simple 1x1x4x4 input, 2x2 kernel, stride 1, no padding
    x_np = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    x = torch.tensor(x_np, device="npu", dtype=torch.float32)
    got = torch.nn.functional.unfold(x, kernel_size=2).to("cpu").numpy()
    # Output shape: (1, C*kH*kW, L) = (1, 4, 9)
    assert got.shape == (1, 4, 9)
    # First column = top-left 2x2 patch: [0, 1, 4, 5]
    np.testing.assert_allclose(got[0, :, 0], [0, 1, 4, 5], atol=1e-6)


# ---------------------------------------------------------------------------
# Regression guard: ops_soc table completeness
# ---------------------------------------------------------------------------

def test_910b_fallback_ops_registered():
    """Ensure all 8 known 910B fallback ops are in the capability table."""
    from candle._backends.npu import ops_soc
    expected = {"std", "nansum", "instance_norm", "avg_pool2d",
                "adaptive_avg_pool2d",
                "upsample_nearest1d", "einsum", "isinf", "im2col"}
    got = set(ops_soc.fallback_ops("910b"))
    assert got == expected
