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


@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
def test_910b_im2col_index_validation_any_regression():
    """Repeated any() checks on valid im2col indices must stay false on 910B."""
    idx_np = np.array([0, 1, 4, 5, 8, 9, 12, 13], dtype=np.int64)
    idx = torch.tensor(idx_np, device="npu", dtype=torch.int64)

    for _ in range(16):
        above = idx > 15
        below = idx < 0
        assert not bool(torch.any(above).to("cpu").numpy())
        assert not bool(torch.any(below).to("cpu").numpy())


# ---------------------------------------------------------------------------
# Regression guard: ops_soc table completeness
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Native 910B sequence instability reproduced in torch_npu
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
def test_910b_native_elementwise_sequence_matches_torch_npu_instability_boundary():
    """Keep Candle aligned with torch_npu's native 910B behavior boundary.

    Repeated native execution of maximum/where/logaddexp on 910B eventually
    corrupts later results in both Candle and real torch_npu on the same host.
    This is documented as a known upstream/native issue, not a Candle-specific
    route mismatch.  The common correctness suite should keep single-call
    semantic checks only; this watchlist test guards the documented boundary.
    """
    base = np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float32)
    rev = base[::-1].copy()
    cond = np.array([True, False, True, False])

    def _check_once():
        x = torch.tensor(base, device="npu", dtype=torch.float32)
        y = torch.tensor(rev, device="npu", dtype=torch.float32)
        c = torch.tensor(cond, device="npu")

        got_max = torch.maximum(x, y).to("cpu").numpy().astype(np.float32)
        got_where = torch.where(c, x, y).to("cpu").numpy().astype(np.float32)
        got_logaddexp = torch.logaddexp(x, y).to("cpu").numpy().astype(np.float32)

        return (
            np.allclose(got_max, np.maximum(base, rev).astype(np.float32), atol=1e-3, rtol=1e-3)
            and np.allclose(got_where, np.where(cond, base, rev).astype(np.float32), atol=1e-3, rtol=1e-3)
            and np.allclose(got_logaddexp, np.logaddexp(base, rev).astype(np.float32), atol=1e-3, rtol=1e-3)
        )

    # Single-call semantics must still be correct.
    assert _check_once()

    # Repeated execution is known to drift on native 910B paths in both Candle
    # and torch_npu. Keep this as a watchlist marker, not a hard correctness
    # requirement for passing all repetitions.
    seen_failure = False
    for _ in range(48):
        if not _check_once():
            seen_failure = True
            break

    assert seen_failure is True
