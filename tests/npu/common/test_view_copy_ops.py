"""Tests for P0 view/copy/slice/reduce ops on NPU (910B common)."""
import numpy as np
import pytest
import candle as torch


pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="NPU not available"
)


# ---------------------------------------------------------------------------
# as_strided_
# ---------------------------------------------------------------------------

class TestAsStrided:
    def test_basic(self):
        x = torch.arange(12, dtype=torch.float32, device="npu")
        x.as_strided_((3, 4), (4, 1))
        assert x.shape == (3, 4)
        assert x.stride() == (4, 1)

    def test_with_offset(self):
        x = torch.arange(12, dtype=torch.float32, device="npu")
        x.as_strided_((2, 3), (3, 1), storage_offset=3)
        assert x.shape == (2, 3)
        # Verify via as_strided_copy which handles offset correctly
        y = x.as_strided_copy((2, 3), (3, 1))
        cpu = y.to("cpu").numpy()
        expected = np.arange(12, dtype=np.float32)[3:9].reshape(2, 3)
        np.testing.assert_allclose(cpu, expected)


# ---------------------------------------------------------------------------
# as_strided_copy
# ---------------------------------------------------------------------------

class TestAsStridedCopy:
    def test_basic(self):
        x = torch.arange(12, dtype=torch.float32, device="npu")
        y = x.as_strided_copy((3, 4), (4, 1))
        assert y.shape == (3, 4)
        assert y.is_contiguous()
        np.testing.assert_allclose(
            y.to("cpu").numpy(),
            np.arange(12, dtype=np.float32).reshape(3, 4),
        )

    def test_strided_view_becomes_contiguous(self):
        x = torch.arange(12, dtype=torch.float32, device="npu")
        # Overlapping strides (diagonal-like)
        y = x.as_strided_copy((3,), (4,), storage_offset=0)
        assert y.is_contiguous()
        np.testing.assert_allclose(y.to("cpu").numpy(), [0, 4, 8])


# ---------------------------------------------------------------------------
# as_strided_scatter
# ---------------------------------------------------------------------------

class TestAsStridedScatter:
    def test_basic(self):
        x = torch.zeros(12, dtype=torch.float32, device="npu")
        src = torch.ones(3, 4, dtype=torch.float32, device="npu")
        out = x.as_strided_scatter(src, (3, 4), (4, 1))
        cpu = out.to("cpu").numpy()
        np.testing.assert_allclose(cpu, np.ones(12, dtype=np.float32))
        # Original unchanged
        np.testing.assert_allclose(x.to("cpu").numpy(), np.zeros(12, dtype=np.float32))


# ---------------------------------------------------------------------------
# expand_copy
# ---------------------------------------------------------------------------

class TestExpandCopy:
    def test_basic(self):
        x = torch.tensor([[1.0], [2.0], [3.0]], device="npu")
        y = x.expand_copy(3, 4)
        assert y.shape == (3, 4)
        assert y.is_contiguous()
        expected = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]], dtype=np.float32)
        np.testing.assert_allclose(y.to("cpu").numpy(), expected)

    def test_after_nansum_still_copies_correctly(self):
        values = torch.tensor([1.0, float("nan"), 3.0, float("nan"), 5.0], device="npu")
        np.testing.assert_allclose(torch.nansum(values).to("cpu").numpy(), np.array(9.0, dtype=np.float32))

        x = torch.tensor([[1.0], [2.0], [3.0]], device="npu")
        y = x.expand_copy(3, 4)
        expected = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]], dtype=np.float32)
        np.testing.assert_allclose(y.to("cpu").numpy(), expected)


# ---------------------------------------------------------------------------
# slice / slice_copy / slice_scatter
# ---------------------------------------------------------------------------

class TestSlice:
    def test_slice_basic(self):
        x = torch.arange(10, dtype=torch.float32, device="npu")
        y = torch.slice(x, 0, 2, 7, 1)
        assert y.shape == (5,)
        assert y.stride() == (1,)
        # Verify values via slice_copy (view + to("cpu") has offset issues)
        yc = torch.slice_copy(x, 0, 2, 7, 1)
        np.testing.assert_allclose(yc.to("cpu").numpy(), np.arange(2, 7, dtype=np.float32))

    def test_slice_step(self):
        x = torch.arange(10, dtype=torch.float32, device="npu")
        y = torch.slice(x, 0, 0, 10, 2)
        assert y.shape == (5,)
        assert y.stride() == (2,)
        yc = torch.slice_copy(x, 0, 0, 10, 2)
        np.testing.assert_allclose(yc.to("cpu").numpy(), np.arange(0, 10, 2, dtype=np.float32))

    def test_slice_copy(self):
        x = torch.arange(10, dtype=torch.float32, device="npu")
        y = torch.slice_copy(x, 0, 2, 7, 1)
        assert y.is_contiguous()
        np.testing.assert_allclose(y.to("cpu").numpy(), np.arange(2, 7, dtype=np.float32))

    def test_slice_scatter(self):
        x = torch.zeros(10, dtype=torch.float32, device="npu")
        src = torch.ones(3, dtype=torch.float32, device="npu")
        out = torch.slice_scatter(x, src, 0, 2, 5, 1)
        cpu = out.to("cpu").numpy()
        expected = np.zeros(10, dtype=np.float32)
        expected[2:5] = 1.0
        np.testing.assert_allclose(cpu, expected)

    def test_slice_scatter_step(self):
        x = torch.zeros(10, dtype=torch.float32, device="npu")
        src = torch.ones(3, dtype=torch.float32, device="npu")
        out = torch.slice_scatter(x, src, 0, 0, 6, 2)
        cpu = out.to("cpu").numpy()
        expected = np.zeros(10, dtype=np.float32)
        expected[0:6:2] = 1.0
        np.testing.assert_allclose(cpu, expected)


# ---------------------------------------------------------------------------
# sum_to_size
# ---------------------------------------------------------------------------

class TestSumToSize:
    def test_identity(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="npu")
        y = torch.sum_to_size(x, (2, 2))
        np.testing.assert_allclose(y.to("cpu").numpy(), x.to("cpu").numpy())

    def test_reduce_dim(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="npu")
        y = torch.sum_to_size(x, (1, 2))
        np.testing.assert_allclose(y.to("cpu").numpy(), [[4.0, 6.0]])

    def test_reduce_all(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="npu")
        y = torch.sum_to_size(x, (1, 1))
        np.testing.assert_allclose(y.to("cpu").numpy(), [[10.0]])

    def test_reduce_leading_dims(self):
        x = torch.ones(2, 3, 4, dtype=torch.float32, device="npu")
        y = torch.sum_to_size(x, (3, 4))
        assert y.shape == (3, 4)
        np.testing.assert_allclose(y.to("cpu").numpy(), np.full((3, 4), 2.0))

    def test_int_accumulates_int64(self):
        x = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32, device="npu")
        y = torch.sum_to_size(x, (1, 2))
        assert y.dtype == torch.int64


# ---------------------------------------------------------------------------
# baddbmm tensor alpha/beta
# ---------------------------------------------------------------------------

class TestBaddbmmTensorAlphaBeta:
    def test_tensor_alpha(self):
        B, N, M, P = 2, 3, 4, 5
        self_t = torch.ones(B, N, P, dtype=torch.float32, device="npu")
        b1 = torch.ones(B, N, M, dtype=torch.float32, device="npu")
        b2 = torch.ones(B, M, P, dtype=torch.float32, device="npu")
        alpha = torch.tensor(2.0, device="npu")
        out = torch.baddbmm(self_t, b1, b2, beta=1.0, alpha=alpha)
        # 1.0 * ones + 2.0 * (ones @ ones) = 1 + 2*4 = 9
        np.testing.assert_allclose(out.to("cpu").numpy(), np.full((B, N, P), 9.0))

    def test_tensor_beta(self):
        B, N, M, P = 2, 3, 4, 5
        self_t = torch.ones(B, N, P, dtype=torch.float32, device="npu") * 3.0
        b1 = torch.ones(B, N, M, dtype=torch.float32, device="npu")
        b2 = torch.ones(B, M, P, dtype=torch.float32, device="npu")
        beta = torch.tensor(2.0, device="npu")
        out = torch.baddbmm(self_t, b1, b2, beta=beta, alpha=1.0)
        # 2.0 * 3 + 1.0 * 4 = 10
        np.testing.assert_allclose(out.to("cpu").numpy(), np.full((B, N, P), 10.0))


# ---------------------------------------------------------------------------
# repeat_interleave tensor repeats
# ---------------------------------------------------------------------------

class TestRepeatInterleaveTensorRepeats:
    def test_1d(self):
        x = torch.tensor([1.0, 2.0, 3.0], device="npu")
        repeats = torch.tensor([2, 0, 3], dtype=torch.int64, device="npu")
        out = torch.repeat_interleave(x, repeats, dim=0)
        expected = np.array([1.0, 1.0, 3.0, 3.0, 3.0])
        np.testing.assert_allclose(out.to("cpu").numpy(), expected)

    def test_2d(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device="npu")
        repeats = torch.tensor([1, 2, 1], dtype=torch.int64, device="npu")
        out = torch.repeat_interleave(x, repeats, dim=0)
        assert out.shape == (4, 2)
        expected = np.array([[1, 2], [3, 4], [3, 4], [5, 6]], dtype=np.float32)
        np.testing.assert_allclose(out.to("cpu").numpy(), expected)

    def test_no_dim_flattens(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="npu")
        repeats = torch.tensor([1, 2, 0, 3], dtype=torch.int64, device="npu")
        out = torch.repeat_interleave(x, repeats)
        expected = np.array([1.0, 2.0, 2.0, 4.0, 4.0, 4.0])
        np.testing.assert_allclose(out.to("cpu").numpy(), expected)
