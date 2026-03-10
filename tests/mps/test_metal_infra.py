"""Tests for MPS Metal infrastructure: multi-dtype, strided access,
axis reduction, and comparison ops."""
import numpy as np
import pytest
import candle as torch


# ---------------------------------------------------------------------------
# Step 1: Multi-dtype GPU ops
# ---------------------------------------------------------------------------

class TestMultiDtypeOps:
    """Verify element-wise ops work on int32, int64, float16."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32, torch.int64])
    def test_add_dtype(self, dtype):
        if dtype in (torch.float32, torch.float16):
            a = torch.tensor([1.0, 2.0, 3.0], dtype=dtype, device="mps")
            b = torch.tensor([4.0, 5.0, 6.0], dtype=dtype, device="mps")
            expected = np.array([5.0, 7.0, 9.0])
        else:
            a = torch.tensor([1, 2, 3], dtype=dtype, device="mps")
            b = torch.tensor([4, 5, 6], dtype=dtype, device="mps")
            expected = np.array([5, 7, 9])
        result = torch.add(a, b)
        assert result.device.type == "mps"
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-3)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32, torch.int64])
    def test_sub_dtype(self, dtype):
        if dtype in (torch.float32, torch.float16):
            a = torch.tensor([5.0, 7.0, 9.0], dtype=dtype, device="mps")
            b = torch.tensor([4.0, 5.0, 6.0], dtype=dtype, device="mps")
            expected = np.array([1.0, 2.0, 3.0])
        else:
            a = torch.tensor([5, 7, 9], dtype=dtype, device="mps")
            b = torch.tensor([4, 5, 6], dtype=dtype, device="mps")
            expected = np.array([1, 2, 3])
        result = torch.sub(a, b)
        assert result.device.type == "mps"
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-3)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32, torch.int64])
    def test_mul_dtype(self, dtype):
        if dtype in (torch.float32, torch.float16):
            a = torch.tensor([1.0, 2.0, 3.0], dtype=dtype, device="mps")
            b = torch.tensor([4.0, 5.0, 6.0], dtype=dtype, device="mps")
            expected = np.array([4.0, 10.0, 18.0])
        else:
            a = torch.tensor([1, 2, 3], dtype=dtype, device="mps")
            b = torch.tensor([4, 5, 6], dtype=dtype, device="mps")
            expected = np.array([4, 10, 18])
        result = torch.mul(a, b)
        assert result.device.type == "mps"
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-3)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.int64])
    def test_neg_dtype(self, dtype):
        if dtype == torch.float32:
            a = torch.tensor([1.0, -2.0, 3.0], dtype=dtype, device="mps")
            expected = np.array([-1.0, 2.0, -3.0])
        else:
            a = torch.tensor([1, -2, 3], dtype=dtype, device="mps")
            expected = np.array([-1, 2, -3])
        result = torch.neg(a)
        assert result.device.type == "mps"
        np.testing.assert_allclose(result.cpu().numpy(), expected)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
    def test_abs_dtype(self, dtype):
        if dtype == torch.float32:
            a = torch.tensor([-1.0, 2.0, -3.0], dtype=dtype, device="mps")
            expected = np.array([1.0, 2.0, 3.0])
        else:
            a = torch.tensor([-1, 2, -3], dtype=dtype, device="mps")
            expected = np.array([1, 2, 3])
        result = torch.abs(a)
        assert result.device.type == "mps"
        np.testing.assert_allclose(result.cpu().numpy(), expected)

    def test_add_scalar_int32(self):
        a = torch.tensor([1, 2, 3], dtype=torch.int32, device="mps")
        result = a + 10
        expected = np.array([11, 12, 13])
        np.testing.assert_allclose(result.cpu().numpy(), expected)


# ---------------------------------------------------------------------------
# Step 2: Strided (non-contiguous) GPU ops
# ---------------------------------------------------------------------------

class TestStridedOps:
    """Verify GPU ops work on non-contiguous tensors (transpose, slice)."""

    def test_neg_transposed(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="mps")
        xt = x.T
        assert not xt.is_contiguous()
        result = torch.neg(xt)
        expected = -np.array([[1.0, 3.0], [2.0, 4.0]])
        np.testing.assert_allclose(result.cpu().numpy(), expected)

    def test_abs_transposed(self):
        x = torch.tensor([[-1.0, 2.0], [-3.0, 4.0]], device="mps")
        xt = x.T
        result = torch.abs(xt)
        expected = np.abs(np.array([[-1.0, -3.0], [2.0, 4.0]]))
        np.testing.assert_allclose(result.cpu().numpy(), expected)

    def test_add_transposed_tensors(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="mps")
        b = torch.tensor([[10.0, 20.0], [30.0, 40.0]], device="mps")
        result = torch.add(a.T, b.T)
        expected = np.array([[1.0, 3.0], [2.0, 4.0]]) + np.array([[10.0, 30.0], [20.0, 40.0]])
        np.testing.assert_allclose(result.cpu().numpy(), expected)

    def test_mul_transposed_scalar(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="mps")
        xt = x.T
        result = xt * 2.0
        expected = np.array([[2.0, 6.0], [4.0, 8.0]])
        np.testing.assert_allclose(result.cpu().numpy(), expected)

    def test_exp_transposed(self):
        x = torch.tensor([[0.0, 1.0], [2.0, 3.0]], device="mps")
        xt = x.T
        result = torch.exp(xt)
        expected = np.exp(np.array([[0.0, 2.0], [1.0, 3.0]]))
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-5)

    def test_sub_transposed(self):
        a = torch.tensor([[10.0, 20.0], [30.0, 40.0]], device="mps")
        b = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="mps")
        result = torch.sub(a.T, b.T)
        expected = np.array([[10.0, 30.0], [20.0, 40.0]]) - np.array([[1.0, 3.0], [2.0, 4.0]])
        np.testing.assert_allclose(result.cpu().numpy(), expected)


# ---------------------------------------------------------------------------
# Step 3: Axis reduction kernels
# ---------------------------------------------------------------------------

class TestAxisReduction:
    """Verify dim-reduce ops on GPU."""

    def test_sum_dim0(self):
        t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="mps")
        result = torch.sum(t, dim=0)
        np.testing.assert_allclose(result.cpu().numpy(), [5.0, 7.0, 9.0])

    def test_sum_dim1(self):
        t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="mps")
        result = torch.sum(t, dim=1)
        np.testing.assert_allclose(result.cpu().numpy(), [6.0, 15.0])

    def test_sum_dim_neg1(self):
        t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="mps")
        result = torch.sum(t, dim=-1)
        np.testing.assert_allclose(result.cpu().numpy(), [6.0, 15.0])

    def test_sum_keepdim(self):
        t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="mps")
        result = torch.sum(t, dim=0, keepdim=True)
        assert result.shape == (1, 3)
        np.testing.assert_allclose(result.cpu().numpy(), [[5.0, 7.0, 9.0]])

    def test_sum_multi_dim(self):
        t = torch.randn((2, 3, 4), device="mps")
        result = torch.sum(t, dim=[0, 2])
        expected = t.cpu().numpy().sum(axis=(0, 2))
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-5)

    def test_mean_dim0(self):
        t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="mps")
        result = torch.mean(t, dim=0)
        np.testing.assert_allclose(result.cpu().numpy(), [2.5, 3.5, 4.5])

    def test_mean_dim1(self):
        t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="mps")
        result = torch.mean(t, dim=1)
        np.testing.assert_allclose(result.cpu().numpy(), [2.0, 5.0])

    def test_argmax_dim(self):
        t = torch.tensor([[1.0, 3.0, 2.0], [6.0, 4.0, 5.0]], device="mps")
        result = torch.argmax(t, dim=1)
        np.testing.assert_array_equal(result.cpu().numpy(), [1, 0])

    def test_argmin_dim(self):
        t = torch.tensor([[1.0, 3.0, 2.0], [6.0, 4.0, 5.0]], device="mps")
        result = torch.argmin(t, dim=1)
        np.testing.assert_array_equal(result.cpu().numpy(), [0, 1])

    def test_amax_dim(self):
        t = torch.tensor([[1.0, 3.0, 2.0], [6.0, 4.0, 5.0]], device="mps")
        result = t.amax(dim=1)
        np.testing.assert_allclose(result.cpu().numpy(), [3.0, 6.0])

    def test_amin_dim(self):
        t = torch.tensor([[1.0, 3.0, 2.0], [6.0, 4.0, 5.0]], device="mps")
        result = t.amin(dim=1)
        np.testing.assert_allclose(result.cpu().numpy(), [1.0, 4.0])

    def test_sum_3d(self):
        t = torch.randn((2, 3, 4), device="mps")
        for dim in [0, 1, 2]:
            result = torch.sum(t, dim=dim)
            expected = t.cpu().numpy().sum(axis=dim)
            np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-5)

    def test_sum_full_tensor(self):
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="mps")
        result = torch.sum(t)
        np.testing.assert_allclose(result.cpu().numpy(), 10.0)

    def test_sum_keepdim_full(self):
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="mps")
        result = torch.sum(t, keepdim=True)
        assert result.shape == (1, 1)
        np.testing.assert_allclose(result.cpu().numpy(), [[10.0]])


# ---------------------------------------------------------------------------
# Step 4: Comparison ops with bool output
# ---------------------------------------------------------------------------

class TestComparisonOps:
    """Verify comparison ops produce bool output on GPU."""

    @pytest.mark.parametrize("op,np_op", [
        ("eq", np.equal), ("ne", np.not_equal),
        ("lt", np.less), ("le", np.less_equal),
        ("gt", np.greater), ("ge", np.greater_equal),
    ])
    def test_comparison_tensor(self, op, np_op):
        a = torch.tensor([1.0, 2.0, 3.0, 2.0], device="mps")
        b = torch.tensor([2.0, 2.0, 1.0, 3.0], device="mps")
        result = getattr(torch, op)(a, b)
        assert result.dtype.name == "bool"
        expected = np_op(
            np.array([1.0, 2.0, 3.0, 2.0]),
            np.array([2.0, 2.0, 1.0, 3.0]),
        )
        np.testing.assert_array_equal(result.cpu().numpy(), expected)

    @pytest.mark.parametrize("op,np_op", [
        ("eq", np.equal), ("ne", np.not_equal),
        ("lt", np.less), ("gt", np.greater),
    ])
    def test_comparison_scalar(self, op, np_op):
        a = torch.tensor([1.0, 2.0, 3.0], device="mps")
        result = getattr(torch, op)(a, 2.0)
        assert result.dtype.name == "bool"
        expected = np_op(np.array([1.0, 2.0, 3.0]), 2.0)
        np.testing.assert_array_equal(result.cpu().numpy(), expected)

    def test_gt_scalar_operator(self):
        a = torch.tensor([1.0, 2.0, 3.0], device="mps")
        result = a > 1.5
        assert result.dtype.name == "bool"
        np.testing.assert_array_equal(result.cpu().numpy(), [False, True, True])

    @pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
    def test_eq_multi_dtype(self, dtype):
        if dtype == torch.float32:
            a = torch.tensor([1.0, 2.0, 3.0], dtype=dtype, device="mps")
            b = torch.tensor([1.0, 0.0, 3.0], dtype=dtype, device="mps")
        else:
            a = torch.tensor([1, 2, 3], dtype=dtype, device="mps")
            b = torch.tensor([1, 0, 3], dtype=dtype, device="mps")
        result = torch.eq(a, b)
        assert result.dtype.name == "bool"
        np.testing.assert_array_equal(result.cpu().numpy(), [True, False, True])


# ---------------------------------------------------------------------------
# Step 5: Phase 1 training ops
# ---------------------------------------------------------------------------

class TestPhase1Ops:
    """Verify Phase 1 GPU kernels: leaky_relu, clamp, isinf/isnan/isfinite, softmax f16."""

    # --- leaky_relu ---

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_leaky_relu_positive(self, dtype):
        a = torch.tensor([1.0, 2.0, 3.0], dtype=dtype, device="mps")
        result = torch.nn.functional.leaky_relu(a, 0.01)
        np.testing.assert_allclose(result.cpu().numpy(), [1.0, 2.0, 3.0], rtol=1e-3)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_leaky_relu_negative(self, dtype):
        a = torch.tensor([-1.0, -2.0, -3.0], dtype=dtype, device="mps")
        result = torch.nn.functional.leaky_relu(a, 0.1)
        np.testing.assert_allclose(result.cpu().numpy(), [-0.1, -0.2, -0.3], rtol=1e-3)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_leaky_relu_default_slope(self, dtype):
        a = torch.tensor([1.0, -1.0, 0.0], dtype=dtype, device="mps")
        result = torch.nn.functional.leaky_relu(a)
        np.testing.assert_allclose(result.cpu().numpy(), [1.0, -0.01, 0.0], rtol=1e-3)

    def test_leaky_relu_strided(self):
        x = torch.tensor([[1.0, -2.0], [-3.0, 4.0]], device="mps")
        xt = x.T
        assert not xt.is_contiguous()
        result = torch.nn.functional.leaky_relu(xt, 0.1)
        expected = np.array([[1.0, -0.3], [-0.2, 4.0]])
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-3)

    # --- clamp ---

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_clamp_both_bounds(self, dtype):
        a = torch.tensor([-3.0, -1.0, 0.5, 2.0, 5.0], dtype=dtype, device="mps")
        result = torch.clamp(a, -1.0, 3.0)
        np.testing.assert_allclose(result.cpu().numpy(),
                                   [-1.0, -1.0, 0.5, 2.0, 3.0], rtol=1e-3)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_clamp_min_only(self, dtype):
        a = torch.tensor([-3.0, 0.0, 5.0], dtype=dtype, device="mps")
        result = torch.clamp(a, 0.0)  # min_val positional
        np.testing.assert_allclose(result.cpu().numpy(), [0.0, 0.0, 5.0], rtol=1e-3)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_clamp_max_only(self, dtype):
        a = torch.tensor([-3.0, 0.0, 5.0], dtype=dtype, device="mps")
        result = torch.clamp(a, None, 1.0)  # min_val=None, max_val=1.0
        np.testing.assert_allclose(result.cpu().numpy(), [-3.0, 0.0, 1.0], rtol=1e-3)

    def test_clamp_strided(self):
        x = torch.tensor([[-5.0, 3.0], [1.0, 8.0]], device="mps")
        xt = x.T
        result = torch.clamp(xt, 0.0, 5.0)
        expected = np.array([[0.0, 1.0], [3.0, 5.0]])
        np.testing.assert_allclose(result.cpu().numpy(), expected)

    def test_clamp_int32(self):
        a = torch.tensor([-5, 0, 3, 10], dtype=torch.int32, device="mps")
        result = torch.clamp(a, 0, 5)
        np.testing.assert_array_equal(result.cpu().numpy(), [0, 0, 3, 5])

    # --- clamp_min / clamp_max ---

    @pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
    def test_clamp_min_op(self, dtype):
        if dtype == torch.float32:
            a = torch.tensor([-3.0, 0.0, 5.0], dtype=dtype, device="mps")
            result = torch.clamp_min(a, 0.0)
            np.testing.assert_allclose(result.cpu().numpy(), [0.0, 0.0, 5.0])
        else:
            a = torch.tensor([-3, 0, 5], dtype=dtype, device="mps")
            result = torch.clamp_min(a, 0)
            np.testing.assert_array_equal(result.cpu().numpy(), [0, 0, 5])

    @pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
    def test_clamp_max_op(self, dtype):
        if dtype == torch.float32:
            a = torch.tensor([-3.0, 0.0, 5.0], dtype=dtype, device="mps")
            result = torch.clamp_max(a, 1.0)
            np.testing.assert_allclose(result.cpu().numpy(), [-3.0, 0.0, 1.0])
        else:
            a = torch.tensor([-3, 0, 5], dtype=dtype, device="mps")
            result = torch.clamp_max(a, 1)
            np.testing.assert_array_equal(result.cpu().numpy(), [-3, 0, 1])

    # --- isinf / isnan / isfinite ---

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_isinf(self, dtype):
        a = torch.tensor([1.0, float('inf'), float('-inf'), 0.0], dtype=dtype, device="mps")
        result = torch.isinf(a)
        assert result.dtype.name == "bool"
        np.testing.assert_array_equal(result.cpu().numpy(), [False, True, True, False])

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_isnan(self, dtype):
        a = torch.tensor([1.0, float('nan'), 0.0, float('nan')], dtype=dtype, device="mps")
        result = torch.isnan(a)
        assert result.dtype.name == "bool"
        np.testing.assert_array_equal(result.cpu().numpy(), [False, True, False, True])

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_isfinite(self, dtype):
        a = torch.tensor([1.0, float('inf'), float('nan'), 0.0], dtype=dtype, device="mps")
        result = torch.isfinite(a)
        assert result.dtype.name == "bool"
        np.testing.assert_array_equal(result.cpu().numpy(), [True, False, False, True])

    def test_isinf_strided(self):
        x = torch.tensor([[1.0, float('inf')], [float('-inf'), 0.0]], device="mps")
        xt = x.T
        assert not xt.is_contiguous()
        result = torch.isinf(xt)
        assert result.dtype.name == "bool"
        expected = np.array([[False, True], [True, False]])
        np.testing.assert_array_equal(result.cpu().numpy(), expected)

    # --- softmax float16 ---

    def test_softmax_f16(self):
        a = torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]],
                         dtype=torch.float16, device="mps")
        result = torch.nn.functional.softmax(a, dim=-1)
        assert result.dtype == torch.float16
        expected = np.array([[0.0900, 0.2447, 0.6652],
                             [0.3333, 0.3333, 0.3333]], dtype=np.float16)
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=5e-2)

    def test_softmax_f32_still_works(self):
        a = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32, device="mps")
        result = torch.nn.functional.softmax(a, dim=-1)
        assert result.dtype == torch.float32
        expected = np.array([[0.0900, 0.2447, 0.6652]], dtype=np.float32)
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-3)
