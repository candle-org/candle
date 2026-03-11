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


# ---------------------------------------------------------------------------
# Step 6: Phase 2 reduction ops
# ---------------------------------------------------------------------------

class TestPhase2Reductions:
    """Verify Phase 2 GPU reduction kernels: prod, any, all, var, std, logsumexp."""

    # --- prod ---

    def test_prod_dim(self):
        t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="mps")
        r0 = torch.prod(t, dim=0)
        np.testing.assert_allclose(r0.cpu().numpy(), [4.0, 10.0, 18.0])
        r1 = torch.prod(t, dim=1)
        np.testing.assert_allclose(r1.cpu().numpy(), [6.0, 120.0])

    def test_prod_full_tensor(self):
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="mps")
        result = torch.prod(t)
        np.testing.assert_allclose(result.cpu().numpy(), 24.0)

    def test_prod_keepdim(self):
        t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="mps")
        result = torch.prod(t, dim=1, keepdim=True)
        assert result.shape == (2, 1)
        np.testing.assert_allclose(result.cpu().numpy(), [[6.0], [120.0]])

    def test_prod_int32(self):
        t = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32, device="mps")
        result = torch.prod(t, dim=1)
        np.testing.assert_array_equal(result.cpu().numpy(), [6, 120])

    # --- any ---

    def test_any_dim(self):
        t = torch.tensor([[True, False, False], [False, False, False]], device="mps")
        r0 = torch.any(t, dim=0)
        assert r0.dtype.name == "bool"
        np.testing.assert_array_equal(r0.cpu().numpy(), [True, False, False])
        r1 = torch.any(t, dim=1)
        np.testing.assert_array_equal(r1.cpu().numpy(), [True, False])

    def test_any_full_tensor(self):
        t_true = torch.tensor([[False, True], [False, False]], device="mps")
        assert torch.any(t_true).cpu().numpy() == True
        t_false = torch.tensor([[False, False], [False, False]], device="mps")
        assert torch.any(t_false).cpu().numpy() == False

    def test_any_numeric(self):
        t = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], device="mps")
        r = torch.any(t, dim=1)
        assert r.dtype.name == "bool"
        np.testing.assert_array_equal(r.cpu().numpy(), [True, False])

    # --- all ---

    def test_all_dim(self):
        t = torch.tensor([[True, True, False], [True, True, True]], device="mps")
        r0 = torch.all(t, dim=0)
        assert r0.dtype.name == "bool"
        np.testing.assert_array_equal(r0.cpu().numpy(), [True, True, False])
        r1 = torch.all(t, dim=1)
        np.testing.assert_array_equal(r1.cpu().numpy(), [False, True])

    def test_all_full_tensor(self):
        t_true = torch.tensor([[True, True], [True, True]], device="mps")
        assert torch.all(t_true).cpu().numpy() == True
        t_false = torch.tensor([[True, True], [True, False]], device="mps")
        assert torch.all(t_false).cpu().numpy() == False

    def test_all_numeric(self):
        t = torch.tensor([[1.0, 2.0, 3.0], [1.0, 0.0, 3.0]], device="mps")
        r = torch.all(t, dim=1)
        assert r.dtype.name == "bool"
        np.testing.assert_array_equal(r.cpu().numpy(), [True, False])

    # --- var ---

    def test_var_dim(self):
        t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="mps")
        # unbiased (default)
        r = torch.var(t, dim=1)
        expected = np.var(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
                         axis=1, ddof=1)
        np.testing.assert_allclose(r.cpu().numpy(), expected, rtol=1e-5)
        # biased
        r_biased = torch.var(t, dim=1, unbiased=False)
        expected_biased = np.var(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
                                axis=1, ddof=0)
        np.testing.assert_allclose(r_biased.cpu().numpy(), expected_biased, rtol=1e-5)

    def test_var_full_tensor(self):
        t = torch.tensor([1.0, 2.0, 3.0, 4.0], device="mps")
        r = torch.var(t)
        expected = np.var(np.array([1, 2, 3, 4], dtype=np.float32), ddof=1)
        np.testing.assert_allclose(r.cpu().numpy(), expected, rtol=1e-5)

    def test_var_keepdim(self):
        t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="mps")
        r = torch.var(t, dim=1, keepdim=True)
        assert r.shape == (2, 1)

    # --- std ---

    def test_std_dim(self):
        t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="mps")
        r = torch.std(t, dim=1)
        expected = np.std(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
                         axis=1, ddof=1)
        np.testing.assert_allclose(r.cpu().numpy(), expected, rtol=1e-5)

    def test_std_full_tensor(self):
        t = torch.tensor([1.0, 2.0, 3.0, 4.0], device="mps")
        r = torch.std(t)
        expected = np.std(np.array([1, 2, 3, 4], dtype=np.float32), ddof=1)
        np.testing.assert_allclose(r.cpu().numpy(), expected, rtol=1e-5)

    # --- logsumexp ---

    def test_logsumexp_dim(self):
        t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="mps")
        r = torch.logsumexp(t, dim=1)
        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        expected = np.log(np.sum(np.exp(arr), axis=1))
        np.testing.assert_allclose(r.cpu().numpy(), expected, rtol=1e-5)

    def test_logsumexp_keepdim(self):
        t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="mps")
        r = torch.logsumexp(t, dim=1, keepdim=True)
        assert r.shape == (2, 1)

    def test_logsumexp_large_values(self):
        """Numerical stability: large inputs should not overflow."""
        t = torch.tensor([1000.0, 1001.0, 1002.0], device="mps")
        r = torch.logsumexp(t, dim=0)
        arr = np.array([1000, 1001, 1002], dtype=np.float32)
        expected = np.log(np.sum(np.exp(arr - arr.max()))) + arr.max()
        np.testing.assert_allclose(r.cpu().numpy(), expected, rtol=1e-5)

    def test_logsumexp_f16(self):
        t = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float16, device="mps")
        r = torch.logsumexp(t, dim=1)
        assert r.dtype == torch.float16
        arr = np.array([[1, 2, 3]], dtype=np.float16)
        expected = np.log(np.sum(np.exp(arr - arr.max(axis=1, keepdims=True)),
                                 axis=1)) + arr.max(axis=1)
        np.testing.assert_allclose(r.cpu().numpy(), expected.astype(np.float16),
                                   rtol=5e-2)


# ---------------------------------------------------------------------------
# Phase 3: Shape & Index GPU kernels
# ---------------------------------------------------------------------------

class TestPhase3ShapeIndex:
    """Verify GPU kernels for where, masked_fill, tril, triu,
    index_select, gather, cat, stack."""

    # --- where ---

    def test_where_f32(self):
        cond = torch.tensor([True, False, True, False], device="mps")
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], device="mps")
        y = torch.tensor([10.0, 20.0, 30.0, 40.0], device="mps")
        result = torch.where(cond, x, y)
        expected = np.array([1.0, 20.0, 3.0, 40.0])
        np.testing.assert_allclose(result.cpu().numpy(), expected)

    def test_where_f16(self):
        cond = torch.tensor([True, False, True], device="mps")
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16, device="mps")
        y = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float16, device="mps")
        result = torch.where(cond, x, y)
        expected = np.array([1.0, 20.0, 3.0])
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-3)

    def test_where_i32(self):
        cond = torch.tensor([True, False, True], device="mps")
        x = torch.tensor([1, 2, 3], dtype=torch.int32, device="mps")
        y = torch.tensor([10, 20, 30], dtype=torch.int32, device="mps")
        result = torch.where(cond, x, y)
        expected = np.array([1, 20, 3])
        np.testing.assert_allclose(result.cpu().numpy(), expected)

    def test_where_scalar_y(self):
        cond = torch.tensor([True, False, True, False], device="mps")
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], device="mps")
        result = torch.where(cond, x, 0.0)
        expected = np.array([1.0, 0.0, 3.0, 0.0])
        np.testing.assert_allclose(result.cpu().numpy(), expected)

    def test_where_2d(self):
        cond = torch.tensor([[True, False], [False, True]], device="mps")
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="mps")
        y = torch.tensor([[10.0, 20.0], [30.0, 40.0]], device="mps")
        result = torch.where(cond, x, y)
        expected = np.array([[1.0, 20.0], [30.0, 4.0]])
        np.testing.assert_allclose(result.cpu().numpy(), expected)

    # --- masked_fill ---

    def test_masked_fill_f32(self):
        a = torch.tensor([1.0, 2.0, 3.0, 4.0], device="mps")
        mask = torch.tensor([True, False, True, False], device="mps")
        result = a.masked_fill(mask, -1.0)
        expected = np.array([-1.0, 2.0, -1.0, 4.0])
        np.testing.assert_allclose(result.cpu().numpy(), expected)

    def test_masked_fill_f16(self):
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16, device="mps")
        mask = torch.tensor([False, True, False], device="mps")
        result = a.masked_fill(mask, 0.0)
        expected = np.array([1.0, 0.0, 3.0])
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-3)

    def test_masked_fill_i32(self):
        a = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="mps")
        mask = torch.tensor([True, True, False, False], device="mps")
        result = a.masked_fill(mask, -99)
        expected = np.array([-99, -99, 3, 4])
        np.testing.assert_allclose(result.cpu().numpy(), expected)

    def test_masked_fill_2d(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="mps")
        mask = torch.tensor([[True, False], [False, True]], device="mps")
        result = a.masked_fill(mask, 0.0)
        expected = np.array([[0.0, 2.0], [3.0, 0.0]])
        np.testing.assert_allclose(result.cpu().numpy(), expected)

    # --- tril ---

    def test_tril_f32(self):
        a = torch.ones(3, 3, device="mps")
        result = torch.tril(a)
        expected = np.tril(np.ones((3, 3)))
        np.testing.assert_allclose(result.cpu().numpy(), expected)

    def test_tril_i32(self):
        a = torch.ones(3, 3, dtype=torch.int32, device="mps")
        result = torch.tril(a)
        expected = np.tril(np.ones((3, 3)))
        np.testing.assert_allclose(result.cpu().numpy(), expected)

    def test_tril_diagonal(self):
        a = torch.ones(4, 4, device="mps")
        result = torch.tril(a, diagonal=1)
        expected = np.tril(np.ones((4, 4)), k=1)
        np.testing.assert_allclose(result.cpu().numpy(), expected)

    def test_tril_batched(self):
        a = torch.ones(2, 3, 3, device="mps")
        result = torch.tril(a)
        expected = np.tril(np.ones((3, 3)))
        for b in range(2):
            np.testing.assert_allclose(result[b].cpu().numpy(), expected)

    # --- triu ---

    def test_triu_f32(self):
        a = torch.ones(3, 3, device="mps")
        result = torch.triu(a)
        expected = np.triu(np.ones((3, 3)))
        np.testing.assert_allclose(result.cpu().numpy(), expected)

    def test_triu_i32(self):
        a = torch.ones(3, 3, dtype=torch.int32, device="mps")
        result = torch.triu(a)
        expected = np.triu(np.ones((3, 3)))
        np.testing.assert_allclose(result.cpu().numpy(), expected)

    def test_triu_diagonal(self):
        a = torch.ones(4, 4, device="mps")
        result = torch.triu(a, diagonal=-1)
        expected = np.triu(np.ones((4, 4)), k=-1)
        np.testing.assert_allclose(result.cpu().numpy(), expected)

    # --- index_select ---

    def test_index_select_dim0_f32(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device="mps")
        idx = torch.tensor([0, 2], dtype=torch.int64, device="mps")
        result = torch.index_select(a, 0, idx)
        expected = np.array([[1.0, 2.0], [5.0, 6.0]])
        np.testing.assert_allclose(result.cpu().numpy(), expected)

    def test_index_select_dim0_i32(self):
        a = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32, device="mps")
        idx = torch.tensor([1, 2], dtype=torch.int64, device="mps")
        result = torch.index_select(a, 0, idx)
        expected = np.array([[3, 4], [5, 6]])
        np.testing.assert_allclose(result.cpu().numpy(), expected)

    def test_index_select_dim1(self):
        a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="mps")
        idx = torch.tensor([0, 2], dtype=torch.int64, device="mps")
        result = torch.index_select(a, 1, idx)
        expected = np.array([[1.0, 3.0], [4.0, 6.0]])
        np.testing.assert_allclose(result.cpu().numpy(), expected)

    def test_index_select_3d(self):
        a = torch.arange(24, dtype=torch.float32, device="mps").reshape(2, 3, 4)
        idx = torch.tensor([1, 2], dtype=torch.int64, device="mps")
        result = torch.index_select(a, 1, idx)
        expected = np.arange(24).reshape(2, 3, 4)[:, [1, 2], :]
        np.testing.assert_allclose(result.cpu().numpy(), expected.astype(np.float32))

    # --- gather ---

    def test_gather_dim0(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="mps")
        idx = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64, device="mps")
        result = torch.gather(a, 0, idx)
        expected = np.array([[1.0, 4.0], [3.0, 2.0]])
        np.testing.assert_allclose(result.cpu().numpy(), expected)

    def test_gather_dim1(self):
        a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="mps")
        idx = torch.tensor([[2, 0], [1, 2]], dtype=torch.int64, device="mps")
        result = torch.gather(a, 1, idx)
        expected = np.array([[3.0, 1.0], [5.0, 6.0]])
        np.testing.assert_allclose(result.cpu().numpy(), expected)

    def test_gather_i32(self):
        a = torch.tensor([[10, 20, 30], [40, 50, 60]], dtype=torch.int32, device="mps")
        idx = torch.tensor([[1, 0], [2, 1]], dtype=torch.int64, device="mps")
        result = torch.gather(a, 1, idx)
        expected = np.array([[20, 10], [60, 50]])
        np.testing.assert_allclose(result.cpu().numpy(), expected)

    # --- cat ---

    def test_cat_dim0_f32(self):
        a = torch.tensor([1.0, 2.0, 3.0], device="mps")
        b = torch.tensor([4.0, 5.0, 6.0], device="mps")
        result = torch.cat([a, b], dim=0)
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        np.testing.assert_allclose(result.cpu().numpy(), expected)

    def test_cat_dim0_f16(self):
        a = torch.tensor([1.0, 2.0], dtype=torch.float16, device="mps")
        b = torch.tensor([3.0, 4.0], dtype=torch.float16, device="mps")
        result = torch.cat([a, b], dim=0)
        expected = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-3)

    def test_cat_dim0_i32(self):
        a = torch.tensor([1, 2, 3], dtype=torch.int32, device="mps")
        b = torch.tensor([4, 5, 6], dtype=torch.int32, device="mps")
        result = torch.cat([a, b], dim=0)
        expected = np.array([1, 2, 3, 4, 5, 6])
        np.testing.assert_allclose(result.cpu().numpy(), expected)

    def test_cat_dim1(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="mps")
        b = torch.tensor([[5.0], [6.0]], device="mps")
        result = torch.cat([a, b], dim=1)
        expected = np.array([[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]])
        np.testing.assert_allclose(result.cpu().numpy(), expected)

    def test_cat_multiple(self):
        a = torch.tensor([1.0, 2.0], device="mps")
        b = torch.tensor([3.0], device="mps")
        c = torch.tensor([4.0, 5.0, 6.0], device="mps")
        result = torch.cat([a, b, c], dim=0)
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        np.testing.assert_allclose(result.cpu().numpy(), expected)

    def test_cat_3d(self):
        a = torch.arange(12, dtype=torch.float32, device="mps").reshape(2, 2, 3)
        b = torch.arange(12, 24, dtype=torch.float32, device="mps").reshape(2, 2, 3)
        result = torch.cat([a, b], dim=0)
        assert result.shape == (4, 2, 3)
        expected = np.arange(24, dtype=np.float32).reshape(4, 2, 3)
        np.testing.assert_allclose(result.cpu().numpy(), expected)

    # --- stack ---

    def test_stack_dim0_f32(self):
        a = torch.tensor([1.0, 2.0, 3.0], device="mps")
        b = torch.tensor([4.0, 5.0, 6.0], device="mps")
        result = torch.stack([a, b], dim=0)
        expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        np.testing.assert_allclose(result.cpu().numpy(), expected)

    def test_stack_dim0_i32(self):
        a = torch.tensor([1, 2, 3], dtype=torch.int32, device="mps")
        b = torch.tensor([4, 5, 6], dtype=torch.int32, device="mps")
        result = torch.stack([a, b], dim=0)
        expected = np.array([[1, 2, 3], [4, 5, 6]])
        np.testing.assert_allclose(result.cpu().numpy(), expected)

    def test_stack_dim1(self):
        a = torch.tensor([1.0, 2.0, 3.0], device="mps")
        b = torch.tensor([4.0, 5.0, 6.0], device="mps")
        result = torch.stack([a, b], dim=1)
        expected = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        np.testing.assert_allclose(result.cpu().numpy(), expected)

    def test_stack_2d(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="mps")
        b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device="mps")
        result = torch.stack([a, b], dim=0)
        expected = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
        np.testing.assert_allclose(result.cpu().numpy(), expected)


# ---------------------------------------------------------------------------
# Conv2d GPU kernel tests
# ---------------------------------------------------------------------------

class TestConv2dGPU:
    """Verify conv2d Metal compute shader correctness."""

    def _conv2d_ref(self, x_np, w_np, b_np=None, stride=1, padding=0, dilation=1):
        """CPU reference conv2d via candle's numpy path."""
        x = torch.tensor(x_np)
        w = torch.tensor(w_np)
        b = torch.tensor(b_np) if b_np is not None else None
        return torch.nn.functional.conv2d(
            x, w, bias=b, stride=stride, padding=padding, dilation=dilation
        ).numpy()

    def test_basic_f32(self):
        np.random.seed(0)
        x = torch.tensor(np.random.randn(1, 3, 8, 8).astype(np.float32), device="mps")
        w = torch.tensor(np.random.randn(4, 3, 3, 3).astype(np.float32), device="mps")
        out = torch.nn.functional.conv2d(x, w)
        ref = self._conv2d_ref(x.cpu().numpy(), w.cpu().numpy())
        assert out.shape == (1, 4, 6, 6)
        assert out.device.type == "mps"
        np.testing.assert_allclose(out.cpu().numpy(), ref, atol=1e-5)

    def test_basic_f16(self):
        np.random.seed(1)
        x = torch.tensor(np.random.randn(1, 3, 8, 8).astype(np.float32), device="mps")
        w = torch.tensor(np.random.randn(4, 3, 3, 3).astype(np.float32), device="mps")
        # f32 reference
        ref_f32 = torch.nn.functional.conv2d(x, w)
        # f16 GPU
        out = torch.nn.functional.conv2d(x.to(torch.float16), w.to(torch.float16))
        assert out.dtype == torch.float16
        np.testing.assert_allclose(
            out.to(torch.float32).cpu().numpy(), ref_f32.cpu().numpy(), atol=0.05)

    def test_with_bias(self):
        np.random.seed(2)
        x_np = np.random.randn(2, 3, 6, 6).astype(np.float32)
        w_np = np.random.randn(8, 3, 3, 3).astype(np.float32)
        b_np = np.random.randn(8).astype(np.float32)
        x = torch.tensor(x_np, device="mps")
        w = torch.tensor(w_np, device="mps")
        b = torch.tensor(b_np, device="mps")
        out = torch.nn.functional.conv2d(x, w, bias=b)
        ref = self._conv2d_ref(x_np, w_np, b_np)
        np.testing.assert_allclose(out.cpu().numpy(), ref, atol=1e-5)

    def test_padding(self):
        np.random.seed(3)
        x_np = np.random.randn(1, 1, 5, 5).astype(np.float32)
        w_np = np.random.randn(1, 1, 3, 3).astype(np.float32)
        x = torch.tensor(x_np, device="mps")
        w = torch.tensor(w_np, device="mps")
        out = torch.nn.functional.conv2d(x, w, padding=1)
        ref = self._conv2d_ref(x_np, w_np, padding=1)
        assert out.shape == (1, 1, 5, 5)
        np.testing.assert_allclose(out.cpu().numpy(), ref, atol=1e-5)

    def test_stride(self):
        np.random.seed(4)
        x_np = np.random.randn(1, 2, 8, 8).astype(np.float32)
        w_np = np.random.randn(4, 2, 3, 3).astype(np.float32)
        x = torch.tensor(x_np, device="mps")
        w = torch.tensor(w_np, device="mps")
        out = torch.nn.functional.conv2d(x, w, stride=2)
        ref = self._conv2d_ref(x_np, w_np, stride=2)
        assert out.shape == (1, 4, 3, 3)
        np.testing.assert_allclose(out.cpu().numpy(), ref, atol=1e-5)

    def test_dilation(self):
        np.random.seed(5)
        x_np = np.random.randn(1, 1, 8, 8).astype(np.float32)
        w_np = np.random.randn(2, 1, 3, 3).astype(np.float32)
        x = torch.tensor(x_np, device="mps")
        w = torch.tensor(w_np, device="mps")
        out = torch.nn.functional.conv2d(x, w, dilation=2)
        ref = self._conv2d_ref(x_np, w_np, dilation=2)
        assert out.shape == (1, 2, 4, 4)
        np.testing.assert_allclose(out.cpu().numpy(), ref, atol=1e-5)

    def test_stride_padding_combined(self):
        np.random.seed(6)
        x_np = np.random.randn(2, 4, 16, 16).astype(np.float32)
        w_np = np.random.randn(8, 4, 3, 3).astype(np.float32)
        b_np = np.random.randn(8).astype(np.float32)
        x = torch.tensor(x_np, device="mps")
        w = torch.tensor(w_np, device="mps")
        b = torch.tensor(b_np, device="mps")
        out = torch.nn.functional.conv2d(x, w, bias=b, stride=2, padding=1)
        ref = self._conv2d_ref(x_np, w_np, b_np, stride=2, padding=1)
        assert out.shape == (2, 8, 8, 8)
        np.testing.assert_allclose(out.cpu().numpy(), ref, atol=1e-5)

    def test_1x1_conv(self):
        """1x1 convolution (pointwise)."""
        np.random.seed(7)
        x_np = np.random.randn(1, 16, 4, 4).astype(np.float32)
        w_np = np.random.randn(32, 16, 1, 1).astype(np.float32)
        x = torch.tensor(x_np, device="mps")
        w = torch.tensor(w_np, device="mps")
        out = torch.nn.functional.conv2d(x, w)
        ref = self._conv2d_ref(x_np, w_np)
        assert out.shape == (1, 32, 4, 4)
        np.testing.assert_allclose(out.cpu().numpy(), ref, atol=1e-5)

    def test_large_kernel(self):
        """Conv with 5x5 kernel."""
        np.random.seed(8)
        x_np = np.random.randn(1, 3, 16, 16).astype(np.float32)
        w_np = np.random.randn(8, 3, 5, 5).astype(np.float32)
        x = torch.tensor(x_np, device="mps")
        w = torch.tensor(w_np, device="mps")
        out = torch.nn.functional.conv2d(x, w, padding=2)
        ref = self._conv2d_ref(x_np, w_np, padding=2)
        assert out.shape == (1, 8, 16, 16)
        np.testing.assert_allclose(out.cpu().numpy(), ref, atol=1e-5)

    def test_batched(self):
        """Batched conv2d with N > 1."""
        np.random.seed(9)
        x_np = np.random.randn(4, 3, 8, 8).astype(np.float32)
        w_np = np.random.randn(16, 3, 3, 3).astype(np.float32)
        x = torch.tensor(x_np, device="mps")
        w = torch.tensor(w_np, device="mps")
        out = torch.nn.functional.conv2d(x, w)
        ref = self._conv2d_ref(x_np, w_np)
        assert out.shape == (4, 16, 6, 6)
        np.testing.assert_allclose(out.cpu().numpy(), ref, atol=1e-5)

    def test_nn_conv2d_module(self):
        """Test nn.Conv2d module on MPS."""
        layer = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1).to("mps")
        x = torch.randn(2, 3, 8, 8, device="mps")
        out = layer(x)
        assert out.shape == (2, 16, 8, 8)
        assert out.device.type == "mps"


# ---------------------------------------------------------------------------
# Normalization GPU kernels
# ---------------------------------------------------------------------------

class TestNormGPU:
    """Verify layer_norm, rms_norm, batch_norm Metal GPU kernels."""

    # --- layer_norm ---

    def test_layer_norm_basic_f32(self):
        np.random.seed(0)
        x_np = np.random.randn(4, 8).astype(np.float32)
        w_np = np.random.randn(8).astype(np.float32)
        b_np = np.random.randn(8).astype(np.float32)
        x = torch.tensor(x_np, device="mps")
        w = torch.tensor(w_np, device="mps")
        b = torch.tensor(b_np, device="mps")
        out = torch.nn.functional.layer_norm(x, [8], weight=w, bias=b)
        assert out.device.type == "mps"
        # CPU reference
        mean = x_np.mean(axis=-1, keepdims=True)
        var = x_np.var(axis=-1, keepdims=True)
        ref = (x_np - mean) / np.sqrt(var + 1e-5) * w_np + b_np
        np.testing.assert_allclose(out.cpu().numpy(), ref, atol=1e-5)

    def test_layer_norm_no_affine(self):
        np.random.seed(1)
        x_np = np.random.randn(4, 16).astype(np.float32)
        x = torch.tensor(x_np, device="mps")
        out = torch.nn.functional.layer_norm(x, [16])
        assert out.device.type == "mps"
        mean = x_np.mean(axis=-1, keepdims=True)
        var = x_np.var(axis=-1, keepdims=True)
        ref = (x_np - mean) / np.sqrt(var + 1e-5)
        np.testing.assert_allclose(out.cpu().numpy(), ref, atol=1e-5)

    def test_layer_norm_2d_normalized_shape(self):
        np.random.seed(2)
        x_np = np.random.randn(2, 4, 8).astype(np.float32)
        w_np = np.ones((4, 8), dtype=np.float32)
        b_np = np.zeros((4, 8), dtype=np.float32)
        x = torch.tensor(x_np, device="mps")
        w = torch.tensor(w_np, device="mps")
        b = torch.tensor(b_np, device="mps")
        out = torch.nn.functional.layer_norm(x, [4, 8], weight=w, bias=b)
        assert out.device.type == "mps"
        mean = x_np.reshape(2, -1).mean(axis=-1, keepdims=True).reshape(2, 1, 1)
        var = x_np.reshape(2, -1).var(axis=-1, keepdims=True).reshape(2, 1, 1)
        ref = (x_np - mean) / np.sqrt(var + 1e-5)
        np.testing.assert_allclose(out.cpu().numpy(), ref, atol=1e-5)

    def test_layer_norm_f16(self):
        np.random.seed(3)
        x_np = np.random.randn(4, 8).astype(np.float32)
        x = torch.tensor(x_np, device="mps").to(torch.float16)
        out = torch.nn.functional.layer_norm(x, [8])
        assert out.device.type == "mps"
        assert out.dtype == torch.float16
        # f32 reference
        mean = x_np.mean(axis=-1, keepdims=True)
        var = x_np.var(axis=-1, keepdims=True)
        ref = (x_np - mean) / np.sqrt(var + 1e-5)
        np.testing.assert_allclose(out.cpu().numpy(), ref.astype(np.float16), atol=0.05)

    def test_layer_norm_large(self):
        np.random.seed(4)
        x_np = np.random.randn(4, 128, 768).astype(np.float32)
        x = torch.tensor(x_np, device="mps")
        out = torch.nn.functional.layer_norm(x, [768])
        assert out.device.type == "mps"
        assert out.shape == (4, 128, 768)
        mean = x_np.mean(axis=-1, keepdims=True)
        var = x_np.var(axis=-1, keepdims=True)
        ref = (x_np - mean) / np.sqrt(var + 1e-5)
        np.testing.assert_allclose(out.cpu().numpy(), ref, atol=1e-4)

    # --- rms_norm ---

    def test_rms_norm_with_weight(self):
        np.random.seed(10)
        x_np = np.random.randn(4, 8).astype(np.float32)
        w_np = np.random.randn(8).astype(np.float32)
        x = torch.tensor(x_np, device="mps")
        w = torch.tensor(w_np, device="mps")
        out = torch.nn.functional.rms_norm(x, [8], weight=w)
        assert out.device.type == "mps"
        rms = np.sqrt(np.mean(x_np ** 2, axis=-1, keepdims=True) + 1e-6)
        ref = x_np / rms * w_np
        np.testing.assert_allclose(out.cpu().numpy(), ref, atol=1e-5)

    def test_rms_norm_no_weight(self):
        np.random.seed(11)
        x_np = np.random.randn(4, 16).astype(np.float32)
        x = torch.tensor(x_np, device="mps")
        out = torch.nn.functional.rms_norm(x, [16])
        assert out.device.type == "mps"
        rms = np.sqrt(np.mean(x_np ** 2, axis=-1, keepdims=True) + 1e-6)
        ref = x_np / rms
        np.testing.assert_allclose(out.cpu().numpy(), ref, atol=1e-5)

    def test_rms_norm_f16(self):
        np.random.seed(12)
        x_np = np.random.randn(4, 8).astype(np.float32)
        x = torch.tensor(x_np, device="mps").to(torch.float16)
        out = torch.nn.functional.rms_norm(x, [8])
        assert out.device.type == "mps"
        assert out.dtype == torch.float16
        rms = np.sqrt(np.mean(x_np ** 2, axis=-1, keepdims=True) + 1e-6)
        ref = x_np / rms
        np.testing.assert_allclose(out.cpu().numpy(), ref.astype(np.float16), atol=0.05)

    def test_rms_norm_large(self):
        np.random.seed(13)
        x_np = np.random.randn(4, 128, 768).astype(np.float32)
        x = torch.tensor(x_np, device="mps")
        out = torch.nn.functional.rms_norm(x, [768])
        assert out.device.type == "mps"
        assert out.shape == (4, 128, 768)
        rms = np.sqrt(np.mean(x_np ** 2, axis=-1, keepdims=True) + 1e-6)
        ref = x_np / rms
        np.testing.assert_allclose(out.cpu().numpy(), ref, atol=1e-4)

    # --- batch_norm ---

    def test_batch_norm_eval_f32(self):
        np.random.seed(20)
        x_np = np.random.randn(2, 4, 8, 8).astype(np.float32)
        rm_np = np.random.randn(4).astype(np.float32)
        rv_np = np.abs(np.random.randn(4)).astype(np.float32) + 0.5
        w_np = np.random.randn(4).astype(np.float32)
        b_np = np.random.randn(4).astype(np.float32)
        x = torch.tensor(x_np, device="mps")
        rm = torch.tensor(rm_np, device="mps")
        rv = torch.tensor(rv_np, device="mps")
        w = torch.tensor(w_np, device="mps")
        b = torch.tensor(b_np, device="mps")
        out = torch.nn.functional.batch_norm(x, rm, rv, w, b, training=False)
        assert out.device.type == "mps"
        ref = (x_np - rm_np[None, :, None, None]) / \
              np.sqrt(rv_np[None, :, None, None] + 1e-5) * \
              w_np[None, :, None, None] + b_np[None, :, None, None]
        np.testing.assert_allclose(out.cpu().numpy(), ref, atol=1e-5)

    def test_batch_norm_eval_no_affine(self):
        np.random.seed(21)
        x_np = np.random.randn(2, 4, 6, 6).astype(np.float32)
        rm_np = np.zeros(4, dtype=np.float32)
        rv_np = np.ones(4, dtype=np.float32)
        x = torch.tensor(x_np, device="mps")
        rm = torch.tensor(rm_np, device="mps")
        rv = torch.tensor(rv_np, device="mps")
        out = torch.nn.functional.batch_norm(x, rm, rv, training=False)
        assert out.device.type == "mps"
        ref = (x_np - rm_np[None, :, None, None]) / \
              np.sqrt(rv_np[None, :, None, None] + 1e-5)
        np.testing.assert_allclose(out.cpu().numpy(), ref, atol=1e-5)

    def test_batch_norm_training(self):
        np.random.seed(22)
        x_np = np.random.randn(4, 8, 4, 4).astype(np.float32)
        rm_np = np.zeros(8, dtype=np.float32)
        rv_np = np.ones(8, dtype=np.float32)
        x = torch.tensor(x_np, device="mps")
        rm = torch.tensor(rm_np.copy(), device="mps")
        rv = torch.tensor(rv_np.copy(), device="mps")
        out = torch.nn.functional.batch_norm(x, rm, rv, training=True, momentum=0.1)
        assert out.device.type == "mps"
        assert out.shape == x.shape
        # Verify output is roughly normalized (mean ~0, std ~1 per channel)
        out_np = out.cpu().numpy()
        channel_means = out_np.mean(axis=(0, 2, 3))
        np.testing.assert_allclose(channel_means, np.zeros(8), atol=1e-4)

    def test_batch_norm_training_updates_running_stats(self):
        np.random.seed(23)
        x_np = np.random.randn(4, 4, 4, 4).astype(np.float32)
        rm_np = np.zeros(4, dtype=np.float32)
        rv_np = np.ones(4, dtype=np.float32)
        x = torch.tensor(x_np, device="mps")
        rm = torch.tensor(rm_np.copy(), device="mps")
        rv = torch.tensor(rv_np.copy(), device="mps")
        torch.nn.functional.batch_norm(x, rm, rv, training=True, momentum=0.1)
        # running_mean should no longer be all zeros
        rm_updated = rm.cpu().numpy()
        assert not np.allclose(rm_updated, np.zeros(4)), \
            "running_mean should be updated during training"

    def test_batch_norm_1d(self):
        """batch_norm on (N, C) input (no spatial dims)."""
        np.random.seed(24)
        x_np = np.random.randn(8, 4).astype(np.float32)
        rm_np = np.zeros(4, dtype=np.float32)
        rv_np = np.ones(4, dtype=np.float32)
        x = torch.tensor(x_np, device="mps")
        rm = torch.tensor(rm_np, device="mps")
        rv = torch.tensor(rv_np, device="mps")
        out = torch.nn.functional.batch_norm(x, rm, rv, training=False)
        assert out.device.type == "mps"
        ref = (x_np - rm_np[None, :]) / np.sqrt(rv_np[None, :] + 1e-5)
        np.testing.assert_allclose(out.cpu().numpy(), ref, atol=1e-5)

    def test_batch_norm_f16(self):
        np.random.seed(25)
        x_np = np.random.randn(2, 4, 4, 4).astype(np.float32)
        rm_np = np.zeros(4, dtype=np.float32)
        rv_np = np.ones(4, dtype=np.float32)
        x = torch.tensor(x_np, device="mps").to(torch.float16)
        rm = torch.tensor(rm_np, device="mps")
        rv = torch.tensor(rv_np, device="mps")
        out = torch.nn.functional.batch_norm(x, rm, rv, training=False)
        assert out.device.type == "mps"
        assert out.dtype == torch.float16
        ref = (x_np - rm_np[None, :, None, None]) / \
              np.sqrt(rv_np[None, :, None, None] + 1e-5)
        np.testing.assert_allclose(out.cpu().numpy(), ref.astype(np.float16),
                                   atol=0.05)
