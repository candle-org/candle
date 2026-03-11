"""Tests for MPS GPU RNG (Philox 4x32-10)."""
import numpy as np
import pytest
import candle as torch


# ---------------------------------------------------------------------------
# Reproducibility: same seed → same output
# ---------------------------------------------------------------------------

class TestPhiloxReproducibility:
    """Verify deterministic RNG with manual_seed."""

    def test_rand_reproducible_f32(self):
        torch.manual_seed(42)
        a = torch.rand(1000, device="mps")
        torch.manual_seed(42)
        b = torch.rand(1000, device="mps")
        np.testing.assert_array_equal(a.cpu().numpy(), b.cpu().numpy())

    def test_rand_reproducible_f16(self):
        torch.manual_seed(42)
        a = torch.rand(1000, dtype=torch.float16, device="mps")
        torch.manual_seed(42)
        b = torch.rand(1000, dtype=torch.float16, device="mps")
        np.testing.assert_array_equal(a.cpu().numpy(), b.cpu().numpy())

    def test_randn_reproducible_f32(self):
        torch.manual_seed(123)
        a = torch.randn(1000, device="mps")
        torch.manual_seed(123)
        b = torch.randn(1000, device="mps")
        np.testing.assert_array_equal(a.cpu().numpy(), b.cpu().numpy())

    def test_randn_reproducible_f16(self):
        torch.manual_seed(123)
        a = torch.randn(1000, dtype=torch.float16, device="mps")
        torch.manual_seed(123)
        b = torch.randn(1000, dtype=torch.float16, device="mps")
        np.testing.assert_array_equal(a.cpu().numpy(), b.cpu().numpy())

    def test_randint_reproducible_i64(self):
        torch.manual_seed(99)
        a = torch.randint(0, 100, (500,), device="mps")
        torch.manual_seed(99)
        b = torch.randint(0, 100, (500,), device="mps")
        np.testing.assert_array_equal(a.cpu().numpy(), b.cpu().numpy())

    def test_randint_reproducible_i32(self):
        torch.manual_seed(99)
        a = torch.randint(0, 100, (500,), dtype=torch.int32, device="mps")
        torch.manual_seed(99)
        b = torch.randint(0, 100, (500,), dtype=torch.int32, device="mps")
        np.testing.assert_array_equal(a.cpu().numpy(), b.cpu().numpy())

    def test_randperm_reproducible(self):
        torch.manual_seed(77)
        a = torch.randperm(100, device="mps")
        torch.manual_seed(77)
        b = torch.randperm(100, device="mps")
        np.testing.assert_array_equal(a.cpu().numpy(), b.cpu().numpy())


# ---------------------------------------------------------------------------
# Independence: different seeds → different output
# ---------------------------------------------------------------------------

class TestPhiloxIndependence:
    """Verify different seeds produce different outputs."""

    def test_rand_different_seeds(self):
        torch.manual_seed(1)
        a = torch.rand(100, device="mps")
        torch.manual_seed(2)
        b = torch.rand(100, device="mps")
        assert not np.array_equal(a.cpu().numpy(), b.cpu().numpy())

    def test_randn_different_seeds(self):
        torch.manual_seed(1)
        a = torch.randn(100, device="mps")
        torch.manual_seed(2)
        b = torch.randn(100, device="mps")
        assert not np.array_equal(a.cpu().numpy(), b.cpu().numpy())

    def test_sequential_calls_different(self):
        torch.manual_seed(42)
        a = torch.rand(100, device="mps")
        b = torch.rand(100, device="mps")
        assert not np.array_equal(a.cpu().numpy(), b.cpu().numpy())


# ---------------------------------------------------------------------------
# Statistical correctness: distributions match expected properties
# ---------------------------------------------------------------------------

class TestPhiloxStatistics:
    """Verify output distributions are statistically correct."""

    def test_rand_range(self):
        torch.manual_seed(42)
        x = torch.rand(10000, device="mps")
        vals = x.cpu().numpy()
        assert vals.min() >= 0.0
        assert vals.max() < 1.0

    def test_rand_mean_std(self):
        torch.manual_seed(42)
        x = torch.rand(50000, device="mps")
        vals = x.cpu().numpy()
        # Uniform [0,1): mean ≈ 0.5, std ≈ 1/sqrt(12) ≈ 0.2887
        assert abs(vals.mean() - 0.5) < 0.01
        assert abs(vals.std() - 0.2887) < 0.01

    def test_randn_mean_std(self):
        torch.manual_seed(42)
        x = torch.randn(50000, device="mps")
        vals = x.cpu().numpy()
        # Standard normal: mean ≈ 0, std ≈ 1
        assert abs(vals.mean()) < 0.02
        assert abs(vals.std() - 1.0) < 0.02

    def test_randint_range(self):
        torch.manual_seed(42)
        x = torch.randint(10, 20, (5000,), device="mps")
        vals = x.cpu().numpy()
        assert vals.min() >= 10
        assert vals.max() < 20

    def test_randint_coverage(self):
        torch.manual_seed(42)
        x = torch.randint(0, 5, (10000,), device="mps")
        vals = x.cpu().numpy()
        unique = set(vals.tolist())
        assert unique == {0, 1, 2, 3, 4}

    def test_randperm_is_permutation(self):
        torch.manual_seed(42)
        x = torch.randperm(100, device="mps")
        vals = sorted(x.cpu().numpy().tolist())
        assert vals == list(range(100))


# ---------------------------------------------------------------------------
# In-place ops: uniform_, normal_, bernoulli_
# ---------------------------------------------------------------------------

class TestPhiloxInPlace:
    """Verify in-place RNG ops use GPU Philox."""

    def test_uniform_reproducible(self):
        torch.manual_seed(42)
        a = torch.empty(1000, device="mps")
        a.uniform_(0.0, 1.0)
        torch.manual_seed(42)
        b = torch.empty(1000, device="mps")
        b.uniform_(0.0, 1.0)
        np.testing.assert_array_equal(a.cpu().numpy(), b.cpu().numpy())

    def test_uniform_range(self):
        torch.manual_seed(42)
        a = torch.empty(5000, device="mps")
        a.uniform_(-3.0, 3.0)
        vals = a.cpu().numpy()
        assert vals.min() >= -3.0
        assert vals.max() < 3.0

    def test_normal_reproducible(self):
        torch.manual_seed(42)
        a = torch.empty(1000, device="mps")
        a.normal_(0.0, 1.0)
        torch.manual_seed(42)
        b = torch.empty(1000, device="mps")
        b.normal_(0.0, 1.0)
        np.testing.assert_array_equal(a.cpu().numpy(), b.cpu().numpy())

    def test_normal_statistics(self):
        torch.manual_seed(42)
        a = torch.empty(50000, device="mps")
        a.normal_(5.0, 2.0)
        vals = a.cpu().numpy()
        assert abs(vals.mean() - 5.0) < 0.05
        assert abs(vals.std() - 2.0) < 0.05

    def test_bernoulli_reproducible(self):
        torch.manual_seed(42)
        a = torch.empty(1000, device="mps")
        a.bernoulli_(0.3)
        torch.manual_seed(42)
        b = torch.empty(1000, device="mps")
        b.bernoulli_(0.3)
        np.testing.assert_array_equal(a.cpu().numpy(), b.cpu().numpy())

    def test_bernoulli_rate(self):
        torch.manual_seed(42)
        a = torch.empty(50000, device="mps")
        a.bernoulli_(0.7)
        vals = a.cpu().numpy()
        assert set(np.unique(vals).tolist()).issubset({0.0, 1.0})
        assert abs(vals.mean() - 0.7) < 0.02


# ---------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------

class TestPhiloxDropout:
    """Verify fused Philox dropout kernel."""

    def test_dropout_reproducible(self):
        torch.manual_seed(42)
        x = torch.randn(1000, device="mps")
        torch.manual_seed(100)
        out1 = torch.nn.functional.dropout(x, p=0.5, training=True)
        torch.manual_seed(100)
        out2 = torch.nn.functional.dropout(x, p=0.5, training=True)
        np.testing.assert_array_equal(out1.cpu().numpy(), out2.cpu().numpy())

    def test_dropout_zeros_fraction(self):
        torch.manual_seed(42)
        x = torch.ones(10000, device="mps")
        out = torch.nn.functional.dropout(x, p=0.5, training=True)
        vals = out.cpu().numpy()
        zero_frac = (vals == 0.0).mean()
        assert abs(zero_frac - 0.5) < 0.03

    def test_dropout_scaling(self):
        torch.manual_seed(42)
        x = torch.ones(10000, device="mps")
        out = torch.nn.functional.dropout(x, p=0.3, training=True)
        vals = out.cpu().numpy()
        nonzero = vals[vals != 0.0]
        expected_scale = 1.0 / (1.0 - 0.3)
        np.testing.assert_allclose(nonzero, expected_scale, rtol=1e-5)

    def test_dropout_eval_passthrough(self):
        x = torch.randn(100, device="mps")
        out = torch.nn.functional.dropout(x, p=0.5, training=False)
        np.testing.assert_array_equal(out.cpu().numpy(), x.cpu().numpy())


# ---------------------------------------------------------------------------
# Multi-dimensional shapes
# ---------------------------------------------------------------------------

class TestPhiloxShapes:
    """Verify RNG works with various tensor shapes."""

    def test_rand_2d(self):
        torch.manual_seed(42)
        a = torch.rand(10, 20, device="mps")
        assert a.shape == (10, 20)
        vals = a.cpu().numpy()
        assert vals.min() >= 0.0
        assert vals.max() < 1.0

    def test_randn_3d(self):
        torch.manual_seed(42)
        a = torch.randn(4, 8, 16, device="mps")
        assert a.shape == (4, 8, 16)

    def test_randint_2d(self):
        torch.manual_seed(42)
        a = torch.randint(0, 10, (5, 5), device="mps")
        assert a.shape == (5, 5)
        vals = a.cpu().numpy()
        assert vals.min() >= 0
        assert vals.max() < 10

    def test_rand_small(self):
        """Edge case: fewer than 4 elements (single Philox round)."""
        torch.manual_seed(42)
        a = torch.rand(3, device="mps")
        torch.manual_seed(42)
        b = torch.rand(3, device="mps")
        np.testing.assert_array_equal(a.cpu().numpy(), b.cpu().numpy())


# ---------------------------------------------------------------------------
# Generator state
# ---------------------------------------------------------------------------

class TestMPSGenerator:
    """Verify MPS generator state management."""

    def test_mps_manual_seed(self):
        torch.mps.manual_seed(42)
        a = torch.rand(100, device="mps")
        torch.mps.manual_seed(42)
        b = torch.rand(100, device="mps")
        np.testing.assert_array_equal(a.cpu().numpy(), b.cpu().numpy())

    def test_generator_device(self):
        gen = torch.mps._get_default_generator()
        assert gen.device.type == "mps"

    def test_generator_initial_seed(self):
        torch.mps.manual_seed(12345)
        gen = torch.mps._get_default_generator()
        assert gen.initial_seed() == 12345
