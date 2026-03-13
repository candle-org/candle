"""Tests for nn.functional NPU implementations (generic, all SoCs)."""
import pytest
import numpy as np

try:
    import candle as torch
    from candle import nn
    NPU_AVAILABLE = torch.npu.is_available() if hasattr(torch, 'npu') else False
except ImportError:
    NPU_AVAILABLE = False

pytestmark = pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")


class TestActivationFunctionsNPU:
    """Test activation functions on NPU."""

    def test_silu_npu(self):
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device="npu")
        out = nn.functional.silu(x)

        # Compare with CPU result
        x_cpu = x.to("cpu")
        expected = nn.functional.silu(x_cpu)

        np.testing.assert_allclose(
            out.to("cpu").numpy(), expected.numpy(), rtol=1e-3, atol=1e-4
        )

    def test_leaky_relu_npu(self):
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device="npu")
        out = nn.functional.leaky_relu(x, negative_slope=0.1)

        x_cpu = x.to("cpu")
        expected = nn.functional.leaky_relu(x_cpu, negative_slope=0.1)

        np.testing.assert_allclose(
            out.to("cpu").numpy(), expected.numpy(), rtol=1e-3, atol=1e-4
        )

    def test_elu_npu(self):
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device="npu")
        out = nn.functional.elu(x, alpha=1.0)

        x_cpu = x.to("cpu")
        expected = nn.functional.elu(x_cpu, alpha=1.0)

        np.testing.assert_allclose(
            out.to("cpu").numpy(), expected.numpy(), rtol=1e-3, atol=1e-4
        )

    def test_prelu_npu(self):
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device="npu")
        weight = torch.tensor([0.25], device="npu")
        out = nn.functional.prelu(x, weight)

        x_cpu = x.to("cpu")
        weight_cpu = weight.to("cpu")
        expected = nn.functional.prelu(x_cpu, weight_cpu)

        np.testing.assert_allclose(
            out.to("cpu").numpy(), expected.numpy(), rtol=1e-3, atol=1e-4
        )


class TestNormalizationNPU:
    """Test normalization functions on NPU."""

    def test_group_norm_npu(self):
        # Input: (N=2, C=4, H=3, W=3)
        x = torch.randn(2, 4, 3, 3, device="npu")
        weight = torch.randn(4, device="npu")
        bias = torch.randn(4, device="npu")

        out = nn.functional.group_norm(x, num_groups=2, weight=weight, bias=bias)

        # Compare with CPU
        x_cpu = x.to("cpu")
        weight_cpu = weight.to("cpu")
        bias_cpu = bias.to("cpu")

        expected = nn.functional.group_norm(x_cpu, num_groups=2, weight=weight_cpu, bias=bias_cpu)

        np.testing.assert_allclose(
            out.to("cpu").numpy(), expected.numpy(), rtol=1e-3, atol=1e-4
        )

    def test_group_norm_no_affine_npu(self):
        x = torch.randn(2, 4, 3, 3, device="npu")
        out = nn.functional.group_norm(x, num_groups=2)

        x_cpu = x.to("cpu")
        expected = nn.functional.group_norm(x_cpu, num_groups=2)

        np.testing.assert_allclose(
            out.to("cpu").numpy(), expected.numpy(), rtol=1e-3, atol=1e-4
        )


class TestModulesNPU:
    """Test nn.Module classes with NPU backend."""

    def test_silu_module_npu(self):
        layer = nn.SiLU()
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device="npu")
        out = layer(x)

        x_cpu = x.to("cpu")
        expected = layer(x_cpu)

        np.testing.assert_allclose(
            out.to("cpu").numpy(), expected.numpy(), rtol=1e-3, atol=1e-4
        )

    def test_leaky_relu_module_npu(self):
        layer = nn.LeakyReLU(negative_slope=0.1)
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device="npu")
        out = layer(x)

        x_cpu = x.to("cpu")
        expected = layer(x_cpu)

        np.testing.assert_allclose(
            out.to("cpu").numpy(), expected.numpy(), rtol=1e-3, atol=1e-4
        )

    def test_elu_module_npu(self):
        layer = nn.ELU(alpha=1.0)
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device="npu")
        out = layer(x)

        x_cpu = x.to("cpu")
        expected = layer(x_cpu)

        np.testing.assert_allclose(
            out.to("cpu").numpy(), expected.numpy(), rtol=1e-3, atol=1e-4
        )

    def test_prelu_module_npu(self):
        layer = nn.PReLU(num_parameters=1, init=0.25)
        layer = layer.to("npu")
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device="npu")
        out = layer(x)

        layer_cpu = nn.PReLU(num_parameters=1, init=0.25)
        x_cpu = x.to("cpu")
        expected = layer_cpu(x_cpu)

        np.testing.assert_allclose(
            out.to("cpu").numpy(), expected.numpy(), rtol=1e-3, atol=1e-4
        )

    def test_group_norm_module_npu(self):
        layer = nn.GroupNorm(num_groups=2, num_channels=4)
        layer = layer.to("npu")

        x = torch.randn(2, 4, 3, 3, device="npu")
        out = layer(x)

        assert out.shape == x.shape
        assert out.device.type == "npu"