"""Tests for MPS AMP (Automatic Mixed Precision) autocast support."""
import numpy as np
import pytest
import candle as torch


class TestMPSAutocastBasic:
    """Verify autocast context manager works for MPS."""

    def test_autocast_enters_and_exits(self):
        assert not torch.amp.autocast_mode.is_autocast_enabled("mps")
        with torch.amp.autocast("mps"):
            assert torch.amp.autocast_mode.is_autocast_enabled("mps")
        assert not torch.amp.autocast_mode.is_autocast_enabled("mps")

    def test_autocast_default_dtype_is_float16(self):
        with torch.amp.autocast("mps"):
            dtype = torch.amp.autocast_mode.get_autocast_dtype("mps")
            assert dtype == torch.float16

    def test_autocast_nested(self):
        with torch.amp.autocast("mps"):
            assert torch.amp.autocast_mode.is_autocast_enabled("mps")
            with torch.amp.autocast("mps", enabled=False):
                assert not torch.amp.autocast_mode.is_autocast_enabled("mps")
            assert torch.amp.autocast_mode.is_autocast_enabled("mps")

    def test_autocast_decorator(self):
        @torch.amp.autocast("mps")
        def fn(x):
            return x + 1.0
        x = torch.ones(4, device="mps")
        result = fn(x)
        assert result.device.type == "mps"


class TestMPSAutocastPolicy:
    """Verify autocast policies cast ops correctly on MPS."""

    def test_matmul_casts_to_float16(self):
        a = torch.randn(4, 4, device="mps", dtype=torch.float32)
        b = torch.randn(4, 4, device="mps", dtype=torch.float32)
        with torch.amp.autocast("mps"):
            out = torch.matmul(a, b)
        assert out.dtype == torch.float16

    def test_softmax_stays_float32(self):
        x = torch.randn(4, 4, device="mps", dtype=torch.float32)
        with torch.amp.autocast("mps"):
            out = torch.nn.functional.softmax(x, dim=1)
        assert out.dtype == torch.float32

    def test_add_stays_float32(self):
        """Element-wise add is not in LOWER_PRECISION_FP list."""
        a = torch.randn(4, device="mps", dtype=torch.float32)
        b = torch.randn(4, device="mps", dtype=torch.float32)
        with torch.amp.autocast("mps"):
            out = a + b
        assert out.dtype == torch.float32

    def test_linear_casts_to_float16(self):
        x = torch.randn(2, 8, device="mps", dtype=torch.float32)
        w = torch.randn(4, 8, device="mps", dtype=torch.float32)
        with torch.amp.autocast("mps"):
            out = torch.nn.functional.linear(x, w)
        assert out.dtype == torch.float16


class TestMPSGradScaler:
    """Verify GradScaler works with MPS tensors."""

    def test_grad_scaler_basic(self):
        scaler = torch.amp.GradScaler(device="mps")
        assert scaler._device == "mps"

    def test_grad_scaler_scale_and_step(self):
        model_w = torch.randn(4, 4, device="mps", requires_grad=True)
        x = torch.randn(2, 4, device="mps")

        scaler = torch.amp.GradScaler(device="mps")
        with torch.amp.autocast("mps"):
            out = torch.matmul(x, model_w.t())
            loss = out.sum()

        scaler.scale(loss).backward()
        scaler.step(torch.optim.SGD([model_w], lr=0.01))
        scaler.update()

    def test_grad_scaler_state_dict(self):
        scaler = torch.amp.GradScaler(device="mps")
        sd = scaler.state_dict()
        assert "scale" in sd
        assert "growth_factor" in sd
        assert "backoff_factor" in sd
