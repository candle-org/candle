import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

import candle as torch
from candle._tensor import Tensor


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_npu_mul_scalar_stays_on_npu():
    torch.manual_seed(1234)
    x = torch.randn(128, device="npu")
    scale = torch.tensor(3.0, device="npu")
    y = torch.mul(x, scale)

    assert y.device.type == "npu"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_npu_golden_ops_stay_on_npu():
    x = torch.randn((32, 8), device="npu")
    w = torch.randn((8, 4), device="npu")
    out = torch.matmul(x, w)
    diff = torch.sub(out, torch.zeros_like(out))
    loss = torch.mean(torch.pow(diff, 2.0))

    assert out.device.type == "npu"
    assert diff.device.type == "npu"
    assert loss.device.type == "npu"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_npu_tril_indices_respects_npu_device():
    out = torch.tril_indices(3, 4, device="npu")
    assert out.device.type == "npu"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_npu_triu_indices_respects_npu_device():
    out = torch.triu_indices(3, 4, device="npu")
    assert out.device.type == "npu"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_npu_cartesian_prod_no_cpu_roundtrip(monkeypatch):
    original_to = Tensor.to

    def guard_to(self, *args, **kwargs):
        device = None
        if args:
            device = args[0]
        elif "device" in kwargs:
            device = kwargs["device"]
        if getattr(self, "device", None) is not None and self.device.type == "npu":
            if device == "cpu" or getattr(device, "type", None) == "cpu":
                raise AssertionError("cartesian_prod should not move NPU tensors to CPU")
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(Tensor, "to", guard_to)
    a = torch.tensor([1.0, 2.0], device="npu")
    b = torch.tensor([3.0, 4.0], device="npu")

    out = torch.cartesian_prod(a, b)
    assert out.device.type == "npu"
    assert out.shape == (4, 2)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_npu_cartesian_prod_then_block_diag_does_not_crash():
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    python_path = [str(repo_root / "src")]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        python_path.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(python_path)

    code = textwrap.dedent(
        """
        import candle as torch

        a = torch.tensor([1.0, 2.0], device="npu")
        b = torch.tensor([3.0, 4.0], device="npu")
        torch.cartesian_prod(a, b)

        x = torch.tensor([[1.0, 2.0]], device="npu")
        y = torch.tensor([[3.0], [4.0]], device="npu")
        out = torch.block_diag(x, y)
        assert out.shape == (3, 3)
        print("ok")
        """
    )

    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, (
        f"subprocess failed with rc={result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_npu_block_diag_no_cpu_roundtrip(monkeypatch):
    original_to = Tensor.to

    def guard_to(self, *args, **kwargs):
        device = None
        if args:
            device = args[0]
        elif "device" in kwargs:
            device = kwargs["device"]
        if getattr(self, "device", None) is not None and self.device.type == "npu":
            if device == "cpu" or getattr(device, "type", None) == "cpu":
                raise AssertionError("block_diag should not move NPU tensors to CPU")
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(Tensor, "to", guard_to)
    a = torch.tensor([[1.0, 2.0]], device="npu")
    b = torch.tensor([[3.0], [4.0]], device="npu")

    out = torch.block_diag(a, b)
    assert out.device.type == "npu"
    assert out.shape == (3, 3)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_npu_repeat_interleave_tensor_repeats_stays_on_device(monkeypatch):
    original_to = Tensor.to

    def guard_to(self, *args, **kwargs):
        device = None
        if args:
            device = args[0]
        elif "device" in kwargs:
            device = kwargs["device"]
        if getattr(self, "device", None) is not None and self.device.type == "npu":
            if device == "cpu" or getattr(device, "type", None) == "cpu":
                raise AssertionError("repeat_interleave should not move NPU tensors to CPU")
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(Tensor, "to", guard_to)
    x = torch.tensor([[1, 2], [3, 4]], device="npu", dtype=torch.float32)
    repeats = torch.tensor([1, 2], device="npu", dtype=torch.int64)
    out = torch.repeat_interleave(x, repeats=repeats, dim=1)
    monkeypatch.setattr(Tensor, "to", original_to)

    assert out.device.type == "npu"
    assert out.shape == (2, 3)
    import numpy as np
    expected = np.array([[1, 2, 2], [3, 4, 4]], dtype=np.float32)
    np.testing.assert_allclose(out.to("cpu").numpy(), expected)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_npu_repeat_interleave_tensor_repeats_prefers_native_aclnn(monkeypatch):
    from candle._backends.npu import aclnn, ops_soc
    from candle._backends.npu.ops import shape as shape_ops

    if not getattr(aclnn, "repeat_interleave_tensor_symbols_ok", lambda: False)():
        pytest.skip("native tensor repeat_interleave symbols not available")

    x = torch.tensor([[1, 2], [3, 4]], device="npu", dtype=torch.float32)
    repeats = torch.tensor([1, 2], device="npu", dtype=torch.int64)

    if ops_soc.use_fallback("repeat_interleave_tensor"):
        used_composite = {"value": False}
        original = shape_ops._build_repeat_interleave_indices

        def track_composite(*args, **kwargs):
            used_composite["value"] = True
            return original(*args, **kwargs)

        monkeypatch.setattr(shape_ops, "_build_repeat_interleave_indices", track_composite)
        out = torch.repeat_interleave(x, repeats=repeats, dim=1)
        assert used_composite["value"]
    else:
        def fail_if_composite(*args, **kwargs):
            raise AssertionError("tensor repeat_interleave should use native ACLNN path when available")

        monkeypatch.setattr(shape_ops, "_build_repeat_interleave_indices", fail_if_composite)
        out = torch.repeat_interleave(x, repeats=repeats, dim=1)

    assert out.device.type == "npu"
    assert out.shape == (2, 3)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_npu_baddbmm_tensor_alpha_beta_stays_on_device():
    B, N, M, P = 2, 3, 5, 4
    self_tensor = torch.ones(B, N, P, dtype=torch.float32, device="npu")
    batch1 = torch.ones(B, N, M, dtype=torch.float32, device="npu")
    batch2 = torch.ones(B, M, P, dtype=torch.float32, device="npu")
    alpha = torch.tensor(0.5, device="npu")
    beta = torch.tensor(2.0, device="npu")
    out = torch.baddbmm(self_tensor, batch1, batch2, beta=beta, alpha=alpha)
    assert out.device.type == "npu"
    # 2.0 * 1 + 0.5 * (ones @ ones) = 2 + 0.5*5 = 4.5
    import numpy as np
    np.testing.assert_allclose(out.to("cpu").numpy(), np.full((B, N, P), 4.5))
