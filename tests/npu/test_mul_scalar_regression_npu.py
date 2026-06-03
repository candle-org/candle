import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

import candle as torch


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_mul_scalar_after_randn_on_npu():
    torch.manual_seed(1234)

    x = torch.randn(128, device="npu")
    scale = torch.tensor(3.0, device="npu")

    y = torch.mul(x, scale)
    assert y.device.type == "npu"
    assert y.shape == x.shape
    expected = x.to("cpu").numpy() * 3.0
    assert y.to("cpu").shape == x.to("cpu").shape
    assert pytest.approx(expected, rel=1e-6, abs=1e-6) == y.to("cpu").numpy()


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_mul_supports_zero_dim_npu_tensors_without_fallback():
    a = torch.tensor(2.0, device="npu")
    b = torch.tensor(3.0, device="npu")

    y = torch.mul(a, b)
    assert y.device.type == "npu"
    assert y.shape == ()
    assert y.item() == pytest.approx(6.0)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_mul_zero_dim_npu_parameter_keeps_gradients():
    x = torch.randn(16, device="npu")
    weight = torch.tensor(3.0, device="npu", requires_grad=True)

    y = torch.mul(x, weight).sum()
    y.backward()

    assert weight.grad is not None
    assert weight.grad.shape == ()
    assert weight.grad.device.type == "npu"
    assert weight.grad.item() == pytest.approx(x.to("cpu").sum().item(), rel=1e-5, abs=1e-5)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_mul_propagates_kernel_error_instead_of_falling_back():
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
        from candle._backends.npu import ops as npu_ops
        import candle._C._aclnn_ffi as ffi_mod

        original_resolve_op = ffi_mod.resolve_op

        def fail_mul_resolve(op_name):
            if op_name == "Mul":
                raise RuntimeError("sentinel mul failure")
            return original_resolve_op(op_name)

        def fail_div(*args, **kwargs):
            raise AssertionError("mul should not fall back to div")

        # Patch before the first Cython binary-op initialization.  The exact
        # torch.mul path cimports _aclnn_ffi.binary_op_no_alpha/execute, so
        # monkeypatching those Python attributes is not on the executing path;
        # resolve_op is the real initialization seam for Mul's ACLNN handles.
        ffi_mod.resolve_op = fail_mul_resolve
        npu_ops.div = fail_div

        x = torch.tensor([2.0, 3.0], device="npu")
        y = torch.tensor([4.0, 5.0], device="npu")
        try:
            torch.mul(x, y)
        except RuntimeError as exc:
            if "sentinel mul failure" not in str(exc):
                raise
        else:
            raise AssertionError("mul did not propagate the sentinel Mul failure")
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
