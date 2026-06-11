"""NPU dtype-only Tensor.to should bypass Python wrappers via Cython fast cast."""
import re
from pathlib import Path

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_npu_dtype_only_to_uses_cython_fast_path_source():
    """tensor_to must route eligible NPU dtype-only casts to fast_cast before dispatch."""
    src = (_REPO_ROOT / "src/candle/_C/_tensor_api.pyx").read_text(encoding="utf-8")
    assert "fast_cast as _cy_fast_npu_cast" in src
    assert "cdef class _NpuToCopyBackward" in src
    assert 'self._name = "ToCopyBackward0"' in src
    assert "cdef inline object _attach_npu_to_copy_grad" in src

    to_match = re.search(
        r"def tensor_to\(self, \*args, \*\*kwargs\):(?P<body>.*?)\n\ndef ",
        src,
        flags=re.DOTALL,
    )
    assert to_match is not None
    body = to_match.group("body")
    assert "_cy_fast_npu_cast(self, dtype)" in body
    assert "_attach_npu_to_copy_grad" in body
    assert body.index("_cy_fast_npu_cast(self, dtype)") < body.index("_to_dispatch_fn(")


def test_npu_dtype_only_to_attaches_cython_autograd_node(npu_device):
    """Autograd dtype-only NPU cast should use the Cython ToCopy node end to end."""
    import candle as torch

    values = np.linspace(-2.0, 2.0, num=16, dtype=np.float32)
    x = torch.tensor(values, device=npu_device, dtype=torch.float32, requires_grad=True)
    out = x.to(torch.float16)

    assert out.dtype == torch.float16
    assert out.device.type == "npu"
    assert out.requires_grad
    assert out.grad_fn is not None
    assert out.grad_fn.name() == "ToCopyBackward0"
    assert type(out.grad_fn).__module__ == "candle._C._tensor_api"

    out.sum().backward()
    torch.npu.synchronize()

    assert x.grad is not None
    assert x.grad.device.type == "npu"
    assert x.grad.dtype == torch.float32
    np.testing.assert_allclose(x.grad.to("cpu").numpy(), np.ones_like(values), rtol=0, atol=0)


def test_npu_dtype_only_to_inference_matches_reference(npu_device):
    """Inference dtype-only NPU cast must stay on device and keep exact cast values."""
    import candle as torch

    values = np.linspace(-3.0, 3.0, num=8, dtype=np.float32)
    x = torch.tensor(values, device=npu_device, dtype=torch.float32)
    out = x.to(torch.float16)

    assert out.grad_fn is None
    assert not out.requires_grad
    assert out.dtype == torch.float16
    assert out.device.type == "npu"
    np.testing.assert_allclose(
        out.to("cpu").numpy().astype(np.float32),
        values.astype(np.float16).astype(np.float32),
        rtol=0,
        atol=0,
    )


def test_npu_non_floating_target_keeps_generic_route(npu_device):
    """requires_grad float -> int casts must keep generic dispatch semantics."""
    import candle as torch

    values = np.array([1.25, -2.5, 3.0, 0.5], dtype=np.float32)
    x = torch.tensor(values, device=npu_device, dtype=torch.float32, requires_grad=True)
    out = x.to(torch.int64)

    assert out.dtype == torch.int64
    assert out.device.type == "npu"
    np.testing.assert_array_equal(
        out.to("cpu").numpy(),
        values.astype(np.int64),
    )
