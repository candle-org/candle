import ctypes
import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_exact_base_npu_pair_guard_requires_same_device_index():
    """Exact Cython NPU pair fast path must not bypass cross-device checks."""
    src = (_REPO_ROOT / "src/candle/_C/_functional_ops.pyx").read_text(encoding="utf-8")
    match = re.search(
        r"cdef inline bint _exact_base_npu_pair\(object a, object b\):(?P<body>.*?)\n\n",
        src,
        flags=re.DOTALL,
    )
    assert match is not None
    body = match.group("body")
    assert "_device_index" in body
    assert "==" in body


def test_small_op_fast_paths_skip_python_defer_executor(npu_device, monkeypatch):
    """add/mul/silu should append raw executor handles without Python normalization."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.randn(4, 4, device=npu_device)
    b = torch.randn(4, 4, device=npu_device)
    torch.npu.synchronize()

    # Warm up so first-use imports/cache population happen before the patch.
    # _ensure_ffi_binary() caches a reference to aclnn._defer_executor on first
    # call; warming up here ensures the real function (not the stub below) is the
    # one cached, so the patch cannot leak into the Cython global and poison
    # later tests.
    _ = torch.add(a, b)
    _ = torch.mul(a, b)
    _ = torch.nn.functional.silu(a)
    torch.npu.synchronize()

    calls = {"count": 0}

    def fail_defer_executor(*args, **kwargs):
        calls["count"] += 1
        raise AssertionError("small-op fast path should append executor handles directly")

    monkeypatch.setattr(aclnn_mod, "_defer_executor", fail_defer_executor)

    _ = torch.add(a, b)
    _ = torch.mul(a, b)
    _ = torch.nn.functional.silu(a)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_top_level_npu_mul_skips_python_dispatch(npu_device, monkeypatch):
    """torch.mul should route NPU tensor pairs through the Cython hot wrapper."""
    import candle as torch
    import candle._functional as functional_mod

    a = torch.randn(4, 4, device=npu_device)
    b = torch.randn(4, 4, device=npu_device)
    torch.npu.synchronize()

    def fail_dispatch(*args, **kwargs):
        raise AssertionError("NPU tensor mul should bypass Python dispatch")

    monkeypatch.setattr(functional_mod, "dispatch", fail_dispatch)

    out = torch.mul(a, b)
    torch.npu.synchronize()

    assert out.device.type == "npu"


def test_nn_silu_npu_skips_python_dispatch(npu_device, monkeypatch):
    """F.silu should route NPU inference tensors directly to the fast kernel."""
    import candle as torch
    import candle._dispatch as dispatch_mod
    import candle.nn.functional as F

    x = torch.randn(4, 4, device=npu_device)
    torch.npu.synchronize()

    def fail_dispatch(*args, **kwargs):
        raise AssertionError("NPU silu inference fast path should bypass dispatch")

    monkeypatch.setattr(dispatch_mod, "dispatch", fail_dispatch)

    out = F.silu(x)
    torch.npu.synchronize()

    assert out.device.type == "npu"


def test_nn_silu_npu_autocast_uses_dispatch(npu_device, monkeypatch):
    """F.silu should preserve autocast handling by falling back to dispatch."""
    import candle as torch
    import candle._dispatch as dispatch_mod
    import candle.nn.functional as F

    x = torch.randn(4, 4, device=npu_device)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_dispatch = dispatch_mod.dispatch

    def wrapped_dispatch(*args, **kwargs):
        calls["count"] += 1
        return original_dispatch(*args, **kwargs)

    def fail_fast(*args, **kwargs):
        raise AssertionError("NPU silu fast path should not run under autocast")

    monkeypatch.setattr(dispatch_mod, "dispatch", wrapped_dispatch)
    monkeypatch.setattr(F, "_npu_silu_fast", fail_fast)

    with torch.amp.autocast("npu"):
        out = F.silu(x)
    torch.npu.synchronize()

    assert calls["count"] >= 1
    assert out.device.type == "npu"


def test_fast_add_skips_storage_method_calls(npu_device, monkeypatch):
    """fast_add should read device pointers directly, not via Tensor.storage()."""
    import candle as torch

    a = torch.randn(4, 4, device=npu_device)
    b = torch.randn(4, 4, device=npu_device)
    torch.npu.synchronize()

    # Warm up once so first-use imports/cache population do not affect the assertion.
    _ = torch.add(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    tensor_type = type(a)
    original_storage = tensor_type.storage

    def wrapped_storage(self, *args, **kwargs):
        calls["count"] += 1
        return original_storage(self, *args, **kwargs)

    monkeypatch.setattr(tensor_type, "storage", wrapped_storage)

    _ = torch.add(a, b)

    assert calls["count"] == 0


def test_fast_add_skips_ctypes_void_p_wrapper(npu_device, monkeypatch):
    """fast_add should defer executors by raw handle, without ctypes.c_void_p."""
    import candle as torch

    a = torch.randn(4, 4, device=npu_device)
    b = torch.randn(4, 4, device=npu_device)

    # Warm up once so first-use imports/cache population do not affect the assertion.
    _ = torch.add(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_c_void_p = ctypes.c_void_p

    def wrapped_c_void_p(value):
        calls["count"] += 1
        return original_c_void_p(value)

    monkeypatch.setattr(ctypes, "c_void_p", wrapped_c_void_p)

    _ = torch.add(a, b)

    assert calls["count"] == 0


def test_fast_add_same_signature_uses_cython_executor_path(npu_device, monkeypatch):
    """fast_add should bypass the Python FFI wrapper for stable signatures."""
    import numpy as np
    import candle as torch
    import candle._C._aclnn_ffi as ffi_mod  # pylint: disable=import-error,no-name-in-module

    a = torch.randn(7, 5, 3, device=npu_device)
    b = torch.randn(7, 5, 3, device=npu_device)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_binary_op_with_alpha = ffi_mod.binary_op_with_alpha

    def wrapped_binary_op_with_alpha(*args, **kwargs):
        calls["count"] += 1
        return original_binary_op_with_alpha(*args, **kwargs)

    monkeypatch.setattr(ffi_mod, "binary_op_with_alpha", wrapped_binary_op_with_alpha)

    out1 = torch.add(a, b)
    out2 = torch.add(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0
    expected = a.cpu().numpy() + b.cpu().numpy()
    assert np.allclose(out1.cpu().numpy(), expected, rtol=1e-4, atol=1e-4)
    assert np.allclose(out2.cpu().numpy(), expected, rtol=1e-4, atol=1e-4)


def test_fast_add_chain_reuses_cached_executor_across_distinct_outputs(npu_device, monkeypatch):
    """A chain of same-signature adds must reuse the PTA cached executor even when
    each op writes to a freshly allocated output pointer.

    This is the pipelined-model regime: outputs differ every op, but the
    (shape, stride, dtype) signature is constant.  torch_npu rebinds tensor
    addresses onto the cached executor (AddTensorAddrToCachedList), so after the
    cache is warm NO op should fall back to the full GetWorkspaceSize path.
    """
    import numpy as np
    import pytest
    import candle as torch
    import candle._C._aclnn_ffi as ffi_mod  # pylint: disable=import-error,no-name-in-module

    if not ffi_mod.is_pta_cache_available():
        pytest.skip("PTA executor cache not available on this CANN build")

    a = torch.randn(1, 128, 64, device=npu_device, dtype=torch.float16)
    b = torch.randn(1, 128, 64, device=npu_device, dtype=torch.float16)
    torch.npu.synchronize()

    # Warm up: populate the PTA cache for this signature.
    warm = [torch.add(a, b) for _ in range(5)]
    torch.npu.synchronize()
    del warm

    calls = {"count": 0}
    original_binary_op_with_alpha = ffi_mod.binary_op_with_alpha

    def wrapped_binary_op_with_alpha(*args, **kwargs):
        calls["count"] += 1
        return original_binary_op_with_alpha(*args, **kwargs)

    monkeypatch.setattr(ffi_mod, "binary_op_with_alpha", wrapped_binary_op_with_alpha)

    # Same inputs, distinct output pointer each op (outputs kept alive).
    outs = [torch.add(a, b) for _ in range(10)]
    torch.npu.synchronize()

    # Every op should hit the cached executor; none should rebuild via GetWorkspaceSize.
    assert calls["count"] == 0

    # Rebinding must still produce correct results.
    expected = a.cpu().float().numpy() + b.cpu().float().numpy()
    got = outs[-1].cpu().float().numpy()
    assert np.allclose(got, expected, rtol=1e-2, atol=1e-2)


def test_fast_mul_same_signature_uses_cython_executor_path(npu_device, monkeypatch):
    """fast_mul should bypass the Python FFI wrapper for stable signatures."""
    import numpy as np
    import candle as torch
    import candle._C._aclnn_ffi as ffi_mod  # pylint: disable=import-error,no-name-in-module

    a = torch.randn(7, 5, 3, device=npu_device)
    b = torch.randn(7, 5, 3, device=npu_device)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_binary_op_no_alpha = ffi_mod.binary_op_no_alpha

    def wrapped_binary_op_no_alpha(*args, **kwargs):
        calls["count"] += 1
        return original_binary_op_no_alpha(*args, **kwargs)

    monkeypatch.setattr(ffi_mod, "binary_op_no_alpha", wrapped_binary_op_no_alpha)

    out1 = torch.mul(a, b)
    out2 = torch.mul(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0
    expected = a.cpu().numpy() * b.cpu().numpy()
    assert np.allclose(out1.cpu().numpy(), expected, rtol=1e-4, atol=1e-4)
    assert np.allclose(out2.cpu().numpy(), expected, rtol=1e-4, atol=1e-4)


def test_fast_mul_chain_reuses_cached_executor_across_distinct_outputs(npu_device, monkeypatch):
    """Same-shape Mul may reuse PTA cached executors across fresh outputs."""
    import numpy as np
    import pytest
    import candle as torch
    import candle._C._aclnn_ffi as ffi_mod  # pylint: disable=import-error,no-name-in-module

    if not ffi_mod.is_pta_cache_available():
        pytest.skip("PTA executor cache not available on this CANN build")

    a = torch.randn(1, 128, 64, device=npu_device, dtype=torch.float16)
    b = torch.randn(1, 128, 64, device=npu_device, dtype=torch.float16)
    torch.npu.synchronize()

    warm = [torch.mul(a, b) for _ in range(5)]
    torch.npu.synchronize()
    del warm

    calls = {"count": 0}
    original_binary_op_no_alpha = ffi_mod.binary_op_no_alpha

    def wrapped_binary_op_no_alpha(*args, **kwargs):
        calls["count"] += 1
        return original_binary_op_no_alpha(*args, **kwargs)

    monkeypatch.setattr(ffi_mod, "binary_op_no_alpha", wrapped_binary_op_no_alpha)

    outs = [torch.mul(a, b) for _ in range(10)]
    torch.npu.synchronize()

    assert calls["count"] == 0
    expected = a.cpu().float().numpy() * b.cpu().float().numpy()
    got = outs[-1].cpu().float().numpy()
    assert np.allclose(got, expected, rtol=1e-2, atol=1e-2)


def test_fast_mul_broadcast_chain_remains_correct_across_distinct_outputs(npu_device, monkeypatch):
    """Broadcast Mul should remain correct while bypassing the Python FFI wrapper."""
    import numpy as np
    import candle as torch
    import candle._C._aclnn_ffi as ffi_mod  # pylint: disable=import-error,no-name-in-module

    a = torch.randn(1, 128, 64, device=npu_device, dtype=torch.float16)
    b = torch.randn(1, 1, 64, device=npu_device, dtype=torch.float16)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_binary_op_no_alpha = ffi_mod.binary_op_no_alpha

    def wrapped_binary_op_no_alpha(*args, **kwargs):
        calls["count"] += 1
        return original_binary_op_no_alpha(*args, **kwargs)

    monkeypatch.setattr(ffi_mod, "binary_op_no_alpha", wrapped_binary_op_no_alpha)

    outs = [torch.mul(a, b) for _ in range(10)]
    torch.npu.synchronize()

    assert calls["count"] == 0
    expected = a.cpu().float().numpy() * b.cpu().float().numpy()
    got = outs[-1].cpu().float().numpy()
    assert np.allclose(got, expected, rtol=1e-2, atol=1e-2)


def test_npu_gelu_functional_wrapper_uses_exact_base_route():
    """GELU should use the exact TensorImpl fast path like SiLU does."""
    src = (_REPO_ROOT / "src/candle/_C/_functional_ops.pyx").read_text(encoding="utf-8")
    assert "fast_gelu_exact as _cy_fast_npu_gelu_exact" in src
    match = re.search(r"def gelu\(a, approximate='none'\):(?P<body>.*?)\n\ndef ", src, flags=re.DOTALL)
    assert match is not None
    body = match.group("body")
    assert "_cy_fast_npu_gelu_exact(<TensorImpl>a)" in body


def test_fast_gelu_exact_defers_executor_on_error_path():
    """GELU exact miss path should defer executor cleanup even if execution raises."""
    src = (_REPO_ROOT / "src/candle/_C/_npu_ops.pyx").read_text(encoding="utf-8")
    match = re.search(
        r"cpdef object fast_gelu_exact\(TensorImpl a\):(?P<body>.*?)\n\ncpdef object fast_silu\(",
        src,
        flags=re.DOTALL,
    )
    assert match is not None
    body = match.group("body")
    assert "executor = 0" in body
    assert re.search(
        r"finally:\n\s+if executor:\n\s+_defer_executor_handle\(executor\)",
        body,
    ) is not None



def test_fast_gelu_skips_python_defer_executor_on_first_use(npu_device):
    """GELU hot path should append raw executor handles without Python normalization."""
    import os
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        f"""
        import candle as torch
        import candle.nn.functional as F
        import candle._backends.npu.aclnn as aclnn_mod

        x = torch.randn(1, 128, 64, device={str(npu_device)!r}, dtype=torch.float16)
        torch.npu.synchronize()

        calls = {{"count": 0}}
        original = aclnn_mod._defer_executor
        def wrapped(*args, **kwargs):
            calls["count"] += 1
            return original(*args, **kwargs)
        aclnn_mod._defer_executor = wrapped

        out = F.gelu(x)
        torch.npu.synchronize()

        assert calls["count"] == 0, calls
        assert out.device.type == "npu"
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stdout + result.stderr



def test_fast_gelu_exact_respects_storage_offset_view(npu_device):
    """GELU exact path should read from a view's storage_offset, not storage base."""
    import math
    import numpy as np
    import candle as torch
    import candle.nn.functional as F

    base = torch.randn(2, 32, device=npu_device, dtype=torch.float16)
    view = base[1]
    out = F.gelu(view)
    torch.npu.synchronize()

    view_np = view.cpu().float().numpy()
    expected = 0.5 * view_np * (1.0 + np.vectorize(math.erf)(view_np / math.sqrt(2.0)))
    got = out.cpu().float().numpy()
    assert np.allclose(got, expected, rtol=2e-2, atol=2e-2)


def test_fast_gelu_dispatch_respects_storage_offset_view(npu_device):
    """Dispatch GELU path should also honor storage_offset for views."""
    import math
    import numpy as np
    import candle as torch
    from candle._dispatch import dispatch

    base = torch.randn(2, 32, device=npu_device, dtype=torch.float16)
    view = base[1]
    out = dispatch("gelu", view.device.type, view)
    torch.npu.synchronize()

    view_np = view.cpu().float().numpy()
    expected = 0.5 * view_np * (1.0 + np.vectorize(math.erf)(view_np / math.sqrt(2.0)))
    got = out.cpu().float().numpy()
    assert np.allclose(got, expected, rtol=2e-2, atol=2e-2)


def test_npu_gelu_tanh_preserves_input_dtype_for_composite_constants(npu_device):
    """GELU tanh approximation should work on NPU float16 without mixed-dtype pow."""
    import numpy as np
    import math
    import candle as torch
    import candle.nn.functional as F

    x = torch.randn(16, 16, device=npu_device, dtype=torch.float16)
    out = F.gelu(x, approximate="tanh")
    torch.npu.synchronize()

    x_np = x.cpu().float().numpy()
    expected = 0.5 * x_np * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x_np + 0.044715 * x_np ** 3)))
    got = out.cpu().float().numpy()
    assert out.dtype == x.dtype
    assert np.allclose(got, expected, rtol=3e-2, atol=3e-2)

    x_grad = torch.randn(8, 8, device=npu_device, dtype=torch.float16, requires_grad=True)
    F.gelu(x_grad, approximate="tanh").sum().backward()
    torch.npu.synchronize()
    assert x_grad.grad is not None
    assert x_grad.grad.device.type == "npu"
    assert x_grad.grad.dtype == x_grad.dtype



def test_fast_gelu_chain_reuses_cached_executor_across_distinct_outputs(npu_device, monkeypatch):
    """Same-signature gelu chains should reuse PTA cached executors across fresh outputs."""
    import numpy as np
    import math
    import pytest
    import candle as torch
    import candle.nn.functional as F
    import candle._C._aclnn_ffi as ffi_mod  # pylint: disable=import-error,no-name-in-module

    if not ffi_mod.is_pta_cache_available():
        pytest.skip("PTA executor cache not available on this CANN build")

    x = torch.randn(1, 128, 64, device=npu_device, dtype=torch.float16)
    torch.npu.synchronize()

    warm = [F.gelu(x) for _ in range(5)]
    torch.npu.synchronize()
    del warm

    calls = {"count": 0}
    original_unary_op = ffi_mod.unary_op

    def wrapped_unary_op(*args, **kwargs):
        calls["count"] += 1
        return original_unary_op(*args, **kwargs)

    monkeypatch.setattr(ffi_mod, "unary_op", wrapped_unary_op)

    outs = [F.gelu(x) for _ in range(10)]
    torch.npu.synchronize()

    assert calls["count"] == 0
    x_np = x.cpu().float().numpy()
    expected = 0.5 * x_np * (1.0 + np.vectorize(math.erf)(x_np / math.sqrt(2.0)))
    got = outs[-1].cpu().float().numpy()
    assert np.allclose(got, expected, rtol=2e-2, atol=2e-2)


def test_fast_gelu_same_signature_uses_cython_executor_path(npu_device, monkeypatch):
    """fast_gelu_exact should bypass the Python FFI wrapper for stable signatures."""
    import numpy as np
    import math
    import candle as torch
    import candle.nn.functional as F
    import candle._C._aclnn_ffi as ffi_mod  # pylint: disable=import-error,no-name-in-module

    x = torch.randn(7, 5, 3, device=npu_device, dtype=torch.float16)
    torch.npu.synchronize()

    # Warm so first-use cache population does not affect the assertion
    _ = F.gelu(x)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_unary_op = ffi_mod.unary_op

    def wrapped_unary_op(*args, **kwargs):
        calls["count"] += 1
        return original_unary_op(*args, **kwargs)

    monkeypatch.setattr(ffi_mod, "unary_op", wrapped_unary_op)

    out1 = F.gelu(x)
    out2 = F.gelu(x)
    torch.npu.synchronize()

    assert calls["count"] == 0
    x_np = x.cpu().float().numpy()
    expected = 0.5 * x_np * (1.0 + np.vectorize(math.erf)(x_np / math.sqrt(2.0)))
    assert np.allclose(out1.cpu().float().numpy(), expected, rtol=2e-2, atol=2e-2)
    assert np.allclose(out2.cpu().float().numpy(), expected, rtol=2e-2, atol=2e-2)


def test_fast_silu_chain_reuses_cached_executor_across_distinct_outputs(npu_device, monkeypatch):
    """Same-signature silu chains should reuse PTA cached executors across fresh outputs."""
    import numpy as np
    import pytest
    import candle as torch
    import candle.nn.functional as F
    import candle._C._aclnn_ffi as ffi_mod  # pylint: disable=import-error,no-name-in-module

    if not ffi_mod.is_pta_cache_available():
        pytest.skip("PTA executor cache not available on this CANN build")

    x = torch.randn(1, 128, 64, device=npu_device, dtype=torch.float16)
    torch.npu.synchronize()

    warm = [F.silu(x) for _ in range(5)]
    torch.npu.synchronize()
    del warm

    calls = {"count": 0}
    original_unary_op = ffi_mod.unary_op

    def wrapped_unary_op(*args, **kwargs):
        calls["count"] += 1
        return original_unary_op(*args, **kwargs)

    monkeypatch.setattr(ffi_mod, "unary_op", wrapped_unary_op)

    outs = [F.silu(x) for _ in range(10)]
    torch.npu.synchronize()

    assert calls["count"] == 0
    x_np = x.cpu().float().numpy()
    expected = x_np / (1.0 + np.exp(-x_np))
    got = outs[-1].cpu().float().numpy()
    assert np.allclose(got, expected, rtol=2e-2, atol=2e-2)


def test_fast_silu_same_signature_uses_cython_executor_path(npu_device, monkeypatch):
    """fast_silu should bypass the Python FFI wrapper for stable signatures."""
    import numpy as np
    import candle as torch
    import candle._C._aclnn_ffi as ffi_mod  # pylint: disable=import-error,no-name-in-module

    x = torch.randn(7, 5, 3, device=npu_device)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_unary_op = ffi_mod.unary_op

    def wrapped_unary_op(*args, **kwargs):
        calls["count"] += 1
        return original_unary_op(*args, **kwargs)

    monkeypatch.setattr(ffi_mod, "unary_op", wrapped_unary_op)

    out1 = torch.nn.functional.silu(x)
    out2 = torch.nn.functional.silu(x)
    torch.npu.synchronize()

    assert calls["count"] == 0
    x_np = x.cpu().float().numpy()
    expected = x_np / (1.0 + np.exp(-x_np))
    assert np.allclose(out1.cpu().float().numpy(), expected, rtol=2e-2, atol=2e-2)
    assert np.allclose(out2.cpu().float().numpy(), expected, rtol=2e-2, atol=2e-2)


def test_fast_mul_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """fast_mul should bypass candle._backends.npu.aclnn.mul on the hot path."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.randn(4, 4, device=npu_device)
    b = torch.randn(4, 4, device=npu_device)
    torch.npu.synchronize()

    # Warm up so first-use imports/caches do not affect assertion
    _ = torch.mul(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_mul = aclnn_mod.mul

    def wrapped_mul(*args, **kwargs):
        calls["count"] += 1
        return original_mul(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "mul", wrapped_mul)

    result = torch.mul(a, b)
    torch.npu.synchronize()

    # Assertion (a): hot path must not call Python aclnn.mul
    assert calls["count"] == 0, (
        f"fast_mul called aclnn.mul {calls['count']} time(s); expected 0"
    )

    # Assertion (b): numerical correctness
    import numpy as np
    expected = a.cpu().numpy() * b.cpu().numpy()
    actual = result.cpu().numpy()
    assert np.allclose(actual, expected, rtol=1e-4, atol=1e-4), (
        f"fast_mul result mismatch: max_diff={np.abs(actual - expected).max()}"
    )


def test_fast_sub_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """fast_sub should bypass candle._backends.npu.aclnn.sub on the hot path."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod
    import numpy as np

    a = torch.randn(4, 4, device=npu_device)
    b = torch.randn(4, 4, device=npu_device)
    torch.npu.synchronize()

    expected = torch.sub(a, b)
    torch.npu.synchronize()

    _ = torch.sub(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_sub = aclnn_mod.sub

    def wrapped_sub(*args, **kwargs):
        calls["count"] += 1
        return original_sub(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "sub", wrapped_sub)

    out = torch.sub(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0, (
        f"fast_sub called aclnn.sub {calls['count']} time(s); expected 0"
    )
    assert np.allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-4, atol=1e-4), (
        "fast_sub output differs from expected"
    )


def test_fast_div_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """fast_div should bypass candle._backends.npu.aclnn.div on the hot path."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod
    import numpy as np

    a = torch.randn(4, 4, device=npu_device)
    b = torch.rand(4, 4, device=npu_device) + 1.0
    torch.npu.synchronize()

    expected = torch.div(a, b)
    torch.npu.synchronize()

    _ = torch.div(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_div = aclnn_mod.div

    def wrapped_div(*args, **kwargs):
        calls["count"] += 1
        return original_div(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "div", wrapped_div)

    out = torch.div(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0, (
        f"fast_div called aclnn.div {calls['count']} time(s); expected 0"
    )
    assert np.allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-4, atol=1e-4), (
        "fast_div output differs from expected"
    )


def test_fast_atan2_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """generic fast_binary_op path should bypass candle._backends.npu.aclnn.atan2."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.randn(4, 4, device=npu_device)
    b = torch.randn(4, 4, device=npu_device)
    torch.npu.synchronize()

    _ = torch.atan2(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_atan2 = aclnn_mod.atan2

    def wrapped_atan2(*args, **kwargs):
        calls["count"] += 1
        return original_atan2(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "atan2", wrapped_atan2)

    _ = torch.atan2(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_fast_logaddexp_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """generic fast_binary_op path should bypass candle._backends.npu.aclnn.slogaddexp."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.randn(4, 4, device=npu_device)
    b = torch.randn(4, 4, device=npu_device)
    torch.npu.synchronize()

    _ = torch.logaddexp(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_logaddexp = aclnn_mod.slogaddexp

    def wrapped_logaddexp(*args, **kwargs):
        calls["count"] += 1
        return original_logaddexp(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "slogaddexp", wrapped_logaddexp)

    _ = torch.logaddexp(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_fast_logaddexp2_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """generic fast_binary_op path should bypass candle._backends.npu.aclnn.slogaddexp2."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.randn(4, 4, device=npu_device)
    b = torch.randn(4, 4, device=npu_device)
    torch.npu.synchronize()

    _ = torch.logaddexp2(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_logaddexp2 = aclnn_mod.slogaddexp2

    def wrapped_logaddexp2(*args, **kwargs):
        calls["count"] += 1
        return original_logaddexp2(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "slogaddexp2", wrapped_logaddexp2)

    _ = torch.logaddexp2(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_fast_remainder_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """generic fast_binary_op path should bypass candle._backends.npu.aclnn.sremainder."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.randn(4, 4, device=npu_device)
    b = torch.randn(4, 4, device=npu_device) + 1.0
    torch.npu.synchronize()

    _ = torch.remainder(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_remainder = aclnn_mod.sremainder

    def wrapped_remainder(*args, **kwargs):
        calls["count"] += 1
        return original_remainder(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "sremainder", wrapped_remainder)

    _ = torch.remainder(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_fast_fmod_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """generic fast_binary_op path should bypass candle._backends.npu.aclnn.sfmod."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.randn(4, 4, device=npu_device)
    b = torch.randn(4, 4, device=npu_device) + 1.0
    torch.npu.synchronize()

    _ = torch.fmod(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_fmod = aclnn_mod.sfmod

    def wrapped_fmod(*args, **kwargs):
        calls["count"] += 1
        return original_fmod(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "sfmod", wrapped_fmod)

    _ = torch.fmod(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_fast_logical_and_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """generic fast_binary_op path should bypass candle._backends.npu.aclnn.logical_and."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.tensor([[True, False], [True, True]], device=npu_device)
    b = torch.tensor([[True, True], [False, True]], device=npu_device)
    torch.npu.synchronize()

    _ = torch.logical_and(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_logical_and = aclnn_mod.logical_and

    def wrapped_logical_and(*args, **kwargs):
        calls["count"] += 1
        return original_logical_and(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "logical_and", wrapped_logical_and)

    _ = torch.logical_and(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_fast_logical_or_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """generic fast_binary_op path should bypass candle._backends.npu.aclnn.logical_or."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.tensor([[True, False], [True, True]], device=npu_device)
    b = torch.tensor([[True, True], [False, True]], device=npu_device)
    torch.npu.synchronize()

    _ = torch.logical_or(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_logical_or = aclnn_mod.logical_or

    def wrapped_logical_or(*args, **kwargs):
        calls["count"] += 1
        return original_logical_or(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "logical_or", wrapped_logical_or)

    _ = torch.logical_or(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_fast_bitwise_and_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """generic fast_binary_op path should bypass candle._backends.npu.aclnn.bitwise_and."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.tensor([[1, 3], [7, 15]], dtype=torch.int64, device=npu_device)
    b = torch.tensor([[1, 1], [3, 7]], dtype=torch.int64, device=npu_device)
    torch.npu.synchronize()

    _ = torch.bitwise_and(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_bitwise_and = aclnn_mod.bitwise_and

    def wrapped_bitwise_and(*args, **kwargs):
        calls["count"] += 1
        return original_bitwise_and(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "bitwise_and", wrapped_bitwise_and)

    _ = torch.bitwise_and(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_fast_bitwise_or_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """generic fast_binary_op path should bypass candle._backends.npu.aclnn.bitwise_or."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.tensor([[1, 3], [7, 15]], dtype=torch.int64, device=npu_device)
    b = torch.tensor([[1, 1], [3, 7]], dtype=torch.int64, device=npu_device)
    torch.npu.synchronize()

    _ = torch.bitwise_or(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_bitwise_or = aclnn_mod.bitwise_or

    def wrapped_bitwise_or(*args, **kwargs):
        calls["count"] += 1
        return original_bitwise_or(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "bitwise_or", wrapped_bitwise_or)

    _ = torch.bitwise_or(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_fast_bitwise_xor_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """generic fast_binary_op path should bypass candle._backends.npu.aclnn.bitwise_xor."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.tensor([[1, 3], [7, 15]], dtype=torch.int64, device=npu_device)
    b = torch.tensor([[1, 1], [3, 7]], dtype=torch.int64, device=npu_device)
    torch.npu.synchronize()

    _ = torch.bitwise_xor(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_bitwise_xor = aclnn_mod.bitwise_xor

    def wrapped_bitwise_xor(*args, **kwargs):
        calls["count"] += 1
        return original_bitwise_xor(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "bitwise_xor", wrapped_bitwise_xor)

    _ = torch.bitwise_xor(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_fast_maximum_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """generic fast_binary_op path should bypass candle._backends.npu.aclnn.maximum."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.randn(4, 4, device=npu_device)
    b = torch.randn(4, 4, device=npu_device)
    torch.npu.synchronize()

    _ = torch.maximum(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_maximum = aclnn_mod.maximum

    def wrapped_maximum(*args, **kwargs):
        calls["count"] += 1
        return original_maximum(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "maximum", wrapped_maximum)

    _ = torch.maximum(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_fast_minimum_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """generic fast_binary_op path should bypass candle._backends.npu.aclnn.minimum."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.randn(4, 4, device=npu_device)
    b = torch.randn(4, 4, device=npu_device)
    torch.npu.synchronize()

    _ = torch.minimum(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_minimum = aclnn_mod.minimum

    def wrapped_minimum(*args, **kwargs):
        calls["count"] += 1
        return original_minimum(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "minimum", wrapped_minimum)

    _ = torch.minimum(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_fast_logical_xor_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """logical_xor should bypass candle._backends.npu.aclnn.logical_xor."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.tensor([[True, False], [True, True]], device=npu_device)
    b = torch.tensor([[True, True], [False, True]], device=npu_device)
    torch.npu.synchronize()

    _ = torch.logical_xor(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_logical_xor = aclnn_mod.logical_xor

    def wrapped_logical_xor(*args, **kwargs):
        calls["count"] += 1
        return original_logical_xor(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "logical_xor", wrapped_logical_xor)

    _ = torch.logical_xor(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_fast_pow_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """tensor-tensor pow should bypass candle._backends.npu.aclnn.pow_tensor_tensor."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.rand(4, 4, device=npu_device) + 0.5
    b = torch.rand(4, 4, device=npu_device) + 0.5
    torch.npu.synchronize()

    _ = torch.pow(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_pow = aclnn_mod.pow_tensor_tensor

    def wrapped_pow(*args, **kwargs):
        calls["count"] += 1
        return original_pow(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "pow_tensor_tensor", wrapped_pow)

    _ = torch.pow(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_fast_fmax_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """fmax should bypass candle._backends.npu.aclnn.maximum."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.randn(4, 4, device=npu_device)
    b = torch.randn(4, 4, device=npu_device)
    torch.npu.synchronize()

    _ = torch.fmax(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_maximum = aclnn_mod.maximum

    def wrapped_maximum(*args, **kwargs):
        calls["count"] += 1
        return original_maximum(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "maximum", wrapped_maximum)

    _ = torch.fmax(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_fast_floor_divide_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """floor_divide should bypass candle._backends.npu.aclnn.floor_divide."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.randn(4, 4, device=npu_device)
    b = torch.rand(4, 4, device=npu_device) + 1.0
    torch.npu.synchronize()

    _ = torch.floor_divide(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_floor_divide = aclnn_mod.floor_divide

    def wrapped_floor_divide(*args, **kwargs):
        calls["count"] += 1
        return original_floor_divide(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "floor_divide", wrapped_floor_divide)

    _ = torch.floor_divide(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_fast_lt_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """lt should bypass candle._backends.npu.aclnn.lt_tensor."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.randn(4, 4, device=npu_device)
    b = torch.randn(4, 4, device=npu_device)
    torch.npu.synchronize()

    _ = torch.lt(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_lt = aclnn_mod.lt_tensor

    def wrapped_lt(*args, **kwargs):
        calls["count"] += 1
        return original_lt(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "lt_tensor", wrapped_lt)

    _ = torch.lt(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_fast_matmul_exact_uses_pta_executor_cache():
    """fast_matmul_exact should reuse torch_npu-style PTA executors before GetWorkspaceSize."""
    src = (_REPO_ROOT / "src/candle/_C/_npu_ops.pyx").read_text(encoding="utf-8")
    match = re.search(
        r"cpdef object fast_matmul_exact\(TensorImpl a, TensorImpl b\):(?P<body>.*?)\n\ncpdef object fast_matmul\(",
        src,
        flags=re.DOTALL,
    )
    assert match is not None
    body = match.group("body")
    assert "_ffi_pta_begin_binary_cache_lookup_raw" in body
    assert 'b"aclnnMatmul"' in body
    assert body.index("_ffi_pta_begin_binary_cache_lookup_raw") < body.index(
        "_ffi_binary_two_inputs_with_int8_op"
    )


def test_fast_mm_backward_helpers_use_pta_executor_cache():
    """Matmul backward helpers should not rebuild ACLNN Matmul executors on every step."""
    src = (_REPO_ROOT / "src/candle/_C/_npu_ops.pyx").read_text(encoding="utf-8")
    for name, next_name in (
        ("fast_mm_mat1_backward", "fast_mm_mat2_backward"),
        ("fast_mm_mat2_backward", "fast_addmm"),
    ):
        match = re.search(
            rf"cpdef object {name}\(.*?\):(?P<body>.*?)\n\ncpdef object {next_name}\(",
            src,
            flags=re.DOTALL,
        )
        assert match is not None
        body = match.group("body")
        assert "_ffi_pta_begin_binary_cache_lookup_raw" in body, name
        assert 'b"aclnnMatmul"' in body, name
        assert body.index("_ffi_pta_begin_binary_cache_lookup_raw") < body.index(
            "_ffi_binary_two_inputs_with_int8_op"
        ), name


def test_fast_gelu_backward_uses_pta_executor_cache():
    """GELU backward should reuse PTA executors before rebuilding workspace."""
    src = (_REPO_ROOT / "src/candle/_C/_npu_ops.pyx").read_text(encoding="utf-8")
    match = re.search(
        r"cpdef object fast_gelu_backward\(.*?\):(?P<body>.*?)\n\n\ncpdef object fast_silu_backward",
        src,
        flags=re.DOTALL,
    )
    assert match is not None
    body = match.group("body")
    assert "_ffi_pta_begin_binary_cache_lookup_raw" in body
    assert 'b"aclnnGeluBackward"' in body
    assert body.index("_ffi_pta_begin_binary_cache_lookup_raw") < body.index(
        "_ffi_binary_op_no_alpha"
    )



def test_fast_gelu_backward_uses_exact_tensorimpl_fast_wrapper():
    """GELU backward base tensors should use direct fields and the fast NPU wrapper."""
    src = (_REPO_ROOT / "src/candle/_C/_npu_ops.pyx").read_text(encoding="utf-8")
    match = re.search(
        r"cpdef object fast_gelu_backward\(.*?\):(?P<body>.*?)\n\n\ncpdef object fast_silu_backward",
        src,
        flags=re.DOTALL,
    )
    assert match is not None
    body = match.group("body")
    for snippet in (
        "cdef TensorImpl grad_t",
        "cdef TensorImpl input_t",
        "grad_t._stride_tuple",
        "input_t._stride_tuple",
        "_tensor_dtype_to_acl_code(grad_t)",
        "_get_stream_obj_fast(dev_idx)",
        "_get_stream_raw_fast(dev_idx)",
        "malloc_large_cached",
        "_make_npu_tensor_fast_large",
    ):
        assert snippet in body
    assert body.index("grad_t._stride_tuple") < body.index("_ffi_pta_begin_binary_cache_lookup_raw")



def test_fast_sum_uses_pta_executor_cache():
    """fast_sum should reuse PTA executors before rebuilding ReduceSum workspace."""
    src = (_REPO_ROOT / "src/candle/_C/_npu_ops.pyx").read_text(encoding="utf-8")
    match = re.search(
        r"cpdef object fast_sum\(.*?\):(?P<body>.*?)\n\ncpdef object fast_mm_mat1_backward",
        src,
        flags=re.DOTALL,
    )
    assert match is not None
    body = match.group("body")
    assert "_ffi_pta_begin_reduce_sum_cache_lookup_raw" in body
    assert body.index("_ffi_pta_begin_reduce_sum_cache_lookup_raw") < body.index(
        "_ffi_reduce_sum_op"
    )



def test_reduce_sum_pta_hash_includes_dims_and_keepdim():
    """ReduceSum PTA keys must include reduction dims and keepdim semantics."""
    src = (_REPO_ROOT / "src/candle/_C/_aclnn_ffi.pyx").read_text(encoding="utf-8")
    match = re.search(
        r"cdef int pta_begin_reduce_sum_cache_lookup_raw\(.*?\) except -1:(?P<body>.*?)\n\n\ncpdef object reduce_sum_op",
        src,
        flags=re.DOTALL,
    )
    assert match is not None
    body = match.group("body")
    assert 'b"aclnnReduceSum"' in body
    assert "dims_buf" in body
    assert "dims_ndim" in body
    assert "&keepdim" in body
    assert body.index("_pta_buf_append_cstr(hash_buf, &hash_offset, b\"aclnnReduceSum\")") < body.index(
        "_pta_buf_append(hash_buf, &hash_offset, dims_buf"
    )
    assert body.index("_pta_buf_append(hash_buf, &hash_offset, &keepdim") < body.index(
        "_pta_buf_append_tensor(hash_buf, &hash_offset, r_shape"
    )



def test_fast_addmm_uses_pta_executor_cache():
    """fast_addmm should reuse PTA executors before rebuilding Addmm workspace."""
    src = (_REPO_ROOT / "src/candle/_C/_npu_ops.pyx").read_text(encoding="utf-8")
    match = re.search(
        r"cpdef object fast_addmm\(.*?\):(?P<body>.*?)\n\n# ---------------------------------------------------------------------------\n# fast_sub",
        src,
        flags=re.DOTALL,
    )
    assert match is not None
    body = match.group("body")
    assert "_ffi_pta_begin_addmm_cache_lookup_raw" in body
    assert body.index("_ffi_pta_begin_addmm_cache_lookup_raw") < body.index(
        "_ffi_four_tensor_two_scalars_one_int8_op"
    )


def test_addmm_ffi_helper_defers_descriptor_cleanup_until_executor_destroyed():
    """Addmm descriptors must outlive GetWorkspaceSize when workspace execute is deferred."""
    src = (_REPO_ROOT / "src/candle/_C/_aclnn_ffi.pyx").read_text(encoding="utf-8")
    match = re.search(
        r"cpdef object four_tensor_two_scalars_one_int8_op\(.*?\):(?P<body>.*?)\n\n\ndef tensor_int_array_two_outputs_op",
        src,
        flags=re.DOTALL,
    )
    assert match is not None
    body = match.group("body")
    assert "_register_executor_cleanup" in body
    assert body.index("_register_executor_cleanup") < body.index("if ws_size == 0:")
    for name in ("a_t", "b_t", "c_t", "out_t"):
        assert f"('t', <uintptr_t>{name})" in body
        assert f"{name} = NULL" in body


def test_addmm_pta_hash_discriminates_operand_aliases():
    """Addmm PTA keys must separate aliased and non-aliased operand patterns."""
    src = (_REPO_ROOT / "src/candle/_C/_aclnn_ffi.pyx").read_text(encoding="utf-8")
    match = re.search(
        r"cdef int pta_begin_addmm_cache_lookup_raw\(.*?\) except -1:(?P<body>.*?)\n\n\ncpdef object pta_begin_unary_cache_lookup",
        src,
        flags=re.DOTALL,
    )
    assert match is not None
    body = match.group("body")
    for name in ("input_mat1_alias", "input_mat2_alias", "mat1_mat2_alias"):
        assert name in body
        assert f"&{name}" in body
    assert body.index("_pta_buf_append_cstr(hash_buf, &hash_offset, b\"aclnnAddmm\")") < body.index(
        "_pta_buf_append(hash_buf, &hash_offset, &input_mat1_alias"
    )
    assert body.index("_pta_buf_append(hash_buf, &hash_offset, &mat1_mat2_alias") < body.index(
        "_pta_buf_append_tensor(hash_buf, &hash_offset, input_shape_buf"
    )


def test_fast_matmul_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """2D NPU matmul should bypass candle._backends.npu.aclnn.matmul when safe."""
    import numpy as np
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.randn(8, 16, device=npu_device, dtype=torch.float16)
    b = torch.randn(16, 12, device=npu_device, dtype=torch.float16)
    torch.npu.synchronize()

    expected = torch.matmul(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_matmul = aclnn_mod.matmul

    def wrapped_matmul(*args, **kwargs):
        calls["count"] += 1
        return original_matmul(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "matmul", wrapped_matmul)

    out = torch.matmul(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0, (
        f"fast_matmul called aclnn.matmul {calls['count']} time(s); expected 0"
    )
    assert np.allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-2, atol=1e-2)


def test_fast_addmm_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """2D NPU addmm should bypass candle._backends.npu.aclnn.addmm when safe."""
    import numpy as np
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    bias = torch.randn(12, device=npu_device, dtype=torch.float16)
    a = torch.randn(8, 16, device=npu_device, dtype=torch.float16)
    b = torch.randn(16, 12, device=npu_device, dtype=torch.float16)
    torch.npu.synchronize()

    expected = torch.addmm(bias, a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_addmm = aclnn_mod.addmm

    def wrapped_addmm(*args, **kwargs):
        calls["count"] += 1
        return original_addmm(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "addmm", wrapped_addmm)

    out = torch.addmm(bias, a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0, (
        f"fast_addmm called aclnn.addmm {calls['count']} time(s); expected 0"
    )
    assert np.allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-2, atol=1e-2)


def test_fast_addmm_default_scalars_are_cached_per_dtype(npu_device, monkeypatch):
    """Default beta=alpha=1 addmm should reuse cached aclScalar handles."""
    import candle as torch
    from candle._C import _aclnn_ffi

    bias = torch.randn(12, device=npu_device, dtype=torch.float16)
    a = torch.randn(8, 16, device=npu_device, dtype=torch.float16)
    b = torch.randn(16, 12, device=npu_device, dtype=torch.float16)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_create_scalar = _aclnn_ffi.create_scalar

    def wrapped_create_scalar(*args, **kwargs):
        calls["count"] += 1
        return original_create_scalar(*args, **kwargs)

    monkeypatch.setattr(_aclnn_ffi, "create_scalar", wrapped_create_scalar)

    _ = torch.addmm(bias, a, b)
    torch.npu.synchronize()
    _ = torch.addmm(bias, a, b)
    torch.npu.synchronize()

    assert calls["count"] <= 1



def test_fast_addmm_pta_alias_and_nonalias_reuse_remain_correct(npu_device):
    """Addmm PTA cache entries must not cross aliased/non-aliased mat operands."""
    import numpy as np
    import candle as torch

    bias = torch.randn(8, device=npu_device, dtype=torch.float16)
    a = torch.randn(8, 8, device=npu_device, dtype=torch.float16)
    b = torch.randn(8, 8, device=npu_device, dtype=torch.float16)
    torch.npu.synchronize()

    # Warm both alias patterns with the same shapes/strides/dtypes.  A stale
    # cached executor that does not distinguish mat1==mat2 can bind the wrong
    # input list for one of these calls.
    _ = [torch.addmm(bias, a, b) for _ in range(5)]
    torch.npu.synchronize()
    aliased = torch.addmm(bias, a, a)
    torch.npu.synchronize()
    _ = [torch.addmm(bias, a, a) for _ in range(5)]
    torch.npu.synchronize()
    nonaliased = torch.addmm(bias, a, b)
    torch.npu.synchronize()

    bias_np = bias.cpu().float().numpy()
    a_np = a.cpu().float().numpy()
    b_np = b.cpu().float().numpy()
    assert np.allclose(aliased.cpu().float().numpy(), bias_np + a_np @ a_np, rtol=2e-2, atol=2e-2)
    assert np.allclose(nonaliased.cpu().float().numpy(), bias_np + a_np @ b_np, rtol=2e-2, atol=2e-2)



def test_linear_bias_uses_fused_addmm_kernel_without_python_wrapper(npu_device, monkeypatch):
    """NPU linear with bias should use the fused addmm kernel, bypassing the Python wrapper."""
    import numpy as np
    import candle as torch
    import candle._functional as functional_mod
    import candle.nn.functional as F

    x = torch.randn(8, 16, device=npu_device, dtype=torch.float16)
    weight = torch.randn(12, 16, device=npu_device, dtype=torch.float16)
    bias = torch.randn(12, device=npu_device, dtype=torch.float16)
    torch.npu.synchronize()

    expected = F.linear(x, weight, bias)
    torch.npu.synchronize()

    calls = {"addmm": 0}
    original_addmm = functional_mod.addmm

    def wrapped_addmm(*args, **kwargs):
        calls["addmm"] += 1
        return original_addmm(*args, **kwargs)

    monkeypatch.setattr(functional_mod, "addmm", wrapped_addmm)

    out = F.linear(x, weight, bias)
    torch.npu.synchronize()

    assert calls["addmm"] == 0
    assert np.allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-2, atol=1e-2)



def test_linear_bias_avoids_matmul_add_decomposition(npu_device, monkeypatch):
    """NPU linear with bias should use generic addmm, not matmul plus add."""
    import numpy as np
    import candle as torch
    import candle._dispatch as dispatch_mod
    import candle.nn.functional as F

    x = torch.randn(8, 16, device=npu_device, dtype=torch.float16)
    weight = torch.randn(12, 16, device=npu_device, dtype=torch.float16)
    bias = torch.randn(12, device=npu_device, dtype=torch.float16)
    torch.npu.synchronize()

    expected = F.linear(x, weight, bias)
    torch.npu.synchronize()

    ops = []
    original_dispatch = dispatch_mod.dispatch

    def wrapped_dispatch(op_name, *args, **kwargs):
        ops.append(op_name)
        return original_dispatch(op_name, *args, **kwargs)

    monkeypatch.setattr(dispatch_mod, "dispatch", wrapped_dispatch)

    out = F.linear(x, weight, bias)
    torch.npu.synchronize()

    assert "matmul" not in ops
    assert "add" not in ops
    assert np.allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-2, atol=1e-2)


def test_linear_bias_addmm_backward_runs_on_npu(npu_device):
    """Linear routed through addmm must preserve NPU autograd for input/weight/bias."""
    import candle as torch
    import candle.nn.functional as F

    x = torch.randn(4, 8, device=npu_device, dtype=torch.float16, requires_grad=True)
    weight = torch.randn(6, 8, device=npu_device, dtype=torch.float16, requires_grad=True)
    bias = torch.randn(6, device=npu_device, dtype=torch.float16, requires_grad=True)

    loss = F.linear(x, weight, bias).sum()
    loss.backward()
    torch.npu.synchronize()

    assert x.grad is not None
    assert weight.grad is not None
    assert bias.grad is not None
    assert x.grad.device.type == "npu"
    assert weight.grad.device.type == "npu"
    assert bias.grad.device.type == "npu"
    assert x.grad.shape == x.shape
    assert weight.grad.shape == weight.shape
    assert bias.grad.shape == bias.shape


def test_npu_parameter_transpose_backward_skips_generated_python_node(npu_device, monkeypatch):
    """NPU Parameter.t should attach a Cython transpose backward node."""
    import candle as torch
    import candle.nn as nn
    import candle._generated.functions as functions_mod

    layer = nn.Linear(8, 6).to(npu_device).to(torch.float16)
    torch.npu.synchronize()

    def fail_generated_node(*args, **kwargs):
        raise AssertionError("NPU transpose should attach a Cython autograd node")

    monkeypatch.setattr(functions_mod, "TransposeIntBackward0", fail_generated_node)

    out = layer.weight.t()
    out.sum().backward()
    torch.npu.synchronize()

    assert layer.weight.grad is not None
    assert layer.weight.grad.device.type == "npu"
    assert layer.weight.grad.shape == layer.weight.shape



def test_npu_parameter_transpose_skips_python_dispatch(npu_device):
    """NPU Parameter.t should use the same safe view/autograd path as base tensors."""
    import os
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        f"""
        import candle as torch
        import candle.nn as nn
        import candle._dispatch as dispatch_pkg

        layer = nn.Linear(8, 6).to({str(npu_device)!r}).to(torch.float16)
        torch.npu.synchronize()

        calls = []
        original_dispatch = dispatch_pkg.dispatch

        def wrapped_dispatch(op_name, *args, **kwargs):
            if op_name == "transpose":
                calls.append(op_name)
                raise AssertionError("Parameter transpose should bypass Python dispatch")
            return original_dispatch(op_name, *args, **kwargs)

        dispatch_pkg.dispatch = wrapped_dispatch
        out = layer.weight.t()
        out.sum().backward()
        torch.npu.synchronize()
        assert calls == []
        assert out.device.type == "npu"
        assert layer.weight.grad is not None and layer.weight.grad.device.type == "npu"
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stdout + result.stderr



def test_npu_linear_with_parameter_bias_skips_addmm_dispatch(npu_device):
    """nn.Linear Parameter tensors should use the safe NPU addmm hot path."""
    import os
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        f"""
        import candle as torch
        import candle.nn as nn
        import candle._dispatch.dispatcher as dispatcher_mod

        layer = nn.Linear(8, 6).to({str(npu_device)!r}).to(torch.float16)
        x = torch.randn(4, 8, device={str(npu_device)!r}, dtype=torch.float16, requires_grad=True)
        torch.npu.synchronize()

        calls = []
        original_dispatch = dispatcher_mod.dispatch

        def wrapped_dispatch(op_name, *args, **kwargs):
            if op_name == "addmm":
                calls.append(op_name)
                raise AssertionError("Parameter linear should bypass addmm Python dispatch")
            return original_dispatch(op_name, *args, **kwargs)

        dispatcher_mod.dispatch = wrapped_dispatch
        out = layer(x)
        out.sum().backward()
        torch.npu.synchronize()
        assert calls == []
        assert out.device.type == "npu"
        assert x.grad is not None and x.grad.device.type == "npu"
        assert layer.weight.grad is not None and layer.weight.grad.device.type == "npu"
        assert layer.bias.grad is not None and layer.bias.grad.device.type == "npu"
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stdout + result.stderr



def test_npu_linear_cython_hot_path_skips_python_functional_addmm(npu_device, monkeypatch):
    """Safe NPU F.linear should bypass the Python nn.functional/addmm glue."""
    import candle as torch
    import candle.nn.functional as F
    import candle._functional as functional_mod

    x = torch.randn(4, 8, device=npu_device, dtype=torch.float16, requires_grad=True)
    weight = torch.randn(6, 8, device=npu_device, dtype=torch.float16, requires_grad=True)
    bias = torch.randn(6, device=npu_device, dtype=torch.float16, requires_grad=True)
    torch.npu.synchronize()

    def fail_python_addmm(*args, **kwargs):
        raise AssertionError("NPU F.linear hot path should bypass Python _functional.addmm glue")

    monkeypatch.setattr(functional_mod, "addmm", fail_python_addmm)

    out = F.linear(x, weight, bias)
    out.sum().backward()
    torch.npu.synchronize()

    assert out.device.type == "npu"
    assert x.grad is not None and x.grad.device.type == "npu"
    assert weight.grad is not None and weight.grad.device.type == "npu"
    assert bias.grad is not None and bias.grad.device.type == "npu"



def test_npu_linear_cython_hot_path_skips_python_weight_t(npu_device, monkeypatch):
    """Safe NPU F.linear should avoid the Python weight.t() wrapper."""
    import candle as torch
    import candle.nn.functional as F

    x = torch.randn(4, 8, device=npu_device, dtype=torch.float16, requires_grad=True)
    weight = torch.randn(6, 8, device=npu_device, dtype=torch.float16, requires_grad=True)
    bias = torch.randn(6, device=npu_device, dtype=torch.float16, requires_grad=True)
    torch.npu.synchronize()

    tensor_type = type(weight)

    def fail_python_t(self, *args, **kwargs):
        raise AssertionError("NPU F.linear hot path should bypass Python weight.t() glue")

    monkeypatch.setattr(tensor_type, "t", fail_python_t)

    out = F.linear(x, weight, bias)
    out.sum().backward()
    torch.npu.synchronize()

    assert out.device.type == "npu"
    assert x.grad is not None and x.grad.device.type == "npu"
    assert weight.grad is not None and weight.grad.device.type == "npu"
    assert bias.grad is not None and bias.grad.device.type == "npu"



def test_npu_addmm_autograd_skips_python_dispatch(npu_device, monkeypatch):
    """NPU addmm training hot path should attach autograd without Python dispatch."""
    import candle as torch
    import candle._functional as functional_mod

    bias = torch.randn(6, device=npu_device, dtype=torch.float16, requires_grad=True)
    a = torch.randn(4, 8, device=npu_device, dtype=torch.float16, requires_grad=True)
    b = torch.randn(8, 6, device=npu_device, dtype=torch.float16, requires_grad=True)
    torch.npu.synchronize()

    original_dispatch = functional_mod.dispatch

    def fail_dispatch(op_name, *args, **kwargs):
        if op_name == "addmm":
            raise AssertionError("NPU addmm autograd hot path should bypass Python dispatch")
        return original_dispatch(op_name, *args, **kwargs)

    monkeypatch.setattr(functional_mod, "dispatch", fail_dispatch)

    out = torch.addmm(bias, a, b)
    out.sum().backward()
    torch.npu.synchronize()

    assert out.device.type == "npu"
    assert a.grad is not None and a.grad.device.type == "npu"
    assert b.grad is not None and b.grad.device.type == "npu"
    assert bias.grad is not None and bias.grad.device.type == "npu"



def test_npu_addmm_autograd_skips_generated_python_node(npu_device, monkeypatch):
    """Default NPU addmm autograd should use the Cython node, not generated Python AddmmBackward0."""
    import candle as torch
    import candle._generated.functions as functions_mod

    bias = torch.randn(6, device=npu_device, dtype=torch.float16, requires_grad=True)
    a = torch.randn(4, 8, device=npu_device, dtype=torch.float16, requires_grad=True)
    b = torch.randn(8, 6, device=npu_device, dtype=torch.float16, requires_grad=True)
    torch.npu.synchronize()

    def fail_generated_node(*args, **kwargs):
        raise AssertionError("NPU addmm should attach a Cython autograd node")

    monkeypatch.setattr(functions_mod, "AddmmBackward0", fail_generated_node)

    out = torch.addmm(bias, a, b)
    out.sum().backward()
    torch.npu.synchronize()

    assert a.grad is not None and a.grad.device.type == "npu"
    assert b.grad is not None and b.grad.device.type == "npu"
    assert bias.grad is not None and bias.grad.device.type == "npu"



def test_addmm_backward_uses_direct_npu_mm_helpers_without_redispatch(npu_device, monkeypatch):
    """Default NPU addmm backward should call direct Cython mm helpers, not redispatch them."""
    import candle as torch
    import candle._generated.functions as functions_mod

    bias = torch.randn(6, device=npu_device, dtype=torch.float16, requires_grad=True)
    a = torch.randn(4, 8, device=npu_device, dtype=torch.float16, requires_grad=True)
    b = torch.randn(8, 6, device=npu_device, dtype=torch.float16, requires_grad=True)
    torch.npu.synchronize()

    calls = {"mm_mat1_backward": 0, "mm_mat2_backward": 0}
    original_redispatch = functions_mod.redispatch

    def wrapped_redispatch(op_name, *args, **kwargs):
        if op_name in calls:
            calls[op_name] += 1
        return original_redispatch(op_name, *args, **kwargs)

    monkeypatch.setattr(functions_mod, "redispatch", wrapped_redispatch)

    torch.addmm(bias, a, b).sum().backward()
    torch.npu.synchronize()

    assert calls == {"mm_mat1_backward": 0, "mm_mat2_backward": 0}



def test_addmm_backward_uses_fused_mm_helpers_without_transpose_dispatch(npu_device, monkeypatch):
    """Default addmm backward should avoid Python transpose/matmul helper composition."""
    import candle as torch
    import candle._dispatch as dispatch_pkg

    bias = torch.randn(6, device=npu_device, dtype=torch.float16, requires_grad=True)
    a = torch.randn(4, 8, device=npu_device, dtype=torch.float16, requires_grad=True)
    b = torch.randn(8, 6, device=npu_device, dtype=torch.float16, requires_grad=True)
    torch.npu.synchronize()

    calls = {"transpose": 0, "matmul": 0}
    original_dispatch = dispatch_pkg.dispatch

    def wrapped_dispatch(op_name, *args, **kwargs):
        if op_name in calls:
            calls[op_name] += 1
        return original_dispatch(op_name, *args, **kwargs)

    monkeypatch.setattr(dispatch_pkg, "dispatch", wrapped_dispatch)

    torch.addmm(bias, a, b).sum().backward()
    torch.npu.synchronize()

    assert calls == {"transpose": 0, "matmul": 0}


def test_npu_tensor_t_autograd_skips_python_dispatch(npu_device, monkeypatch):
    """NPU tensor.t() should attach transpose autograd without Python dispatch."""
    import candle as torch
    import candle._dispatch as dispatch_mod

    weight = torch.randn(6, 8, device=npu_device, dtype=torch.float16, requires_grad=True)
    torch.npu.synchronize()

    def fail_dispatch(op_name, *args, **kwargs):
        if op_name == "transpose":
            raise AssertionError("NPU tensor.t() hot path should bypass Python transpose dispatch")
        return original_dispatch(op_name, *args, **kwargs)

    original_dispatch = dispatch_mod.dispatch
    monkeypatch.setattr(dispatch_mod, "dispatch", fail_dispatch)

    view = weight.t()
    view.sum().backward()
    torch.npu.synchronize()

    assert view.device.type == "npu"
    assert view.shape == (8, 6)
    assert weight.grad is not None and weight.grad.device.type == "npu"



def test_linear_backward_skips_transpose_redispatch_for_weight_view(npu_device, monkeypatch):
    """Linear weight.T view backward should avoid Python transpose redispatch."""
    import candle as torch
    import candle.nn.functional as F
    import candle._generated.functions as functions_mod

    x = torch.randn(4, 8, device=npu_device, dtype=torch.float16, requires_grad=True)
    weight = torch.randn(6, 8, device=npu_device, dtype=torch.float16, requires_grad=True)
    bias = torch.randn(6, device=npu_device, dtype=torch.float16, requires_grad=True)
    loss = F.linear(x, weight, bias).sum()
    torch.npu.synchronize()

    calls = {"transpose": 0}
    original_redispatch = functions_mod.redispatch

    def wrapped_redispatch(op_name, *args, **kwargs):
        if op_name == "transpose":
            calls["transpose"] += 1
        return original_redispatch(op_name, *args, **kwargs)

    monkeypatch.setattr(functions_mod, "redispatch", wrapped_redispatch)

    loss.backward()
    torch.npu.synchronize()

    assert calls["transpose"] == 0


def test_npu_backward_implicit_grad_seed_stays_on_device(npu_device, monkeypatch):
    """Implicit scalar backward grad should be created on NPU, not copied from CPU."""
    import candle as torch
    import candle._functional as functional_mod

    x = torch.randn(8, 8, device=npu_device, dtype=torch.float16, requires_grad=True)
    loss = x.sum()
    torch.npu.synchronize()

    calls = {"cpu_to_npu": 0}
    original_to = functional_mod.to
    original_dispatch = functional_mod.dispatch

    def is_cpu_to_npu(a, device):
        target_type = getattr(device, "type", None)
        if target_type is None and isinstance(device, str):
            target_type = device.split(":", 1)[0]
        return a.device.type == "cpu" and target_type == "npu"

    def wrapped_to(a, device=None, *args, **kwargs):
        if is_cpu_to_npu(a, device):
            calls["cpu_to_npu"] += 1
        return original_to(a, device, *args, **kwargs)

    def wrapped_dispatch(op_name, *dispatch_args, **kwargs):
        if op_name == "to" and len(dispatch_args) >= 3 and is_cpu_to_npu(dispatch_args[1], dispatch_args[2]):
            calls["cpu_to_npu"] += 1
        return original_dispatch(op_name, *dispatch_args, **kwargs)

    monkeypatch.setattr(functional_mod, "to", wrapped_to)
    monkeypatch.setattr(functional_mod, "dispatch", wrapped_dispatch)

    loss.backward()
    torch.npu.synchronize()

    assert calls["cpu_to_npu"] == 0




def test_npu_backward_implicit_grad_seed_uses_cached_device_scalar(npu_device, monkeypatch):
    """Implicit scalar backward grad should not launch a fresh NPU ones kernel per call."""
    import candle as torch
    import candle._functional as functional_mod

    # Warm the scalar seed cache and op caches before asserting the hot path.
    warm = torch.randn(8, 8, device=npu_device, dtype=torch.float16, requires_grad=True)
    warm.sum().backward()
    torch.npu.synchronize()

    x = torch.randn(8, 8, device=npu_device, dtype=torch.float16, requires_grad=True)
    loss = x.sum()
    torch.npu.synchronize()

    calls = {"ones": 0}
    original_dispatch = functional_mod.dispatch

    def wrapped_dispatch(op_name, *dispatch_args, **kwargs):
        if op_name == "ones":
            calls["ones"] += 1
        return original_dispatch(op_name, *dispatch_args, **kwargs)

    monkeypatch.setattr(functional_mod, "dispatch", wrapped_dispatch)

    loss.backward()
    torch.npu.synchronize()

    assert calls["ones"] == 0


def test_linear_backward_fresh_npu_grads_skip_leaf_clone(npu_device, monkeypatch):
    """Fresh NPU linear backward grads should be stolen into .grad without clone copies."""
    import candle as torch
    import candle.nn.functional as F

    x = torch.randn(4, 8, device=npu_device, dtype=torch.float16, requires_grad=True)
    weight = torch.randn(6, 8, device=npu_device, dtype=torch.float16, requires_grad=True)
    bias = torch.randn(6, device=npu_device, dtype=torch.float16, requires_grad=True)
    loss = F.linear(x, weight, bias).sum()
    torch.npu.synchronize()

    clones = []
    tensor_type = type(x)
    original_clone = tensor_type.clone

    def wrapped_clone(self, *args, **kwargs):
        clones.append(tuple(self.shape))
        return original_clone(self, *args, **kwargs)

    monkeypatch.setattr(tensor_type, "clone", wrapped_clone)

    loss.backward()
    torch.npu.synchronize()

    assert clones == []


def test_npu_linear_addmm_backward_skips_internal_leaf_accumulate_grad_nodes(npu_device):
    """NPU hot backward should not create public AccumulateGrad nodes unless inspected."""
    import candle as torch
    import candle.nn.functional as F

    def assert_no_accumulate_grad_nodes(*tensors):
        for tensor in tensors:
            assert getattr(tensor, "_accumulate_grad_node", None) is None

    x = torch.randn(4, 8, device=npu_device, dtype=torch.float16, requires_grad=True)
    weight = torch.randn(6, 8, device=npu_device, dtype=torch.float16, requires_grad=True)
    bias = torch.randn(6, device=npu_device, dtype=torch.float16, requires_grad=True)
    linear_out = F.linear(x, weight, bias)
    assert_no_accumulate_grad_nodes(x, weight, bias)

    linear_out.sum().backward()
    torch.npu.synchronize()

    assert x.grad is not None
    assert weight.grad is not None
    assert bias.grad is not None
    assert_no_accumulate_grad_nodes(x, weight, bias)
    assert linear_out.grad_fn.next_functions[0][0] is not None
    assert linear_out.grad_fn.next_functions[1][0] is not None
    assert linear_out.grad_fn.next_functions[2][0] is not None

    addmm_bias = torch.randn(6, device=npu_device, dtype=torch.float16, requires_grad=True)
    mat1 = torch.randn(4, 8, device=npu_device, dtype=torch.float16, requires_grad=True)
    mat2 = torch.randn(8, 6, device=npu_device, dtype=torch.float16, requires_grad=True)
    addmm_out = torch.addmm(addmm_bias, mat1, mat2)
    assert_no_accumulate_grad_nodes(addmm_bias, mat1, mat2)

    addmm_out.sum().backward()
    torch.npu.synchronize()

    assert addmm_bias.grad is not None
    assert mat1.grad is not None
    assert mat2.grad is not None
    assert_no_accumulate_grad_nodes(addmm_bias, mat1, mat2)
    assert addmm_out.grad_fn.next_functions[0][0] is not None
    assert addmm_out.grad_fn.next_functions[1][0] is not None
    assert addmm_out.grad_fn.next_functions[2][0] is not None


def test_addmm_bias_grad_sum_uses_direct_npu_sum_without_redispatch(npu_device, monkeypatch):
    """NPU addmm bias gradients should reduce through direct Cython sum, not redispatch sum."""
    import candle as torch
    import candle._generated.functions as functions_mod

    bias = torch.randn(6, device=npu_device, dtype=torch.float16, requires_grad=True)
    a = torch.randn(4, 8, device=npu_device, dtype=torch.float16, requires_grad=True)
    b = torch.randn(8, 6, device=npu_device, dtype=torch.float16, requires_grad=True)
    torch.npu.synchronize()

    calls = {"sum": 0}
    original_redispatch = functions_mod.redispatch

    def wrapped_redispatch(op_name, *args, **kwargs):
        if op_name == "sum":
            calls["sum"] += 1
        return original_redispatch(op_name, *args, **kwargs)

    monkeypatch.setattr(functions_mod, "redispatch", wrapped_redispatch)

    torch.addmm(bias, a, b).sum().backward()
    torch.npu.synchronize()

    assert calls["sum"] == 0



def test_full_sum_backward_skips_contiguous_materialization(npu_device, monkeypatch):
    """Full sum backward should pass the expanded grad view through without materializing it."""
    import candle as torch
    import candle.nn.functional as F
    import candle._generated.functions as functions_mod

    x = torch.randn(4, 8, device=npu_device, dtype=torch.float16, requires_grad=True)
    weight = torch.randn(6, 8, device=npu_device, dtype=torch.float16, requires_grad=True)
    bias = torch.randn(6, device=npu_device, dtype=torch.float16, requires_grad=True)
    loss = F.linear(x, weight, bias).sum()
    torch.npu.synchronize()

    calls = {"contiguous": 0}
    original_redispatch = functions_mod.redispatch

    def wrapped_redispatch(op_name, *args, **kwargs):
        if op_name == "contiguous":
            calls["contiguous"] += 1
        return original_redispatch(op_name, *args, **kwargs)

    monkeypatch.setattr(functions_mod, "redispatch", wrapped_redispatch)

    loss.backward()
    torch.npu.synchronize()

    assert calls["contiguous"] == 0



def test_addmm_backward_skips_unused_metadata_redispatch(npu_device, monkeypatch):
    """addmm backward helpers do not need sym_strides/layout metadata dispatches."""
    import candle as torch
    import candle._generated.functions as functions_mod

    bias = torch.randn(6, device=npu_device, dtype=torch.float16, requires_grad=True)
    a = torch.randn(4, 8, device=npu_device, dtype=torch.float16, requires_grad=True)
    b = torch.randn(8, 6, device=npu_device, dtype=torch.float16, requires_grad=True)
    torch.npu.synchronize()

    calls = {"sym_strides": 0, "layout": 0}
    original_redispatch = functions_mod.redispatch

    def wrapped_redispatch(op_name, *args, **kwargs):
        if op_name in calls:
            calls[op_name] += 1
        return original_redispatch(op_name, *args, **kwargs)

    monkeypatch.setattr(functions_mod, "redispatch", wrapped_redispatch)

    torch.addmm(bias, a, b).sum().backward()
    torch.npu.synchronize()

    assert calls == {"sym_strides": 0, "layout": 0}



def test_addmm_backward_default_scalars_skip_mul_redispatch(npu_device, monkeypatch):
    """Default alpha=beta=1 addmm backward should not launch no-op mul kernels."""
    import candle as torch
    import candle._generated.functions as functions_mod

    bias = torch.randn(6, device=npu_device, dtype=torch.float16, requires_grad=True)
    a = torch.randn(4, 8, device=npu_device, dtype=torch.float16, requires_grad=True)
    b = torch.randn(8, 6, device=npu_device, dtype=torch.float16, requires_grad=True)
    torch.npu.synchronize()

    calls = {"mul": 0}
    original_redispatch = functions_mod.redispatch

    def wrapped_redispatch(op_name, *args, **kwargs):
        if op_name == "mul":
            calls["mul"] += 1
        return original_redispatch(op_name, *args, **kwargs)

    monkeypatch.setattr(functions_mod, "redispatch", wrapped_redispatch)

    torch.addmm(bias, a, b).sum().backward()
    torch.npu.synchronize()

    assert calls["mul"] == 0


def test_sum_hot_paths_skip_python_aclnn_wrapper(npu_device, monkeypatch):
    """NPU scalar sum and dim=0 sum should bypass candle._backends.npu.aclnn.reduce_sum."""
    import numpy as np
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    x = torch.randn(8, 16, device=npu_device, dtype=torch.float16)
    torch.npu.synchronize()
    expected_all = x.sum()
    expected_dim0 = x.sum(dim=0)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_reduce_sum = aclnn_mod.reduce_sum

    def wrapped_reduce_sum(*args, **kwargs):
        calls["count"] += 1
        return original_reduce_sum(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "reduce_sum", wrapped_reduce_sum)

    out_all = x.sum()
    out_dim0 = x.sum(dim=0)
    torch.npu.synchronize()

    assert calls["count"] == 0
    assert np.allclose(out_all.cpu().numpy(), expected_all.cpu().numpy(), rtol=1e-2, atol=1e-2)
    assert np.allclose(out_dim0.cpu().numpy(), expected_dim0.cpu().numpy(), rtol=1e-2, atol=1e-2)


def test_npu_tensor_sum_backward_skips_generated_python_node(npu_device, monkeypatch):
    """Default full NPU Tensor.sum backward should use the Cython node, not Python SumBackward0."""
    import candle as torch
    import candle._generated.functions as functions_mod

    x = torch.randn(8, 16, device=npu_device, dtype=torch.float16, requires_grad=True)
    torch.npu.synchronize()

    def fail_generated_node(*args, **kwargs):
        raise AssertionError("NPU Tensor.sum should attach a Cython autograd node")

    monkeypatch.setattr(functions_mod, "SumBackward0", fail_generated_node)

    out = x.sum()
    out.backward()
    torch.npu.synchronize()

    assert x.grad is not None
    assert x.grad.device.type == "npu"
    assert x.grad.shape == x.shape



def test_npu_tensor_sum_method_skips_python_dispatch(npu_device):
    """Safe base NPU Tensor.sum should bypass Python dispatch and attach autograd in Cython."""
    import os
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        f"""
        import candle as torch
        import candle._dispatch as dispatch_mod

        x = torch.randn(8, 16, device={str(npu_device)!r}, dtype=torch.float16, requires_grad=True)
        torch.npu.synchronize()

        calls = []
        original_dispatch = dispatch_mod.dispatch

        def wrapped_dispatch(op_name, *args, **kwargs):
            if op_name == "sum":
                calls.append(op_name)
                raise AssertionError("Tensor.sum should bypass Python dispatch")
            return original_dispatch(op_name, *args, **kwargs)

        dispatch_mod.dispatch = wrapped_dispatch
        out = x.sum()
        out.backward()
        torch.npu.synchronize()
        assert calls == []
        assert out.device.type == "npu"
        assert x.grad is not None
        assert x.grad.device.type == "npu"
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_gelu_backward_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """NPU GELU backward should use the Cython native kernel path, not Python aclnn wrapper."""
    import candle as torch
    import candle.nn.functional as F
    import candle._backends.npu.aclnn as aclnn_mod

    x = torch.randn(16, 32, device=npu_device, dtype=torch.float16, requires_grad=True)
    F.gelu(x).sum().backward()
    torch.npu.synchronize()
    x.grad = None

    calls = {"count": 0}
    original_gelu_backward = aclnn_mod.gelu_backward

    def wrapped_gelu_backward(*args, **kwargs):
        calls["count"] += 1
        return original_gelu_backward(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "gelu_backward", wrapped_gelu_backward)

    F.gelu(x).sum().backward()
    torch.npu.synchronize()

    assert calls["count"] == 0
    assert x.grad is not None
    assert x.grad.device.type == "npu"


def test_gelu_backward_uses_direct_npu_kernel_without_formula_redispatch(npu_device, monkeypatch):
    """Safe base NPU GELU backward should not expand into the Python formula ops."""
    import candle as torch
    import candle.nn.functional as F
    import candle._generated.functions as functions_mod

    x = torch.randn(16, 32, device=npu_device, dtype=torch.float16, requires_grad=True)
    F.gelu(x).sum().backward()
    torch.npu.synchronize()
    x.grad = None

    formula_ops = {"div", "erf", "exp"}
    calls = []
    original_redispatch = functions_mod.redispatch

    def wrapped_redispatch(op_name, *args, **kwargs):
        if op_name in formula_ops:
            calls.append(op_name)
            raise AssertionError("gelu backward should use native NPU kernel, not formula redispatch")
        return original_redispatch(op_name, *args, **kwargs)

    monkeypatch.setattr(functions_mod, "redispatch", wrapped_redispatch)

    F.gelu(x).sum().backward()
    torch.npu.synchronize()

    assert calls == []
    assert x.grad is not None
    assert x.grad.device.type == "npu"


def test_fast_le_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """le should bypass candle._backends.npu.aclnn.le_tensor."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.randn(4, 4, device=npu_device)
    b = torch.randn(4, 4, device=npu_device)
    torch.npu.synchronize()

    _ = torch.le(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_le = aclnn_mod.le_tensor

    def wrapped_le(*args, **kwargs):
        calls["count"] += 1
        return original_le(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "le_tensor", wrapped_le)

    _ = torch.le(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_fast_gt_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """gt should bypass candle._backends.npu.aclnn.gt_tensor."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.randn(4, 4, device=npu_device)
    b = torch.randn(4, 4, device=npu_device)
    torch.npu.synchronize()

    _ = torch.gt(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_gt = aclnn_mod.gt_tensor

    def wrapped_gt(*args, **kwargs):
        calls["count"] += 1
        return original_gt(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "gt_tensor", wrapped_gt)

    _ = torch.gt(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_fast_ge_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """ge should bypass candle._backends.npu.aclnn.ge_tensor."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.randn(4, 4, device=npu_device)
    b = torch.randn(4, 4, device=npu_device)
    torch.npu.synchronize()

    _ = torch.ge(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_ge = aclnn_mod.ge_tensor

    def wrapped_ge(*args, **kwargs):
        calls["count"] += 1
        return original_ge(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "ge_tensor", wrapped_ge)

    _ = torch.ge(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_fast_eq_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """eq should bypass candle._backends.npu.aclnn.eq_tensor."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.randn(4, 4, device=npu_device)
    b = torch.randn(4, 4, device=npu_device)
    torch.npu.synchronize()

    _ = torch.eq(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_eq = aclnn_mod.eq_tensor

    def wrapped_eq(*args, **kwargs):
        calls["count"] += 1
        return original_eq(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "eq_tensor", wrapped_eq)

    _ = torch.eq(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_fast_ne_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """ne should bypass candle._backends.npu.aclnn.ne_tensor."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.randn(4, 4, device=npu_device)
    b = torch.randn(4, 4, device=npu_device)
    torch.npu.synchronize()

    _ = torch.ne(a, b)
    torch.npu.synchronize()

    calls = {"count": 0}
    original_ne = aclnn_mod.ne_tensor

    def wrapped_ne(*args, **kwargs):
        calls["count"] += 1
        return original_ne(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "ne_tensor", wrapped_ne)

    _ = torch.ne(a, b)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_clamp_min_tensor_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """clamp with tensor min should bypass candle._backends.npu.aclnn.clamp_min_tensor."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.randn(4, 4, device=npu_device)
    min_val = torch.zeros(4, 4, device=npu_device)
    torch.npu.synchronize()

    _ = torch.clamp(a, min_val)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.clamp_min_tensor

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "clamp_min_tensor", wrapped)

    _ = torch.clamp(a, min_val)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_clamp_max_tensor_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """clamp with tensor max should bypass candle._backends.npu.aclnn.clamp_max_tensor."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.randn(4, 4, device=npu_device)
    max_val = torch.zeros(4, 4, device=npu_device)
    torch.npu.synchronize()

    _ = torch.clamp(a, None, max_val)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.clamp_max_tensor

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "clamp_max_tensor", wrapped)

    _ = torch.clamp(a, None, max_val)
    torch.npu.synchronize()

    assert calls["count"] == 0


def test_clamp_tensor_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """clamp with tensor min and max should bypass candle._backends.npu.aclnn.clamp_tensor."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.randn(4, 4, device=npu_device)
    min_val = torch.full((4, 4), -0.5, device=npu_device)
    max_val = torch.full((4, 4), 0.5, device=npu_device)
    torch.npu.synchronize()

    _ = torch.clamp(a, min_val, max_val)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.clamp_tensor

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "clamp_tensor", wrapped)

    _ = torch.clamp(a, min_val, max_val)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_trunc_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.trunc should bypass candle._backends.npu.aclnn.trunc."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), 0.5, device=npu_device)
    torch.npu.synchronize()

    _ = torch.trunc(a)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.trunc

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "trunc", wrapped)

    _ = torch.trunc(a)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_abs_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.abs should bypass candle._backends.npu.aclnn.abs."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), -0.5, device=npu_device)
    torch.npu.synchronize()

    _ = torch.abs(a)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.abs

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "abs", wrapped)

    _ = torch.abs(a)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_neg_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.neg should bypass candle._backends.npu.aclnn.neg."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), -0.5, device=npu_device)
    torch.npu.synchronize()

    _ = torch.neg(a)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.neg

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "neg", wrapped)

    _ = torch.neg(a)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_sign_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.sign should bypass candle._backends.npu.aclnn.sign."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), -0.5, device=npu_device)
    torch.npu.synchronize()

    _ = torch.sign(a)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.sign

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "sign", wrapped)

    _ = torch.sign(a)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_square_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.square should bypass candle._backends.npu.aclnn.square."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), 0.5, device=npu_device)
    torch.npu.synchronize()

    _ = torch.square(a)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.square

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "square", wrapped)

    _ = torch.square(a)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_signbit_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.signbit should bypass candle._backends.npu.aclnn.signbit."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), -0.5, device=npu_device)
    torch.npu.synchronize()

    _ = torch.signbit(a)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.signbit

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "signbit", wrapped)

    _ = torch.signbit(a)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_isfinite_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.isfinite should bypass candle._backends.npu.aclnn.isfinite."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), 0.5, device=npu_device)
    torch.npu.synchronize()

    _ = torch.isfinite(a)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.isfinite

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "isfinite", wrapped)

    _ = torch.isfinite(a)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_isinf_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.isinf should bypass candle._backends.npu.aclnn.isinf."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), 0.5, device=npu_device)
    torch.npu.synchronize()

    _ = torch.isinf(a)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.isinf

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "isinf", wrapped)

    _ = torch.isinf(a)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_isinf_skips_python_aclnn_logical_wrappers(npu_device, monkeypatch):
    """torch.isinf should bypass candle._backends.npu.aclnn logical wrappers on the composite path."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.tensor([0.0, float("inf"), float("nan"), -float("inf")], device=npu_device)
    torch.npu.synchronize()

    _ = torch.isinf(a)
    torch.npu.synchronize()

    calls = {"not": 0, "and": 0}
    original_not = aclnn_mod.logical_not
    original_and = aclnn_mod.logical_and

    def wrapped_not(*args, **kwargs):
        calls["not"] += 1
        return original_not(*args, **kwargs)

    def wrapped_and(*args, **kwargs):
        calls["and"] += 1
        return original_and(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "logical_not", wrapped_not)
    monkeypatch.setattr(aclnn_mod, "logical_and", wrapped_and)

    out = torch.isinf(a)
    torch.npu.synchronize()

    assert calls["not"] == 0
    assert calls["and"] == 0
    assert out.tolist() == [False, True, False, True]



def test_isposinf_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.isposinf should bypass candle._backends.npu.aclnn.isposinf."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), 0.5, device=npu_device)
    torch.npu.synchronize()

    _ = torch.isposinf(a)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.isposinf

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "isposinf", wrapped)

    _ = torch.isposinf(a)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_isneginf_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.isneginf should bypass candle._backends.npu.aclnn.isneginf."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), -0.5, device=npu_device)
    torch.npu.synchronize()

    _ = torch.isneginf(a)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.isneginf

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "isneginf", wrapped)

    _ = torch.isneginf(a)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_logical_not_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.logical_not should bypass candle._backends.npu.aclnn.logical_not."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), True, device=npu_device)
    torch.npu.synchronize()

    _ = torch.logical_not(a)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.logical_not

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "logical_not", wrapped)

    _ = torch.logical_not(a)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_bitwise_not_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.bitwise_not should bypass candle._backends.npu.aclnn.bitwise_not."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), 1, device=npu_device, dtype=torch.int32)
    torch.npu.synchronize()

    _ = torch.bitwise_not(a)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.bitwise_not

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "bitwise_not", wrapped)

    _ = torch.bitwise_not(a)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_silu_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.nn.functional.silu should bypass candle._backends.npu.aclnn.silu."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), 0.5, device=npu_device)
    torch.npu.synchronize()

    _ = torch.nn.functional.silu(a)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.silu

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "silu", wrapped)

    _ = torch.nn.functional.silu(a)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_mish_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.nn.functional.mish should bypass candle._backends.npu.aclnn.mish."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), 0.5, device=npu_device)
    torch.npu.synchronize()

    _ = torch.nn.functional.mish(a)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.mish

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "mish", wrapped)

    _ = torch.nn.functional.mish(a)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_gelu_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.nn.functional.gelu should bypass candle._backends.npu.aclnn.gelu."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), 0.5, device=npu_device)
    torch.npu.synchronize()

    _ = torch.nn.functional.gelu(a)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.gelu

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "gelu", wrapped)

    _ = torch.nn.functional.gelu(a)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_npu_gelu_autograd_skips_python_dispatch(npu_device, monkeypatch):
    """Safe base NPU GELU autograd should attach in Cython without Python dispatch."""
    import candle as torch
    import candle.nn.functional as F
    import candle._dispatch as dispatch_mod

    x = torch.randn(16, 32, device=npu_device, dtype=torch.float16, requires_grad=True)
    F.gelu(x).sum().backward()
    torch.npu.synchronize()
    x.grad = None

    calls = []
    original_dispatch = dispatch_mod.dispatch

    def wrapped_dispatch(op_name, *args, **kwargs):
        if op_name == "gelu":
            calls.append(op_name)
            raise AssertionError("gelu should bypass Python dispatch")
        return original_dispatch(op_name, *args, **kwargs)

    monkeypatch.setattr(dispatch_mod, "dispatch", wrapped_dispatch)

    F.gelu(x).sum().backward()
    torch.npu.synchronize()

    assert calls == []
    assert x.grad is not None
    assert x.grad.device.type == "npu"



def test_relu_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.nn.functional.relu should bypass candle._backends.npu.aclnn.relu."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), 0.5, device=npu_device)
    torch.npu.synchronize()

    _ = torch.nn.functional.relu(a)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.relu

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "relu", wrapped)

    _ = torch.nn.functional.relu(a)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_leaky_relu_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.nn.functional.leaky_relu should bypass candle._backends.npu.aclnn.leaky_relu."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), 0.5, device=npu_device)
    torch.npu.synchronize()

    _ = torch.nn.functional.leaky_relu(a, negative_slope=0.1)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.leaky_relu

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "leaky_relu", wrapped)

    _ = torch.nn.functional.leaky_relu(a, negative_slope=0.1)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_elu_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.nn.functional.elu should bypass candle._backends.npu.aclnn.elu."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), 0.5, device=npu_device)
    torch.npu.synchronize()

    _ = torch.nn.functional.elu(a, alpha=1.0)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.elu

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "elu", wrapped)

    _ = torch.nn.functional.elu(a, alpha=1.0)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_softmax_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.nn.functional.softmax should bypass candle._backends.npu.aclnn.softmax."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), 0.5, device=npu_device)
    torch.npu.synchronize()

    _ = torch.nn.functional.softmax(a, dim=-1)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.softmax

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "softmax", wrapped)

    _ = torch.nn.functional.softmax(a, dim=-1)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_log_softmax_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.nn.functional.log_softmax should bypass candle._backends.npu.aclnn.log_softmax."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), 0.5, device=npu_device)
    torch.npu.synchronize()

    _ = torch.nn.functional.log_softmax(a, dim=-1)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.log_softmax

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "log_softmax", wrapped)

    _ = torch.nn.functional.log_softmax(a, dim=-1)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_hardtanh_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.nn.functional.hardtanh should bypass candle._backends.npu.aclnn.hardtanh."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), 0.5, device=npu_device)
    torch.npu.synchronize()

    _ = torch.nn.functional.hardtanh(a, min_val=-0.5, max_val=0.5)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.hardtanh

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "hardtanh", wrapped)

    _ = torch.nn.functional.hardtanh(a, min_val=-0.5, max_val=0.5)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_softplus_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.nn.functional.softplus should bypass candle._backends.npu.aclnn.softplus."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), 0.5, device=npu_device)
    torch.npu.synchronize()

    _ = torch.nn.functional.softplus(a, beta=1.0, threshold=20.0)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.softplus

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "softplus", wrapped)

    _ = torch.nn.functional.softplus(a, beta=1.0, threshold=20.0)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_prelu_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.nn.functional.prelu should bypass candle._backends.npu.aclnn.prelu."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((2, 3, 4, 4), 0.5, device=npu_device)
    weight = torch.full((3,), 0.25, device=npu_device)
    torch.npu.synchronize()

    _ = torch.nn.functional.prelu(a, weight)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.prelu

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "prelu", wrapped)

    _ = torch.nn.functional.prelu(a, weight)
    torch.npu.synchronize()

    assert calls["count"] == 0



def test_relu__skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.nn.functional.relu_ should bypass candle._backends.npu.aclnn.relu."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), -0.5, device=npu_device)
    torch.npu.synchronize()

    _ = torch.nn.functional.relu_(a)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.relu

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "relu", wrapped)

    _ = torch.nn.functional.relu_(a)
    torch.npu.synchronize()

    assert calls["count"] == 0

    expected = torch.zeros((4, 4), device=npu_device)
    assert torch.equal(a, expected)
    assert torch.equal(_, a)



def test_embedding_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.nn.functional.embedding should bypass candle._backends.npu.aclnn.embedding."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    weight = torch.arange(0, 24, device=npu_device, dtype=torch.float32).reshape(6, 4)
    indices = torch.tensor([0, 2, 5], device=npu_device, dtype=torch.int64)
    torch.npu.synchronize()

    _ = torch.nn.functional.embedding(indices, weight)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.embedding

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "embedding", wrapped)

    out = torch.nn.functional.embedding(indices, weight)
    torch.npu.synchronize()

    assert calls["count"] == 0
    assert out.shape == (3, 4)
    assert torch.equal(out[0], weight[0])
    assert torch.equal(out[1], weight[2])
    assert torch.equal(out[2], weight[5])



def test_dropout_skips_python_aclnn_wrappers(npu_device, monkeypatch):
    """torch.nn.functional.dropout should bypass candle._backends.npu.aclnn dropout wrappers."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((8, 8), 1.0, device=npu_device)
    torch.npu.synchronize()

    _ = torch.nn.functional.dropout(a, p=0.25, training=True)
    torch.npu.synchronize()

    calls = {"gen": 0, "do": 0}
    original_gen = aclnn_mod.dropout_gen_mask
    original_do = aclnn_mod.dropout_do_mask

    def wrapped_gen(*args, **kwargs):
        calls["gen"] += 1
        return original_gen(*args, **kwargs)

    def wrapped_do(*args, **kwargs):
        calls["do"] += 1
        return original_do(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "dropout_gen_mask", wrapped_gen)
    monkeypatch.setattr(aclnn_mod, "dropout_do_mask", wrapped_do)

    out = torch.nn.functional.dropout(a, p=0.25, training=True)
    torch.npu.synchronize()

    assert calls["gen"] == 0
    assert calls["do"] == 0
    assert out.shape == a.shape
    assert out.device == a.device
    assert out._backward_data["p"] == 0.25
    assert out._backward_data["mask_numel"] > 0
    assert out._backward_data["mask_ptr"]



def test_isnan_skips_python_aclnn_wrappers(npu_device, monkeypatch):
    """torch.isnan should bypass candle._backends.npu.aclnn logical wrappers."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.tensor([0.0, float("nan"), float("inf"), -1.0], device=npu_device)
    torch.npu.synchronize()

    _ = torch.isnan(a)
    torch.npu.synchronize()

    calls = {"not": 0, "and": 0}
    original_not = aclnn_mod.logical_not
    original_and = aclnn_mod.logical_and

    def wrapped_not(*args, **kwargs):
        calls["not"] += 1
        return original_not(*args, **kwargs)

    def wrapped_and(*args, **kwargs):
        calls["and"] += 1
        return original_and(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "logical_not", wrapped_not)
    monkeypatch.setattr(aclnn_mod, "logical_and", wrapped_and)

    out = torch.isnan(a)
    torch.npu.synchronize()

    assert calls["not"] == 0
    assert calls["and"] == 0
    assert out.tolist() == [False, True, False, False]



def test_reciprocal__skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """Tensor.reciprocal_ should bypass candle._backends.npu.aclnn.reciprocal."""
    import numpy as np
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), 2.0, device=npu_device)
    torch.npu.synchronize()

    _ = a.reciprocal_()
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.reciprocal

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "reciprocal", wrapped)

    out = a.reciprocal_()
    torch.npu.synchronize()

    assert calls["count"] == 0
    expected = torch.full((4, 4), 2.0, device=npu_device)
    assert np.allclose(a.cpu().numpy(), expected.cpu().numpy())
    assert torch.equal(out, a)



def test_reciprocal_skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """torch.reciprocal should bypass candle._backends.npu.aclnn.reciprocal."""
    import numpy as np
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), 2.0, device=npu_device)
    torch.npu.synchronize()

    _ = torch.reciprocal(a)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.reciprocal

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "reciprocal", wrapped)

    out = torch.reciprocal(a)
    torch.npu.synchronize()

    assert calls["count"] == 0
    expected = torch.full((4, 4), 0.5, device=npu_device)
    assert np.allclose(out.cpu().numpy(), expected.cpu().numpy())



def test_zero__skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """Tensor.zero_ should bypass candle._backends.npu.aclnn.inplace_zero."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.full((4, 4), 2.0, device=npu_device)
    torch.npu.synchronize()

    _ = a.zero_()
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.inplace_zero

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "inplace_zero", wrapped)

    out = a.zero_()
    torch.npu.synchronize()

    assert calls["count"] == 0
    expected = torch.zeros((4, 4), device=npu_device)
    assert torch.equal(a, expected)
    assert torch.equal(out, a)



def test_fill__skips_python_aclnn_wrapper(npu_device, monkeypatch):
    """Tensor.fill_ should bypass candle._backends.npu.aclnn.inplace_fill_scalar."""
    import candle as torch
    import candle._backends.npu.aclnn as aclnn_mod

    a = torch.zeros((4, 4), device=npu_device)
    torch.npu.synchronize()

    _ = a.fill_(2.0)
    torch.npu.synchronize()

    calls = {"count": 0}
    original = aclnn_mod.inplace_fill_scalar

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(aclnn_mod, "inplace_fill_scalar", wrapped)

    out = a.fill_(3.0)
    torch.npu.synchronize()

    assert calls["count"] == 0
    expected = torch.full((4, 4), 3.0, device=npu_device)
    assert torch.equal(a, expected)
    assert torch.equal(out, a)




