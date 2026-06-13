import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]


def _read(relpath):
    return (_REPO_ROOT / relpath).read_text(encoding="utf-8")


def test_backward_redispatch_fast_path_is_cython_only():
    """Hot NPU backward redispatch ops should route directly to Cython kernels."""
    src = _read("src/candle/_C/_dispatcher_core.pyx")
    assert "from candle._C._npu_ops cimport" in src
    for symbol in (
        "fast_add_exact as _redispatch_fast_npu_add_exact",
        "fast_sub_exact as _redispatch_fast_npu_sub_exact",
        "fast_mul_exact as _redispatch_fast_npu_mul_exact",
        "fast_div_exact as _redispatch_fast_npu_div_exact",
        "fast_neg as _redispatch_fast_npu_neg",
    ):
        assert symbol in src

    assert "cdef inline object _try_fast_npu_backward_redispatch" in src
    fast_path = re.search(
        r"cdef inline object _try_fast_npu_backward_redispatch\(.*?\) except \*:(?P<body>.*?)\n\n\n#",
        src,
        flags=re.DOTALL,
    )
    assert fast_path is not None
    body = fast_path.group("body")
    for call in (
        "_redispatch_fast_npu_mul_exact(<TensorImpl>a, <TensorImpl>b)",
        "_redispatch_fast_npu_div_exact(<TensorImpl>a, <TensorImpl>b)",
        "_redispatch_fast_npu_add_exact(<TensorImpl>a, <TensorImpl>b)",
        "_redispatch_fast_npu_sub_exact(<TensorImpl>a, <TensorImpl>b)",
        "_redispatch_fast_npu_neg(a)",
    ):
        assert call in body
    assert "_registry" not in body
    assert "schema_obj" not in body
    assert ".bind(" not in body


def test_backward_redispatch_fast_path_has_raw_exact_npu_guards():
    """The Cython bypass is only valid for raw same-device exact-base NPU redispatch."""
    src = _read("src/candle/_C/_dispatcher_core.pyx")
    fast_path = re.search(
        r"cdef inline object _try_fast_npu_backward_redispatch\(.*?\) except \*:(?P<body>.*?)\n\n\n#",
        src,
        flags=re.DOTALL,
    )
    assert fast_path is not None
    body = fast_path.group("body")

    assert "if keyset is None:" in body
    assert "if kwargs:" in body
    assert "m != _KEY_NPU" in body
    assert "name == \"true_divide\"" in body
    assert "type(a) is not _BaseTensor" in body
    assert "type(b) is not _BaseTensor" in body
    assert "(<TensorImpl>a)._device_type != 1" in body
    assert "(<TensorImpl>b)._device_type != 1" in body
    assert "(<TensorImpl>a)._device_index != (<TensorImpl>b)._device_index" in body
    assert "(<TensorImpl>a)._shape_tuple != (<TensorImpl>b)._shape_tuple" in body


def test_fast_neg_is_cpdef_for_cython_redispatch_cimport():
    """Unary neg must be cimportable by the dispatcher Cython fast path."""
    pxd = _read("src/candle/_C/_npu_ops.pxd")
    pyx = _read("src/candle/_C/_npu_ops.pyx")
    assert "cpdef object fast_neg(object a)" in pxd
    assert "cpdef object fast_neg(object a):" in pyx


def test_backward_redispatch_public_python_remains_thin():
    """Do not add a parallel Python hot helper; redispatch should stay a thin Cython entry."""
    src = _read("src/candle/_dispatch/dispatcher.py")
    match = re.search(r"def redispatch\(name, keyset, \*args, \*\*kwargs\):(?P<body>.*?)\n\n", src, re.DOTALL)
    assert match is not None
    body = match.group("body")
    assert "dispatch_with_keyset(name, keyset, None, *args, **kwargs)" in body
    assert "fast_npu" not in body
    assert "_C._npu_ops" not in body


def test_npu_backward_redispatch_mul_bypasses_python_backend(npu_device, monkeypatch):
    """Generated/backward redispatch mul should not call the Python NPU backend wrapper."""
    import numpy as np
    import candle as torch
    from candle._dispatch.keys import DispatchKey
    from candle._dispatch.registry import registry

    calls = {"mul": 0}

    def fail_python_mul(*args, **kwargs):
        calls["mul"] += 1
        raise AssertionError("NPU backward redispatch mul should use the Cython exact kernel")

    monkeypatch.setitem(registry.get("mul").kernels, DispatchKey.NPU, fail_python_mul)

    x = torch.tensor([1.0, 2.0, 3.0, 4.0], device=npu_device, dtype=torch.float32, requires_grad=True)
    y = torch.tensor([5.0, 6.0, 7.0, 8.0], device=npu_device, dtype=torch.float32, requires_grad=True)
    out = (x * y).sum()
    out.backward()
    torch.npu.synchronize()

    assert calls["mul"] == 0
    np.testing.assert_allclose(x.grad.to("cpu").numpy(), [5.0, 6.0, 7.0, 8.0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(y.grad.to("cpu").numpy(), [1.0, 2.0, 3.0, 4.0], rtol=1e-6, atol=1e-6)


def test_npu_backward_redispatch_div_neg_bypasses_python_backend(npu_device, monkeypatch):
    """Div backward uses div/mul/neg redispatch; covered exact ops should stay in Cython."""
    import numpy as np
    import candle as torch
    from candle._dispatch.keys import DispatchKey
    from candle._dispatch.registry import registry

    calls = {"div": 0, "mul": 0, "neg": 0}

    def fail(name):
        def inner(*args, **kwargs):
            calls[name] += 1
            raise AssertionError(f"NPU backward redispatch {name} should use Cython exact kernel")
        return inner

    monkeypatch.setitem(registry.get("div").kernels, DispatchKey.NPU, fail("div"))
    monkeypatch.setitem(registry.get("mul").kernels, DispatchKey.NPU, fail("mul"))
    monkeypatch.setitem(registry.get("neg").kernels, DispatchKey.NPU, fail("neg"))

    x_values = np.array([4.0, 9.0, 16.0, 25.0], dtype=np.float32)
    y_values = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    x = torch.tensor(x_values, device=npu_device, dtype=torch.float32, requires_grad=True)
    y = torch.tensor(y_values, device=npu_device, dtype=torch.float32, requires_grad=True)
    out = (x / y).sum()
    out.backward()
    torch.npu.synchronize()

    assert calls == {"div": 0, "mul": 0, "neg": 0}
    np.testing.assert_allclose(x.grad.to("cpu").numpy(), 1.0 / y_values, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(y.grad.to("cpu").numpy(), -x_values / (y_values * y_values), rtol=1e-6, atol=1e-6)
