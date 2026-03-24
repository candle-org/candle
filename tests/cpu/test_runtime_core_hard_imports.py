import importlib
import os
import subprocess
import sys

import pytest



def test_tensor_module_import_without_tensor_impl_is_hard_failure(monkeypatch):
    monkeypatch.setitem(sys.modules, "candle._cython._tensor_impl", None)
    monkeypatch.delitem(sys.modules, "candle._tensor", raising=False)

    with pytest.raises(ImportError) as exc:
        importlib.import_module("candle._tensor")

    msg = str(exc.value)
    assert "_tensor_impl" in msg
    assert any(
        hint in msg
        for hint in ("build_ext", "pip install", "setup.py", "[cython]")
    )



def test_device_module_import_without_fastdevice_is_hard_failure(monkeypatch):
    monkeypatch.setitem(sys.modules, "candle._cython._device", None)
    monkeypatch.delitem(sys.modules, "candle._device", raising=False)

    with pytest.raises(ImportError) as exc:
        importlib.import_module("candle._device")

    msg = str(exc.value)
    assert "_device" in msg
    assert any(
        hint in msg
        for hint in ("build_ext", "pip install", "setup.py", "[cython]")
    )



def test_dtype_module_import_without_fastdtype_is_hard_failure(monkeypatch):
    monkeypatch.setitem(sys.modules, "candle._cython._dtype", None)
    monkeypatch.delitem(sys.modules, "candle._dtype", raising=False)

    with pytest.raises(ImportError) as exc:
        importlib.import_module("candle._dtype")

    msg = str(exc.value)
    assert "_dtype" in msg
    assert any(
        hint in msg
        for hint in ("build_ext", "pip install", "setup.py", "[cython]")
    )



def test_cython_package_import_without_dispatcher_core_is_hard_failure(monkeypatch):
    monkeypatch.setitem(sys.modules, "candle._cython._dispatcher_core", None)
    monkeypatch.delitem(sys.modules, "candle._cython", raising=False)

    with pytest.raises(ImportError) as exc:
        importlib.import_module("candle._cython")

    msg = str(exc.value)
    assert "_dispatcher_core" in msg
    assert any(
        hint in msg
        for hint in ("build_ext", "pip install", "setup.py", "[cython]")
    )



def test_public_dtype_singletons_are_fastdtype_instances():
    import candle as torch
    from candle._cython._dtype import FastDType

    assert isinstance(torch.float32, FastDType)
    assert isinstance(torch.float16, FastDType)
    assert isinstance(torch.int64, FastDType)



def test_cython_package_runtime_core_flags_are_true():
    import candle._cython as c

    assert c._HAS_CYTHON_TENSOR_IMPL is True
    assert c._HAS_CYTHON_DISPATCHER_CORE is True
    assert c._HAS_CYTHON_DEVICE is True
    assert c._HAS_CYTHON_DTYPE is True
    assert c._HAS_CYTHON_DISPATCH is True
    assert c._HAS_CYTHON_STORAGE is True
    assert c._HAS_CYTHON_FAST_OPS is True



def test_import_candle_succeeds_with_built_runtime_core():
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import candle; import candle._dispatch.dispatcher as d; print('OK', d.dispatch.__module__)",
        ],
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "_dispatcher_core" in proc.stdout
