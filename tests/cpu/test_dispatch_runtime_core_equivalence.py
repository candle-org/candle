import importlib
import sys

import candle as torch

from candle._dispatch import dispatcher as dispatcher_mod
from candle._dispatch.dispatcher import dispatch



def test_public_dispatch_api_is_bound_to_cython_runtime_core():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])

    out = dispatch("add", None, a, b)

    torch.testing.assert_close(out, a + b)
    assert hasattr(dispatcher_mod, "cy_dispatch_full")
    assert hasattr(dispatcher_mod, "cy_dispatch_with_keyset_fast")
    assert dispatcher_mod.dispatch is dispatcher_mod.cy_dispatch_full
    assert dispatcher_mod.dispatch_with_keyset is dispatcher_mod.cy_dispatch_with_keyset_fast



def test_public_dispatch_inplace_updates_version_counter():
    a = torch.tensor([1.0])
    inc = torch.tensor([2.0])
    v0 = a._version_counter.value

    dispatch("add_", None, a, inc)

    assert a._version_counter.value == v0 + 1



def test_public_dispatch_autograd_attachment_matches_runtime_core():
    x = torch.tensor([1.0, 2.0]).requires_grad_()
    y = torch.tensor([3.0, 4.0]).requires_grad_()

    out = dispatch("add", None, x, y)

    assert out.requires_grad is True
    assert out.grad_fn is not None
    assert type(out.grad_fn).__name__ in ("AddBackward0", "AddTensorBackward0")
    assert len(out.grad_fn.next_functions) == 2



def test_dispatch_module_no_longer_exposes_python_fallback_entrypoints():
    assert not hasattr(dispatcher_mod, "_py_dispatch")
    assert not hasattr(dispatcher_mod, "_py_dispatch_with_keyset")



def test_dispatch_import_without_dispatcher_core_is_hard_failure(monkeypatch):
    monkeypatch.setitem(sys.modules, "candle._cython._dispatcher_core", None)
    monkeypatch.delitem(sys.modules, "candle._dispatch.dispatcher", raising=False)

    try:
        importlib.import_module("candle._dispatch.dispatcher")
    except ImportError as exc:
        msg = str(exc)
        assert "_dispatcher_core" in msg
        assert any(
            hint in msg
            for hint in ("build_ext", "pip install", "setup.py", "[cython]")
        )
        return

    raise AssertionError(
        "dispatcher import must hard-fail when _dispatcher_core is unavailable; "
        "Python fallback is forbidden."
    )
