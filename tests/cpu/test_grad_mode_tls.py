import importlib.machinery
import threading

import candle as torch
from candle._C import _grad_mode_state


def test_grad_mode_state_lives_in_compiled_c_boundary():
    assert isinstance(
        _grad_mode_state.__loader__,
        importlib.machinery.ExtensionFileLoader,
    )
    assert torch.no_grad.__module__ == "candle._C._grad_mode_state"
    assert torch.enable_grad.__module__ == "candle._C._grad_mode_state"
    assert torch.set_grad_enabled.__module__ == "candle._C._grad_mode_state"
    assert torch.inference_mode.__module__ == "candle._C._grad_mode_state"


def _cython_source_path_for_extension_path(path):
    """Return the .pyx path for a compiled extension path on any CI platform."""
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        if path.endswith(suffix):
            base = path[: -len(suffix)]
            marker_index = base.find(".cpython-")
            if marker_index != -1:
                base = base[:marker_index]
            return base + ".pyx"
    if path.endswith(".so"):
        base = path[: -len(".so")]
        marker_index = base.find(".cpython-")
        if marker_index != -1:
            base = base[:marker_index]
        return base + ".pyx"
    return path


def test_grad_mode_source_lookup_handles_platform_extension_suffixes():
    assert _cython_source_path_for_extension_path(
        "/tmp/_grad_mode_state.cpython-311-x86_64-linux-gnu.so"
    ) == "/tmp/_grad_mode_state.pyx"
    assert _cython_source_path_for_extension_path(
        "/tmp/_grad_mode_state.cpython-311-aarch64-linux-gnu.so"
    ) == "/tmp/_grad_mode_state.pyx"


def test_grad_mode_default_check_is_c_thread_local_fast_path():
    src = _cython_source_path_for_extension_path(_grad_mode_state.__loader__.path)
    with open(src, encoding="utf-8") as handle:
        content = handle.read()
    # Default-on grad mode must be a C thread-local read, not a
    # threading.local getattr that raises AttributeError on every call
    # before the first no_grad/set_grad_enabled in a thread.
    assert "__thread" in content
    assert 'getattr(_STATE, "enabled", True)' not in content
    assert "_STATE.enabled" not in content


def test_no_grad_is_thread_local():
    parent_states = []
    child_states = []

    with torch.no_grad():
        parent_states.append(torch.is_grad_enabled())

        def _worker():
            child_states.append(torch.is_grad_enabled())
            with torch.no_grad():
                child_states.append(torch.is_grad_enabled())
            child_states.append(torch.is_grad_enabled())

        t = threading.Thread(target=_worker)
        t.start()
        t.join()

        parent_states.append(torch.is_grad_enabled())

    assert parent_states == [False, False]
    assert child_states == [True, False, True]


def test_set_grad_enabled_is_thread_local():
    observed = []

    def _worker():
        observed.append(torch.is_grad_enabled())
        with torch.set_grad_enabled(False):
            observed.append(torch.is_grad_enabled())
        observed.append(torch.is_grad_enabled())

    with torch.set_grad_enabled(False):
        assert torch.is_grad_enabled() is False
        t = threading.Thread(target=_worker)
        t.start()
        t.join()
        assert torch.is_grad_enabled() is False

    assert observed == [True, False, True]


def test_no_grad_decorator_forms():
    @torch.no_grad()
    def fn_a():
        return torch.is_grad_enabled()

    @torch.no_grad
    def fn_b():
        return torch.is_grad_enabled()

    assert fn_a() is False
    assert fn_b() is False
    assert torch.is_grad_enabled() is True


def test_no_grad_context_does_not_leak_after_backward():
    x = torch.tensor([2.0], requires_grad=True)
    with torch.no_grad():
        assert torch.is_grad_enabled() is False
        y = x.clone()
    assert torch.is_grad_enabled() is True
    (x * x).sum().backward()
    assert x.grad is not None
    assert torch.is_grad_enabled() is True
