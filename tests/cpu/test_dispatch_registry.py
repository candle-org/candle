import candle as torch
from candle._dispatch import dispatcher
from candle._dispatch.dispatcher import dispatch_with_keyset
from candle._dispatch.keys import DispatchKey, DispatchKeySet
from candle._dispatch.registry import registry


def test_backend_registration_uses_keys():
    entry = registry.get("aten::add")
    assert DispatchKey.CPU in entry.kernels
    assert DispatchKey.NPU in entry.kernels
    assert DispatchKey.CUDA in entry.kernels
    assert DispatchKey.Meta in entry.kernels



def test_autograd_backend_key_registration_prefers_device_keys():
    entry = registry.get("aten::add")
    assert DispatchKey.AutogradCPU in entry.kernels
    assert DispatchKey.AutogradNPU in entry.kernels
    assert DispatchKey.AutogradCUDA in entry.kernels
    assert DispatchKey.AutogradMeta in entry.kernels



def test_dispatch_core_executes_registered_kernel_without_python_relookup(monkeypatch):
    x = torch.tensor([1.0, 2.0], dtype=torch.float32)
    y = torch.tensor([3.0, 4.0], dtype=torch.float32)
    keyset = DispatchKeySet.from_tensors(
        [x, y],
        grad_enabled=False,
        pipeline_enabled=False,
        functionalize_enabled=False,
        device=None,
        autocast_enabled=False,
    )

    def _unexpected_python_lookup(*args, **kwargs):
        raise AssertionError("python _kernel_for_entry should not run on cython hot path")

    monkeypatch.setattr(dispatcher, "_kernel_for_entry", _unexpected_python_lookup)

    out = dispatch_with_keyset("add", keyset, None, x, y)
    assert out.tolist() == [4.0, 6.0]
