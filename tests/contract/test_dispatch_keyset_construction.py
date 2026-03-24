from candle._dispatch.keys import DispatchKey, DispatchKeySet
import candle as torch


def test_keyset_excludes_placeholders_by_default():
    t = torch.ones((2,))
    keyset = DispatchKeySet.from_tensors((t,))
    assert DispatchKey.BackendSelect not in keyset
    assert DispatchKey.AutogradNPU not in keyset
    assert DispatchKey.Autocast not in keyset


def test_keyset_includes_pipeline_only_when_enabled():
    t = torch.ones((2,))
    keyset = DispatchKeySet.from_tensors((t,), pipeline_enabled=False)
    assert DispatchKey.Pipeline not in keyset
    keyset = DispatchKeySet.from_tensors((t,), pipeline_enabled=True)
    assert DispatchKey.Pipeline in keyset


def test_keyset_includes_autograd_cpu_when_grad_enabled():
    t = torch.ones((2,)).requires_grad_()
    keyset = DispatchKeySet.from_tensors((t,), grad_enabled=True)
    assert DispatchKey.Autograd in keyset
    assert DispatchKey.AutogradCPU in keyset


def test_keyset_includes_autograd_meta_when_grad_enabled():
    t = torch.ones((2,), device="meta").requires_grad_()
    keyset = DispatchKeySet.from_tensors((t,), grad_enabled=True)
    assert DispatchKey.Autograd in keyset
    assert DispatchKey.AutogradMeta in keyset


def test_keyset_includes_autograd_npu_when_grad_enabled():
    if not torch.npu.is_available():
        return
    t = torch.ones((2,), device="npu").requires_grad_()
    keyset = DispatchKeySet.from_tensors((t,), grad_enabled=True)
    assert DispatchKey.Autograd in keyset
    assert DispatchKey.AutogradNPU in keyset


def test_keyset_includes_adinplaceorview_when_grad_enabled():
    t = torch.ones((2,)).requires_grad_()
    keyset = DispatchKeySet.from_tensors((t,), grad_enabled=True)
    assert DispatchKey.ADInplaceOrView in keyset


def test_registration_helpers_accept_mps_labels():
    import uuid

    from candle._dispatch.registration import register_autograd_kernels, register_forward_kernels
    from candle._dispatch.registry import dispatch_key_from_string, registry

    op_name = f"mps_registration_{uuid.uuid4().hex}"
    registry.register_schema(op_name, f"{op_name}(Tensor input) -> Tensor")

    def _forward(x):
        return x

    def _autograd(x):
        return x

    register_forward_kernels(op_name, mps=_forward)
    register_autograd_kernels(op_name, mps=_autograd)

    entry = registry.get(op_name)
    assert entry.kernels[dispatch_key_from_string("MPS")] is _forward
    assert entry.kernels[dispatch_key_from_string("AutogradMPS")] is _autograd
