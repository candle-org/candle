import pytest
import candle as torch

from candle._dispatch.keys import DispatchKey, DispatchKeySet


class _FakeDevice:
    def __init__(self, device_type, index=None):
        self.type = device_type
        self.index = index


class _FakeTensor:
    """Minimal tensor-like object for keyset construction tests.

    DispatchKeySet.from_tensors() only inspects `.device` and `requires_grad`
    for these cases, so a fake MPS tensor is sufficient on Linux CI where real
    MPS tensors are unavailable.
    """

    def __init__(self, device_type, *, requires_grad=False):
        self.device = _FakeDevice(device_type)
        self.requires_grad = requires_grad


def _mask(*keys):
    value = 0
    for key in keys:
        value |= int(key)
    return value


def test_dispatch_key_numeric_values_match_runtime_core_expectations():
    expected = {
        "CPU": 1 << 15,
        "NPU": 1 << 13,
        "CUDA": 1 << 14,
        "Meta": 1 << 12,
        "AutogradCPU": 1 << 6,
        "AutogradNPU": 1 << 7,
        "AutogradCUDA": 1 << 8,
        "AutogradMeta": 1 << 10,
        "ADInplaceOrView": 1 << 4,
        "Autograd": 1 << 11,
        "Functionalize": 1 << 3,
        "Autocast": 1 << 19,
        "Pipeline": 1 << 1,
        "Python": 1 << 2,
        "PrivateUse2": 1 << 21,
        "PrivateUse3": 1 << 22,
    }
    actual = {name: int(getattr(DispatchKey, name)) for name in expected}
    assert actual == expected


@pytest.mark.parametrize(
    "label,tensors,kwargs,expected_mask,reason",
    [
        (
            "cpu",
            lambda: (torch.ones((2,)),),
            {"grad_enabled": False},
            _mask(DispatchKey.CPU),
            "Plain Tensor instances must not set DispatchKey.Python; a Python bit here indicates DispatchKeySet.from_tensors() is treating the base Tensor class as a __torch_dispatch__ subclass.",
        ),
        (
            "cpu_autograd",
            lambda: (torch.ones((2,)).requires_grad_(),),
            {"grad_enabled": True},
            _mask(
                DispatchKey.CPU,
                DispatchKey.ADInplaceOrView,
                DispatchKey.Autograd,
                DispatchKey.AutogradCPU,
            ),
            "CPU autograd keyset should include only CPU + autograd-related bits.",
        ),
        (
            "meta_autograd",
            lambda: (torch.ones((2,), device="meta").requires_grad_(),),
            {"grad_enabled": True},
            _mask(
                DispatchKey.Meta,
                DispatchKey.ADInplaceOrView,
                DispatchKey.Autograd,
                DispatchKey.AutogradMeta,
            ),
            "Meta autograd keyset should include only Meta + autograd-related bits.",
        ),
        (
            "mps_autograd",
            lambda: (_FakeTensor("mps", requires_grad=True),),
            {"grad_enabled": True},
            _mask(
                DispatchKey.PrivateUse2,
                DispatchKey.ADInplaceOrView,
                DispatchKey.Autograd,
                DispatchKey.PrivateUse3,
            ),
            "MPS is represented as PrivateUse2 / PrivateUse3 in DispatchKey; from_tensors() should derive that mask from .device.type and requires_grad.",
        ),
    ],
)
def test_dispatch_keyset_masks_match_runtime_core_constants(
    label, tensors, kwargs, expected_mask, reason
):
    keyset = DispatchKeySet.from_tensors(tensors(), **kwargs)
    assert keyset.mask == expected_mask, (
        f"{label} keyset mask mismatch: expected {expected_mask:#x}, got {keyset.mask:#x}. {reason}"
    )
