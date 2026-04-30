import pytest

import candle as torch
from candle._backends.cuda import creation as cuda_creation
from candle._backends.mps import creation as mps_creation
from candle._backends.npu import creation as npu_creation
from candle._device import device


def test_meta_empty_supports_channels_last_stride():
    x = torch.empty((2, 3, 4, 5), device="meta", memory_format=torch.channels_last)
    assert x.stride() == (60, 1, 15, 3)
    assert x.is_contiguous(memory_format=torch.channels_last) is True


def test_meta_to_channels_last_preserves_layout_metadata():
    x = torch.empty((2, 3, 4, 5), device="meta")
    y = x.to(memory_format=torch.channels_last)
    assert y.stride() == (60, 1, 15, 3)
    assert y.is_contiguous(memory_format=torch.channels_last) is True


def test_preserve_format_rejects_channels_last_relayout_to_non_cpu_meta(monkeypatch):
    x = torch.empty((2, 3, 4, 5), memory_format=torch.channels_last)

    def _fail_copy(*args, **kwargs):
        raise AssertionError("preserve_format should reject before device copy")

    monkeypatch.setattr(npu_creation.npu_runtime, "_copy_cpu_to_npu", _fail_copy)

    with pytest.raises(NotImplementedError, match="channels_last|memory_format"):
        x.to("npu", memory_format=torch.preserve_format)


def test_channels_last_to_non_cpu_rejects_before_copy(monkeypatch):
    x = torch.empty((2, 3, 4, 5))

    def _fail_copy(*args, **kwargs):
        raise AssertionError("channels_last should reject before device copy")

    monkeypatch.setattr(npu_creation.npu_runtime, "_copy_cpu_to_npu", _fail_copy)

    with pytest.raises(NotImplementedError, match="channels_last|memory_format"):
        x.to("npu", memory_format=torch.channels_last)


def test_contiguous_format_to_makes_channels_last_tensor_row_major():
    x = torch.empty((2, 3, 4, 5), memory_format=torch.channels_last)
    y = x.to(memory_format=torch.contiguous_format)

    assert y is not x
    assert y.stride() == (60, 20, 5, 1)
    assert y.is_contiguous() is True
    assert y.is_contiguous(memory_format=torch.channels_last) is False


def test_backend_creation_functions_reject_channels_last_before_allocation():
    creation_cases = (
        (mps_creation, "mps", ("empty_create", "zeros_create", "ones_create", "randn_create", "rand_create")),
        (cuda_creation, "cuda", ("empty_create", "zeros_create", "ones_create")),
        (npu_creation, "npu", ("empty_create", "zeros_create", "ones_create", "randn_create", "rand_create")),
    )
    for creation_module, device_type, names in creation_cases:
        for name in names:
            create = getattr(creation_module, name)
            with pytest.raises(NotImplementedError, match="channels_last|memory format"):
                create(
                    (2, 3, 4, 5),
                    device=device(device_type),
                    memory_format=torch.channels_last,
                )


def test_npu_like_and_new_creation_helpers_reject_channels_last_before_allocation():
    class _FakeTensor:
        shape = (2, 3, 4, 5)
        dtype = torch.float32
        device = device("npu")

    base = _FakeTensor()

    helper_cases = (
        (npu_creation.empty_like_create, (base,)),
        (npu_creation.zeros_like_create, (base,)),
        (npu_creation.ones_like_create, (base,)),
        (npu_creation.randn_like_create, (base,)),
        (npu_creation.rand_like_create, (base,)),
        (npu_creation.new_empty_create, (base, (2, 3, 4, 5))),
        (npu_creation.new_zeros_create, (base, (2, 3, 4, 5))),
        (npu_creation.new_ones_create, (base, (2, 3, 4, 5))),
    )
    for create, args in helper_cases:
        with pytest.raises(NotImplementedError, match="channels_last|memory format"):
            create(*args, memory_format=torch.channels_last)
