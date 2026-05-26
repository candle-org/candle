import re
from pathlib import Path

import candle  # noqa: F401

from candle._dispatch.keys import DispatchKey
from candle._dispatch.registry import registry


def _autograd_backend_source() -> str:
    pkg_dir = Path(candle.__file__).resolve().parent
    return (pkg_dir / "_backends" / "autograd.py").read_text(encoding="utf-8")


def test_backward_formulas_do_not_access_storage_payload_directly():
    src = _autograd_backend_source()
    assert "storage().data" not in src
    assert "storage()._data" not in src


def test_target_autograd_npu_kernels_are_registered_and_not_cpu_only_config():
    src = _autograd_backend_source()
    target_ops = (
        "relu",
        "relu_",
        "abs",
        "neg",
        "silu",
        "leaky_relu",
        "elu",
        "mish",
        "prelu",
    )

    for op in target_ops:
        cpu_only_pattern = (
            rf'_autograd_(?:unary|unary_args|inplace)\("{re.escape(op)}"[^\n]*cpu_only=True'
        )
        assert re.search(cpu_only_pattern, src) is None, (
            f"{op} should not use cpu_only=True in autograd wrapper config"
        )

        entry = registry.get(f"aten::{op}")
        assert DispatchKey.AutogradNPU in entry.kernels, (
            f"missing AutogradNPU registration for {op}"
        )


def test_autograd_post_ops_keep_npu_specific_autograd_overrides():
    src = _autograd_backend_source()
    dispatcher = (Path(candle.__file__).resolve().parent / "_C" / "_dispatcher_core.pyx").read_text(
        encoding="utf-8"
    )

    assert "def _npu_autograd_conv" in src
    assert "not _has_device_autograd_override(entry, m)" in dispatcher

    for op in ("conv1d", "conv2d", "conv3d", "conv_transpose1d", "conv_transpose2d", "conv_transpose3d"):
        entry = registry.get(f"aten::{op}")
        assert entry.autograd_post is not None
        assert DispatchKey.AutogradNPU in entry.kernels
        assert entry.kernels[DispatchKey.AutogradNPU] is not entry.kernels[DispatchKey.Autograd]
