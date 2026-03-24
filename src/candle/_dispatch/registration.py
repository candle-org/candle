from .keys import DispatchKey
from .registry import registry


_FORWARD_KEY_BY_DEVICE = {
    "cpu": DispatchKey.CPU,
    "npu": DispatchKey.NPU,
    "meta": DispatchKey.Meta,
    "cuda": DispatchKey.CUDA,
    "mps": DispatchKey.PrivateUse2,
}

_AUTOGRAD_KEY_BY_DEVICE = {
    "cpu": DispatchKey.AutogradCPU,
    "npu": DispatchKey.AutogradNPU,
    "meta": DispatchKey.AutogradMeta,
    "cuda": DispatchKey.AutogradCUDA,
    "mps": DispatchKey.PrivateUse3,
}


def _normalize_device_label(device):
    if device is None:
        raise ValueError("device label cannot be None")
    return str(device).lower()


def register_forward_kernels(name, **kernels):
    """Register forward kernels by device label in one call.

    Supported labels: cpu, npu, meta, cuda.
    cuda maps to the explicit CUDA dispatch key.
    """
    for device, fn in kernels.items():
        if fn is None:
            continue
        label = _normalize_device_label(device)
        key = _FORWARD_KEY_BY_DEVICE.get(label)
        if key is None:
            raise ValueError(f"unsupported forward registration device: {device}")
        registry.register_kernel(name, key, fn)


def register_autograd_kernels(name, *, default=None, **kernels):
    """Register autograd kernels by device label in one call.

    Supported labels: cpu, npu, meta, cuda.
    cuda maps to the explicit AutogradCUDA dispatch key.
    """
    if default is not None:
        registry.register_kernel(name, DispatchKey.Autograd, default)

    for device, fn in kernels.items():
        if fn is None:
            continue
        label = _normalize_device_label(device)
        key = _AUTOGRAD_KEY_BY_DEVICE.get(label)
        if key is None:
            raise ValueError(f"unsupported autograd registration device: {device}")
        registry.register_kernel(name, key, fn)


def register_autograd_post_kernels(name, fn):
    """Register an autograd post-processing kernel for single-pass dispatch."""
    entry = registry._entry(name)
    entry.autograd_post = fn


__all__ = ["register_forward_kernels", "register_autograd_kernels", "register_autograd_post_kernels"]
