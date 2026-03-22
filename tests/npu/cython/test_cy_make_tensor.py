"""Tests for cy_make_npu_tensor Cython function."""
import pytest


def test_cy_make_npu_tensor_equivalence(npu_device):
    """Tensor from cy_make_npu_tensor must match one from normal constructor."""
    import candle as torch
    from candle._cython._storage import cy_make_npu_tensor  # pylint: disable=import-error,no-name-in-module
    from candle._storage import npu_typed_storage_from_ptr
    import candle

    dtype = candle.float32
    device = torch.device("npu:0")
    shape = (4, 4)
    stride = (4, 1)
    n = 16

    from candle._backends.npu import runtime as rt, allocator as alloc_mod
    runtime = rt.get_runtime(0)
    alloc = alloc_mod.get_allocator(0)
    torch.npu.synchronize()

    # Allocate one pointer and wrap it in a reference tensor via npu_typed_storage_from_ptr.
    # cy_make_npu_tensor gets a SEPARATE allocation so we avoid double-free.
    ptr_ref = alloc.malloc(n * 4, stream=runtime.stream)
    ptr_fast = alloc.malloc(n * 4, stream=runtime.stream)
    try:
        storage_normal = npu_typed_storage_from_ptr(ptr_ref, n, dtype, device=device)
        t_normal = torch.Tensor(storage_normal, shape, stride)
        t_fast = cy_make_npu_tensor(ptr_fast, n, dtype, device, shape, stride)
        assert t_fast.shape == t_normal.shape
        assert tuple(t_fast.stride) == tuple(t_normal.stride)
        assert t_fast.dtype == t_normal.dtype
        assert t_fast.device.type == t_normal.device.type
        # Verify cy_make_npu_tensor wraps exactly the pointer we passed.
        assert t_fast._storage.data_ptr() == ptr_fast
        assert t_fast.requires_grad == t_normal.requires_grad
        assert t_fast.grad_fn is None
        assert t_fast.grad is None
    finally:
        # Let storages handle freeing ptr_ref and ptr_fast via __dealloc__.
        del t_normal, t_fast, storage_normal
        torch.npu.synchronize()


def test_add_correctness_with_cy_make(npu_device):
    """torch.add still produces correct results after replacing tensor construction."""
    import candle as torch
    import numpy as np
    a_np = np.ones((4, 4), dtype=np.float32)
    b_np = np.ones((4, 4), dtype=np.float32) * 2.0
    a = torch.tensor(a_np, device=npu_device)
    b = torch.tensor(b_np, device=npu_device)
    c = torch.add(a, b)
    torch.npu.synchronize()
    result = c.cpu().numpy()
    np.testing.assert_allclose(result, np.full((4, 4), 3.0, dtype=np.float32))


def test_add_result_has_no_grad(npu_device):
    """Result of add (no grad) must have grad_fn=None and requires_grad=False."""
    import candle as torch
    a = torch.randn(4, 4, device=npu_device)
    b = torch.randn(4, 4, device=npu_device)
    c = torch.add(a, b)
    torch.npu.synchronize()
    assert c.grad_fn is None
    assert not c.requires_grad
