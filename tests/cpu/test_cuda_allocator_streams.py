import pytest

import candle as torch


cuda_allocator = pytest.importorskip(
    "candle._backends.cuda.allocator",
    reason="CUDA allocator module not present in current worktree",
)


@pytest.fixture(autouse=True)
def _reset_cuda_allocator_state():
    saved = dict(getattr(cuda_allocator, "_ALLOCATORS", {}))
    cuda_allocator._ALLOCATORS.clear()
    yield
    cuda_allocator._ALLOCATORS.clear()
    cuda_allocator._ALLOCATORS.update(saved)



def test_cuda_tensor_record_stream_uses_cuda_allocator(monkeypatch):
    seen = {}

    class DummyAlloc:
        def record_stream(self, ptr, stream):
            seen["call"] = (ptr, stream)

    monkeypatch.setattr(cuda_allocator, "get_allocator", lambda device_id=0: DummyAlloc())

    class DummyStorage:
        def __init__(self):
            self.device = torch.device("cuda", 0)

        def data_ptr(self):
            return 1234

    x = torch.tensor([1.0], device=torch.device("cuda", 0))
    x._storage._untyped = DummyStorage()

    class DummyStream:
        stream = 777

    x.record_stream(DummyStream())
    assert seen["call"] == (1234, 777)



def test_cuda_allocator_free_defers_until_event_completes(monkeypatch):
    alloc = cuda_allocator.CudaAllocator(device_id=0)
    ptr = alloc.allocate(64, stream=11)
    alloc.record_stream(ptr, 22)
    alloc.free(ptr)

    assert ptr not in alloc._active
    assert len(alloc._pending) == 1
    block = alloc._pending[0]
    assert block.ptr == ptr
    assert block.event is not None
    assert block.stream == 22



def test_cuda_allocator_synchronize_reclaims_completed_pending_blocks(monkeypatch):
    alloc = cuda_allocator.CudaAllocator(device_id=0)
    ptr = alloc.allocate(64, stream=11)
    alloc.free(ptr)
    assert len(alloc._pending) == 1

    alloc.synchronize()

    assert len(alloc._pending) == 0
    assert ptr not in alloc._active
    assert any(block.ptr == ptr for block in alloc._cached)
