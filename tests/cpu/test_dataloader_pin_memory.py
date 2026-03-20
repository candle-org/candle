"""Tests for pin_memory_thread."""
import queue

import candle as torch
import numpy as np
import pytest

from candle.utils.data._pin_memory_thread import PinMemoryThread, _PinMemoryException
from candle.utils.data import DataLoader, Dataset, IterableDataset


class _BrokenPin:
    """Picklable object whose pin_memory always fails."""

    def __init__(self, value):
        self.value = value

    def pin_memory(self):
        raise RuntimeError(f"boom-{self.value}")



def test_pin_memory_thread_passes_through_batches():
    """Thread takes batches from data_queue and puts them on done_queue."""
    data_queue = queue.Queue()
    done_queue = queue.Queue()

    thread = PinMemoryThread(data_queue, done_queue, device_type="cpu", do_pin=False)
    thread.start()
    try:
        for i in range(3):
            data_queue.put((i, torch.tensor([float(i)])))
        data_queue.put(None)

        results = {}
        for _ in range(3):
            idx, batch = done_queue.get(timeout=5)
            results[idx] = batch

        assert len(results) == 3
        for i in range(3):
            np.testing.assert_almost_equal(results[i].numpy(), [float(i)])
    finally:
        thread.join(timeout=5)



def test_pin_memory_thread_handles_non_tensor_batches():
    """Non-tensor data passes through unchanged."""
    data_queue = queue.Queue()
    done_queue = queue.Queue()

    thread = PinMemoryThread(data_queue, done_queue, device_type="cpu", do_pin=False)
    thread.start()
    try:
        data_queue.put((0, {"text": "hello", "label": 1}))
        data_queue.put(None)

        idx, batch = done_queue.get(timeout=5)
        assert idx == 0
        assert batch == {"text": "hello", "label": 1}
    finally:
        thread.join(timeout=5)



def test_pin_memory_thread_stops_on_sentinel():
    """Thread exits cleanly when it receives None sentinel."""
    data_queue = queue.Queue()
    done_queue = queue.Queue()

    thread = PinMemoryThread(data_queue, done_queue, device_type="cpu", do_pin=False)
    thread.start()
    data_queue.put(None)
    thread.join(timeout=5)
    assert not thread.is_alive()



def test_pin_memory_thread_reports_errors():
    """Thread returns a structured error instead of hanging on pin failure."""
    data_queue = queue.Queue()
    done_queue = queue.Queue()

    thread = PinMemoryThread(data_queue, done_queue, device_type="cpu", do_pin=True)
    thread.start()
    try:
        data_queue.put((7, [_BrokenPin("x")]))
        idx, payload = done_queue.get(timeout=5)
        assert idx == 7
        assert isinstance(payload, _PinMemoryException)
        assert "boom-x" in payload.message
    finally:
        data_queue.put(None)
        thread.join(timeout=5)


class SimpleDataset(Dataset):
    """Dataset returning float tensors for pin_memory integration tests."""

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return torch.tensor([float(idx)], dtype=torch.float32)


class BrokenPinDataset(Dataset):
    """Dataset returning objects that fail during pin_memory."""

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return _BrokenPin(idx)


class SimpleIterableDataset(IterableDataset):
    """Sharded iterable dataset for pin_memory tests."""

    def __iter__(self):
        from candle.utils.data import get_worker_info

        worker_info = get_worker_info()
        if worker_info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = worker_info.id, worker_info.num_workers
        for idx in range(8):
            if idx % num_workers == worker_id:
                yield torch.tensor([float(idx)], dtype=torch.float32)



def test_dataloader_pin_memory_thread_integration():
    """DataLoader with pin_memory=True uses the pin_memory thread."""
    dataset = SimpleDataset(8)
    loader = DataLoader(dataset, batch_size=2, num_workers=2, pin_memory=True)
    batches = [batch.numpy().ravel().tolist() for batch in loader]
    assert batches == [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]



def test_dataloader_pin_memory_thread_propagates_errors():
    """DataLoader raises a RuntimeError when pin_memory thread fails."""
    loader = DataLoader(BrokenPinDataset(), batch_size=1, num_workers=1, pin_memory=True)
    with pytest.raises(RuntimeError, match="pin_memory thread failed"):
        list(loader)



def test_dataloader_pin_memory_iterable_integration():
    """Iterable pin_memory path returns the expected set of sharded batches."""
    loader = DataLoader(
        SimpleIterableDataset(),
        batch_size=2,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    batches = [batch.numpy().ravel().tolist() for batch in loader]
    flat = sorted(value for batch in batches for value in batch)
    assert flat == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    assert len(batches) == 4



def test_dataloader_pin_memory_persistent_workers_integration():
    """Persistent workers work with pin_memory across epochs."""
    dataset = SimpleDataset(8)
    loader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    first = [batch.numpy().ravel().tolist() for batch in loader]
    second = [batch.numpy().ravel().tolist() for batch in loader]
    assert first == second == [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]
