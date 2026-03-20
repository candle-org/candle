"""Tests for pin_memory_thread."""
import queue

import candle as torch
import numpy as np

from candle.utils.data._pin_memory_thread import PinMemoryThread
from candle.utils.data import DataLoader, Dataset


def test_pin_memory_thread_passes_through_batches():
    """Thread takes batches from data_queue and puts them on done_queue."""
    data_queue = queue.Queue()
    done_queue = queue.Queue()

    t = PinMemoryThread(data_queue, done_queue, device_type="cpu", do_pin=False)
    t.start()
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
        t.join(timeout=5)


def test_pin_memory_thread_handles_non_tensor_batches():
    """Non-tensor data passes through unchanged."""
    data_queue = queue.Queue()
    done_queue = queue.Queue()

    t = PinMemoryThread(data_queue, done_queue, device_type="cpu", do_pin=False)
    t.start()
    try:
        data_queue.put((0, {"text": "hello", "label": 1}))
        data_queue.put(None)

        idx, batch = done_queue.get(timeout=5)
        assert batch == {"text": "hello", "label": 1}
    finally:
        t.join(timeout=5)


def test_pin_memory_thread_stops_on_sentinel():
    """Thread exits cleanly when it receives None sentinel."""
    data_queue = queue.Queue()
    done_queue = queue.Queue()

    t = PinMemoryThread(data_queue, done_queue, device_type="cpu", do_pin=False)
    t.start()
    data_queue.put(None)
    t.join(timeout=5)
    assert not t.is_alive()


class SimpleDataset(Dataset):
    """Dataset returning float tensors for pin_memory integration tests."""
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return torch.tensor([float(idx)], dtype=torch.float32)


def test_dataloader_pin_memory_thread_integration():
    """DataLoader with pin_memory=True uses the pin_memory thread."""
    ds = SimpleDataset(8)
    loader = DataLoader(ds, batch_size=2, num_workers=2, pin_memory=True)
    batches = [b.numpy().ravel().tolist() for b in loader]
    assert batches == [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]
