"""Tests for shared-memory buffer pool and DataLoader shm integration."""
import numpy as np
import pytest

import candle as torch
from candle.utils.data._shm_buffer import (
    ShmBufferPool,
    serialize_batch_to_slot,
    deserialize_batch_from_slot,
)
from candle.utils.data import DataLoader, Dataset


# -----------------------------------------------------------------------
# ShmBufferPool unit tests
# -----------------------------------------------------------------------

def test_pool_create_and_acquire_release():
    """Pool creates N slots, acquire returns slot ids, release recycles them."""
    pool = ShmBufferPool(num_slots=4, slot_bytes=1024)
    try:
        slots = [pool.acquire() for _ in range(4)]
        assert len(set(slots)) == 4

        assert pool.try_acquire() is None

        pool.release(slots[0])
        reacquired = pool.acquire()
        assert reacquired == slots[0]
    finally:
        pool.close()


def test_pool_write_and_read_numpy():
    """Worker writes numpy array into slot, main reads it zero-copy."""
    pool = ShmBufferPool(num_slots=2, slot_bytes=4096)
    try:
        slot_id = pool.acquire()
        buf = pool.get_buffer(slot_id)

        arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        buf[:arr.nbytes] = arr.tobytes()

        read_buf = pool.get_buffer(slot_id)
        result = np.frombuffer(bytes(read_buf[:arr.nbytes]), dtype=np.float32)
        np.testing.assert_array_equal(result, arr)

        pool.release(slot_id)
    finally:
        pool.close()


def test_pool_close_is_idempotent():
    pool = ShmBufferPool(num_slots=2, slot_bytes=256)
    pool.close()
    pool.close()  # should not raise


# -----------------------------------------------------------------------
# Serialize / deserialize round-trip tests
# -----------------------------------------------------------------------

def test_serialize_deserialize_tensor_batch():
    """Round-trip a stacked tensor through a shm slot."""
    pool = ShmBufferPool(num_slots=2, slot_bytes=65536)
    try:
        slot_id = pool.acquire()
        batch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        meta = serialize_batch_to_slot(batch, pool, slot_id)
        result = deserialize_batch_from_slot(meta, pool, slot_id)
        assert result.shape == (2, 2)
        np.testing.assert_array_almost_equal(result.numpy(), batch.numpy())
        pool.release(slot_id)
    finally:
        pool.close()


def test_serialize_deserialize_tuple_of_tensors():
    """Round-trip a (tensor, tensor) tuple through a shm slot."""
    pool = ShmBufferPool(num_slots=2, slot_bytes=65536)
    try:
        slot_id = pool.acquire()
        t1 = torch.tensor([1.0, 2.0], dtype=torch.float32)
        t2 = torch.tensor([10, 20], dtype=torch.int64)
        batch = (t1, t2)
        meta = serialize_batch_to_slot(batch, pool, slot_id)
        result = deserialize_batch_from_slot(meta, pool, slot_id)
        assert isinstance(result, tuple)
        assert len(result) == 2
        np.testing.assert_array_almost_equal(result[0].numpy(), t1.numpy())
        np.testing.assert_array_almost_equal(result[1].numpy(), t2.numpy())
        pool.release(slot_id)
    finally:
        pool.close()


def test_serialize_deserialize_fallback_pickle():
    """Non-tensor batch falls back to pickle."""
    pool = ShmBufferPool(num_slots=2, slot_bytes=65536)
    try:
        slot_id = pool.acquire()
        batch = {"text": ["hello", "world"], "labels": [0, 1]}
        meta = serialize_batch_to_slot(batch, pool, slot_id)
        result = deserialize_batch_from_slot(meta, pool, slot_id)
        assert result == batch
        pool.release(slot_id)
    finally:
        pool.close()


# -----------------------------------------------------------------------
# DataLoader integration tests with shm
# -----------------------------------------------------------------------

class FloatRangeDataset(Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return torch.tensor([float(idx)], dtype=torch.float32)


def test_dataloader_shm_map_style():
    """DataLoader with num_workers > 0 produces correctly ordered tensor batches."""
    ds = FloatRangeDataset(8)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=2)
    batches = list(loader)
    assert len(batches) == 4
    values = [b.numpy().ravel().tolist() for b in batches]
    assert values == [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]


def test_dataloader_shm_persistent_workers():
    """Persistent workers reuse pool across epochs."""
    ds = FloatRangeDataset(4)
    loader = DataLoader(
        ds, batch_size=2, shuffle=False, num_workers=2, persistent_workers=True
    )
    first = [b.numpy().ravel().tolist() for b in loader]
    second = [b.numpy().ravel().tolist() for b in loader]
    assert first == second == [[0.0, 1.0], [2.0, 3.0]]
