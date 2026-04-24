"""Shared-memory buffer pool for zero-copy DataLoader IPC.

Manages a fixed ring of shared-memory slots. Each slot is a contiguous byte
buffer. Workers acquire a slot, write collated batch data, then signal
completion. The main process reads directly from the shared memory (zero-copy
for numpy-backed tensors) and releases the slot for reuse.
"""

import pickle
import threading
from collections import deque
from multiprocessing import shared_memory

import numpy as np

from ..._cython._tensor_impl import cy_make_tensor_from_storage  # pylint: disable=import-error,no-name-in-module
from ..._dtype import from_numpy_dtype, to_numpy_dtype
from ..._C import typed_storage_from_numpy
from ..._tensor import Tensor


class ShmBufferPool:
    """Fixed-size pool of shared-memory slots.

    Parameters
    ----------
    num_slots : int
        Number of buffer slots (typically prefetch_factor * num_workers).
    slot_bytes : int
        Size of each slot in bytes. Must be >= the largest collated batch.
    """

    def __init__(self, num_slots, slot_bytes):
        self._num_slots = num_slots
        self._slot_bytes = slot_bytes
        self._lock = threading.Lock()
        self._free = deque(range(num_slots))
        self._shms = []
        self._closed = False

        for _ in range(num_slots):
            shm = shared_memory.SharedMemory(create=True, size=slot_bytes)
            self._shms.append(shm)

    @property
    def num_slots(self):
        """Number of buffer slots."""
        return self._num_slots

    @property
    def slot_bytes(self):
        """Size of each slot in bytes."""
        return self._slot_bytes

    def acquire(self):
        """Block until a slot is available, return slot_id."""
        import time
        while True:
            with self._lock:
                if self._free:
                    return self._free.popleft()
            time.sleep(0.0001)

    def try_acquire(self):
        """Non-blocking acquire. Returns slot_id or None."""
        with self._lock:
            if self._free:
                return self._free.popleft()
            return None

    def release(self, slot_id):
        """Return a slot to the free pool."""
        with self._lock:
            self._free.append(slot_id)

    def get_buffer(self, slot_id):
        """Return the raw buffer (memoryview) for a slot."""
        return self._shms[slot_id].buf

    def get_shm_name(self, slot_id):
        """Return the shared memory name for cross-process access."""
        return self._shms[slot_id].name

    def close(self):
        """Unlink and close all shared memory segments."""
        if self._closed:
            return
        self._closed = True
        for shm in self._shms:
            try:
                shm.close()
                shm.unlink()
            except Exception:  # pylint: disable=broad-except
                pass
        self._shms.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:  # pylint: disable=broad-except
            pass


# -----------------------------------------------------------------------
# Batch metadata (small, safe to send through mp.Queue)
# -----------------------------------------------------------------------

class ShmSlotMeta:
    """Describes the layout of batch data within a shared-memory slot.

    For tensor batches, stores shape/dtype/offset so the main process can
    reconstruct numpy views directly from the shm buffer.
    For non-tensor batches, stores pickle byte length.
    """
    __slots__ = ("kind", "tensor_metas", "pickle_nbytes", "total_bytes")

    def __init__(self, kind, tensor_metas=None, pickle_nbytes=0, total_bytes=0):
        self.kind = kind  # "tensor", "tuple_of_tensors", "list_of_tensors", "pickle"
        self.tensor_metas = tensor_metas  # list of (shape, dtype_name, offset, nbytes)
        self.pickle_nbytes = pickle_nbytes
        self.total_bytes = total_bytes


def _is_candle_tensor(obj):
    return hasattr(obj, "_storage") and hasattr(obj, "shape") and hasattr(obj, "stride")


def serialize_batch_to_slot(batch, pool, slot_id):
    """Serialize a collated batch into the shared-memory slot.

    Returns ShmSlotMeta (small, safe to send through mp.Queue).
    """
    buf = pool.get_buffer(slot_id)

    # Fast path: single tensor
    if _is_candle_tensor(batch):
        arr = np.ascontiguousarray(batch._numpy_view())
        nbytes = arr.nbytes
        buf[:nbytes] = arr.tobytes()
        return ShmSlotMeta(
            kind="tensor",
            tensor_metas=[(tuple(arr.shape), str(arr.dtype), 0, nbytes)],
            total_bytes=nbytes,
        )

    # Fast path: tuple/list of tensors
    if isinstance(batch, (tuple, list)) and batch and all(
        _is_candle_tensor(x) for x in batch
    ):
        tensor_metas = []
        offset = 0
        for t in batch:
            arr = np.ascontiguousarray(t._numpy_view())
            nbytes = arr.nbytes
            buf[offset:offset + nbytes] = arr.tobytes()
            tensor_metas.append((tuple(arr.shape), str(arr.dtype), offset, nbytes))
            offset += nbytes
        kind = "tuple_of_tensors" if isinstance(batch, tuple) else "list_of_tensors"
        return ShmSlotMeta(kind=kind, tensor_metas=tensor_metas, total_bytes=offset)

    # Fallback: pickle
    data = pickle.dumps(batch)
    nbytes = len(data)
    buf[:nbytes] = data
    return ShmSlotMeta(kind="pickle", pickle_nbytes=nbytes, total_bytes=nbytes)


def _tensor_from_buffer(buf, shape, dtype_str, offset, nbytes):
    """Reconstruct a candle Tensor from a shared-memory region."""
    arr = np.frombuffer(
        bytes(buf[offset:offset + nbytes]), dtype=np.dtype(dtype_str)
    ).reshape(shape).copy()
    candle_dtype = from_numpy_dtype(np.dtype(dtype_str))
    storage = typed_storage_from_numpy(arr, candle_dtype)
    stride = tuple(np.array(arr.strides) // arr.itemsize) if arr.ndim > 0 else ()
    return cy_make_tensor_from_storage(storage, arr.shape, stride, 0, False)


def deserialize_batch_from_slot(meta, pool, slot_id):
    """Reconstruct a batch from shared-memory slot using metadata."""
    buf = pool.get_buffer(slot_id)

    if meta.kind == "tensor":
        shape, dtype_str, offset, nbytes = meta.tensor_metas[0]
        return _tensor_from_buffer(buf, shape, dtype_str, offset, nbytes)

    if meta.kind in ("tuple_of_tensors", "list_of_tensors"):
        tensors = [
            _tensor_from_buffer(buf, shape, dtype_str, offset, nbytes)
            for shape, dtype_str, offset, nbytes in meta.tensor_metas
        ]
        return tuple(tensors) if meta.kind == "tuple_of_tensors" else tensors

    if meta.kind == "pickle":
        data = bytes(buf[:meta.pickle_nbytes])
        return pickle.loads(data)

    raise ValueError(f"Unknown ShmSlotMeta kind: {meta.kind}")


class WorkerShmView:
    """Lightweight shm accessor for worker processes.

    Workers attach to existing shared memory by name. This class provides
    the same get_buffer() interface as ShmBufferPool.
    """

    def __init__(self, shm_handles, slot_bytes):
        self._shms = shm_handles
        self._slot_bytes = slot_bytes

    def get_buffer(self, slot_id):
        """Return the raw buffer for a slot."""
        return self._shms[slot_id].buf
