import dataclasses
import threading
from dataclasses import dataclass

import numpy as np

from ..._functional import stack as _stack
from ..._creation import from_numpy as _from_numpy

try:
    from ..._C._dataloader_ops import cy_fast_collate_tensors as _cy_collate
except ImportError:
    _cy_collate = None


@dataclass
class WorkerInfo:
    id: int
    num_workers: int
    seed: int
    dataset: object


_worker_local = threading.local()


def _set_worker_info(info):
    _worker_local.info = info


def _clear_worker_info():
    if hasattr(_worker_local, "info"):
        delattr(_worker_local, "info")


def get_worker_info():
    return getattr(_worker_local, "info", None)


def default_convert(data):
    return data


def default_collate(batch):
    if len(batch) == 0:
        return batch
    first = batch[0]
    # Candle tensors
    if hasattr(first, "device") and hasattr(first, "shape"):
        if _cy_collate is not None:
            return _cy_collate(batch)
        return _stack(batch, dim=0)
    # Numpy arrays -> convert to tensors then stack
    if isinstance(first, np.ndarray):
        return _stack([_from_numpy(b) for b in batch], dim=0)
    # Numpy scalars (np.float64, np.int64, etc.)
    if isinstance(first, np.generic):
        return default_collate([b.item() for b in batch])
    if isinstance(first, float):
        return batch
    if isinstance(first, int):
        return batch
    if isinstance(first, (str, bytes)):
        return batch
    # Named tuples
    if isinstance(first, tuple) and hasattr(first, '_fields'):
        return type(first)(*(default_collate(list(samples)) for samples in zip(*batch)))
    # Dataclasses
    if dataclasses.is_dataclass(first) and not isinstance(first, type):
        return type(first)(
            **{
                f.name: default_collate([getattr(d, f.name) for d in batch])
                for f in dataclasses.fields(first)
            }
        )
    if isinstance(first, tuple):
        transposed = list(zip(*batch))
        return tuple(default_collate(list(samples)) for samples in transposed)
    if isinstance(first, list):
        transposed = list(zip(*batch))
        return [default_collate(list(samples)) for samples in transposed]
    if isinstance(first, dict):
        return {k: default_collate([d[k] for d in batch]) for k in first.keys()}
    return batch


def _pin_memory_batch(batch):
    if hasattr(batch, "pin_memory"):
        return batch.pin_memory()
    if isinstance(batch, tuple):
        return tuple(_pin_memory_batch(x) for x in batch)
    if isinstance(batch, list):
        return [_pin_memory_batch(x) for x in batch]
    if isinstance(batch, dict):
        return {k: _pin_memory_batch(v) for k, v in batch.items()}
    return batch
