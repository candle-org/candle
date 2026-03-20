"""Pin-memory daemon thread for DataLoader.

Runs in the main process. Reads (idx, batch) tuples from data_queue,
applies pin_memory() to tensor data, and puts (idx, pinned_batch)
on done_queue. Exits when it reads None sentinel from data_queue.
"""

import threading


class _PinMemoryException:
    """Pin-memory failure marker sent back through the done queue."""

    def __init__(self, message):
        self.message = message


def _pin_batch(batch):
    """Recursively pin_memory on tensor-like objects."""
    if hasattr(batch, "pin_memory"):
        return batch.pin_memory()
    if isinstance(batch, tuple):
        return tuple(_pin_batch(x) for x in batch)
    if isinstance(batch, list):
        return [_pin_batch(x) for x in batch]
    if isinstance(batch, dict):
        return {k: _pin_batch(v) for k, v in batch.items()}
    return batch


class PinMemoryThread(threading.Thread):
    """Daemon thread that pins batches and optionally transfers to device.

    Parameters
    ----------
    data_queue : queue.Queue
        Input: (idx, batch) tuples. None sentinel to stop.
    done_queue : queue.Queue
        Output: (idx, pinned_batch) tuples or (idx, _PinMemoryException).
    device_type : str
        Target device type ("cpu", "npu", "cuda").
    do_pin : bool
        Whether to actually call pin_memory(). False = passthrough.
    """

    def __init__(self, data_queue, done_queue, device_type="cpu", do_pin=True):
        super().__init__(daemon=True)
        self._data_queue = data_queue
        self._done_queue = done_queue
        self._device_type = device_type
        self._do_pin = do_pin

    def run(self):
        while True:
            item = self._data_queue.get()
            if item is None:
                return  # sentinel: stop thread
            idx, batch = item
            try:
                if self._do_pin:
                    batch = _pin_batch(batch)
            except Exception as exc:  # pylint: disable=broad-except
                self._done_queue.put((idx, _PinMemoryException(repr(exc))))
                return
            self._done_queue.put((idx, batch))
