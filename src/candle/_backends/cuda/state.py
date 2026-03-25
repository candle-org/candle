import threading
from contextlib import contextmanager

from ..._device import device as Device
from . import runtime as cuda_runtime

_tls = threading.local()
_default_streams = {}
_default_streams_lock = threading.Lock()


class _State:
    def __init__(self):
        self.current_device = 0
        self.current_streams = {}
        self.default_streams = _default_streams


def _state():
    state = getattr(_tls, "state", None)
    if state is None:
        state = _State()
        _tls.state = state
    if getattr(state, "default_streams", None) is not _default_streams:
        state.default_streams = _default_streams
    return state


def current_device():
    return _state().current_device


def set_device(device_id):
    device_id = int(device_id)
    state = _state()
    state.current_device = device_id
    try:
        cuda_runtime.set_device(device_id)
    except OSError:
        pass


@contextmanager
def device_guard(device_id):
    state = _state()
    prev = state.current_device
    set_device(device_id)
    try:
        yield Device("cuda", index=int(device_id))
    finally:
        set_device(prev)


def default_stream(device_id=None):
    dev = current_device() if device_id is None else int(device_id)
    with _default_streams_lock:
        stream = _default_streams.get(dev)
        if stream is None:
            from ...cuda import Stream
            stream = Stream(device=Device("cuda", index=dev))
            _default_streams[dev] = stream
    return stream


def current_stream(device_id=None):
    state = _state()
    dev = current_device() if device_id is None else int(device_id)
    stream = state.current_streams.get(dev)
    if stream is None:
        stream = default_stream(dev)
        state.current_streams[dev] = stream
    return stream


def set_current_stream(stream):
    state = _state()
    state.current_streams[int(stream.device.index or 0)] = stream
