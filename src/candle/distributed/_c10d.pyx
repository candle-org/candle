# cython: language_level=3, boundscheck=False, wraparound=False
"""
Cython implementation of core distributed types for candle.distributed.

Ports the exact logic from:
  _reduce_op.py  -> RedOpType, ReduceOp, reduce_op
  _work.py       -> Work
  _store.py      -> _recvall, _recv_bytes, _send_bytes, _StoreServer, TCPStore
  _backend.py    -> Store (base), PrefixStore
  _process_group.py -> ProcessGroup (base class only)

Plus new: HashStore (dict + lock, thread-safe), Options structs.
"""

import threading
import socket
import struct
import time
from enum import IntEnum


# ---------------------------------------------------------------------------
# ReduceOp section
# ---------------------------------------------------------------------------

class RedOpType(IntEnum):
    SUM = 0
    PRODUCT = 1
    MAX = 2
    MIN = 3
    BAND = 4
    BOR = 5
    BXOR = 6
    AVG = 7
    PREMUL_SUM = 8
    UNUSED = 9


cdef class ReduceOp:
    # cdef fields declared in _c10d.pxd

    SUM = RedOpType.SUM
    PRODUCT = RedOpType.PRODUCT
    MAX = RedOpType.MAX
    MIN = RedOpType.MIN
    BAND = RedOpType.BAND
    BOR = RedOpType.BOR
    BXOR = RedOpType.BXOR
    AVG = RedOpType.AVG
    PREMUL_SUM = RedOpType.PREMUL_SUM
    UNUSED = RedOpType.UNUSED

    RedOpType = RedOpType

    def __init__(self, op=None):
        if op is not None:
            self._op = int(RedOpType(op))
        else:
            self._op = int(RedOpType.SUM)

    def __int__(self):
        return self._op

    def __eq__(self, other):
        if isinstance(other, ReduceOp):
            return self._op == (<ReduceOp>other)._op
        if isinstance(other, (int, RedOpType)):
            return self._op == int(other)
        return NotImplemented

    def __hash__(self):
        return hash(self._op)

    def __repr__(self):
        return f"<ReduceOp.{RedOpType(self._op).name}: {self._op}>"


# Deprecated alias matching PyTorch
class reduce_op:
    SUM = ReduceOp.SUM
    PRODUCT = ReduceOp.PRODUCT
    MAX = ReduceOp.MAX
    MIN = ReduceOp.MIN


# ---------------------------------------------------------------------------
# Options structs
# ---------------------------------------------------------------------------

cdef class AllreduceOptions:
    # cdef fields declared in _c10d.pxd

    def __init__(self, reduceOp=int(RedOpType.SUM), double timeout=300.0,
                 bint asyncOp=False):
        self.reduceOp = reduceOp
        self.timeout = timeout
        self.asyncOp = asyncOp


cdef class BroadcastOptions:
    # cdef fields declared in _c10d.pxd

    def __init__(self, int rootRank=0, int rootTensor=0, double timeout=300.0,
                 bint asyncOp=False):
        self.rootRank = rootRank
        self.rootTensor = rootTensor
        self.timeout = timeout
        self.asyncOp = asyncOp


cdef class ReduceOptions:
    # cdef fields declared in _c10d.pxd

    def __init__(self, reduceOp=int(RedOpType.SUM), int rootRank=0,
                 int rootTensor=0, double timeout=300.0, bint asyncOp=False):
        self.reduceOp = reduceOp
        self.rootRank = rootRank
        self.rootTensor = rootTensor
        self.timeout = timeout
        self.asyncOp = asyncOp


cdef class AllgatherOptions:
    # cdef fields declared in _c10d.pxd

    def __init__(self, double timeout=300.0, bint asyncOp=False):
        self.timeout = timeout
        self.asyncOp = asyncOp


cdef class GatherOptions:
    # cdef fields declared in _c10d.pxd

    def __init__(self, int rootRank=0, double timeout=300.0,
                 bint asyncOp=False):
        self.rootRank = rootRank
        self.timeout = timeout
        self.asyncOp = asyncOp


cdef class ScatterOptions:
    # cdef fields declared in _c10d.pxd

    def __init__(self, int rootRank=0, double timeout=300.0,
                 bint asyncOp=False):
        self.rootRank = rootRank
        self.timeout = timeout
        self.asyncOp = asyncOp


cdef class ReduceScatterOptions:
    # cdef fields declared in _c10d.pxd

    def __init__(self, reduceOp=int(RedOpType.SUM), double timeout=300.0,
                 bint asyncOp=False):
        self.reduceOp = reduceOp
        self.timeout = timeout
        self.asyncOp = asyncOp


cdef class BarrierOptions:
    # cdef fields declared in _c10d.pxd

    def __init__(self, double timeout=300.0, bint asyncOp=False,
                 device_ids=None):
        self.timeout = timeout
        self.asyncOp = asyncOp
        self.device_ids = device_ids if device_ids is not None else []


cdef class AllToAllOptions:
    # cdef fields declared in _c10d.pxd

    def __init__(self, double timeout=300.0, bint asyncOp=False):
        self.timeout = timeout
        self.asyncOp = asyncOp


# ---------------------------------------------------------------------------
# Work class
# ---------------------------------------------------------------------------

cdef class Work:
    # cdef fields declared in _c10d.pxd

    def __init__(self, stream=None, device_id=None, int source_rank=-1):
        self._completed = False
        self._stream = stream
        self._device_id = device_id
        self._exception = None
        self._source_rank = source_rank
        self._on_wait = None

    cpdef bint wait(self, timeout=None):
        if not self._completed and self._stream is not None:
            try:
                from .._backends.npu import runtime as npu_runtime
                dev = self._device_id if self._device_id is not None else 0
                npu_runtime.get_runtime(dev).synchronize_stream(self._stream)
            except Exception as e:
                self._exception = e
                raise
        if self._on_wait is not None:
            try:
                self._on_wait()
            except Exception as e:
                self._exception = e
                raise
            finally:
                self._on_wait = None
        self._completed = True
        return True

    cpdef bint is_completed(self):
        return self._completed

    cpdef bint is_success(self):
        return self._completed and self._exception is None

    def exception(self):
        return self._exception

    def source_rank(self):
        return self._source_rank

    def result(self):
        return []

    def synchronize(self):
        self.wait()

    def get_future(self):
        from ..futures import Future

        fut = Future()
        try:
            self.wait()
        except Exception as exc:
            fut.set_exception(exc)
            return fut

        fut.set_result(self.result())
        return fut


# ---------------------------------------------------------------------------
# Store hierarchy
# ---------------------------------------------------------------------------

# Wire protocol constants
cdef int _CMD_SET = 0
cdef int _CMD_GET = 1
cdef int _CMD_WAIT = 2

cdef int _RESP_OK = 0
cdef int _RESP_VALUE = 1

cdef double _DEFAULT_TIMEOUT = 300.0


# Internal helpers — cdef for C-speed, not callable from Python

cdef bytes _recvall(object sock, int n):
    cdef bytearray buf = bytearray()
    cdef bytes chunk
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


cdef bytes _recv_bytes(object sock):
    cdef bytes raw = _recvall(sock, 4)
    if raw is None:
        raise ConnectionError("connection closed")
    cdef int length = struct.unpack("!I", raw)[0]
    return _recvall(sock, length)


cdef void _send_bytes(object sock, bytes data):
    sock.sendall(struct.pack("!I", len(data)))
    sock.sendall(data)


# --- Store base class ---

cdef class Store:
    """Base class for distributed stores (API compatibility)."""

    cpdef set(self, str key, bytes value):
        raise NotImplementedError

    cpdef bytes get(self, str key):
        raise NotImplementedError

    cpdef wait(self, list keys, timeout=None):
        raise NotImplementedError

    cpdef close(self):
        pass


# --- _StoreServer (NOT in .pxd, keep cdef fields here) ---

cdef class _StoreServer:
    cdef dict _data
    cdef object _lock
    cdef object _cond
    cdef int _world_size
    cdef double _timeout
    cdef bint _closed
    cdef object _server
    cdef object _thread

    def __init__(self, int port, int world_size, double timeout=_DEFAULT_TIMEOUT):
        self._data = {}
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._world_size = world_size
        self._timeout = timeout
        self._closed = False
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("", port))
        self._server.listen(world_size * 4)
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def _accept_loop(self):
        while not self._closed:
            try:
                conn, _ = self._server.accept()
            except OSError:
                break
            t = threading.Thread(target=self._handle, args=(conn,),
                                 daemon=True)
            t.start()

    def _handle(self, conn):
        cdef bytes hdr
        cdef int cmd
        cdef str key
        cdef bytes value
        cdef bytes raw
        cdef int n
        cdef int offset
        cdef int klen
        cdef list keys
        cdef double deadline
        cdef double remaining
        cdef str k
        cdef bint all_present
        try:
            while not self._closed:
                hdr = _recvall(conn, 1)
                if hdr is None:
                    break
                cmd = hdr[0]
                if cmd == _CMD_SET:
                    key = _recv_bytes(conn).decode("utf-8")
                    value = _recv_bytes(conn)
                    with self._cond:
                        self._data[key] = value
                        self._cond.notify_all()
                    conn.sendall(bytes([_RESP_OK]))
                elif cmd == _CMD_GET:
                    key = _recv_bytes(conn).decode("utf-8")
                    deadline = time.monotonic() + self._timeout
                    with self._cond:
                        while key not in self._data:
                            remaining = deadline - time.monotonic()
                            if remaining <= 0 or self._closed:
                                raise TimeoutError(
                                    f"TCPStore server: GET '{key}' timed out")
                            self._cond.wait(timeout=min(remaining, 1.0))
                        value = self._data[key]
                    conn.sendall(bytes([_RESP_VALUE]))
                    _send_bytes(conn, value)
                elif cmd == _CMD_WAIT:
                    raw = _recv_bytes(conn)
                    n = struct.unpack("!I", raw[:4])[0]
                    keys = []
                    offset = 4
                    for _ in range(n):
                        klen = struct.unpack("!I", raw[offset:offset + 4])[0]
                        offset += 4
                        keys.append(raw[offset:offset + klen].decode("utf-8"))
                        offset += klen
                    deadline = time.monotonic() + self._timeout
                    with self._cond:
                        while True:
                            all_present = True
                            for k in keys:
                                if k not in self._data:
                                    all_present = False
                                    break
                            if all_present:
                                break
                            remaining = deadline - time.monotonic()
                            if remaining <= 0 or self._closed:
                                raise TimeoutError(
                                    "TCPStore server: WAIT timed out")
                            self._cond.wait(timeout=min(remaining, 1.0))
                    conn.sendall(bytes([_RESP_OK]))
        except (ConnectionError, OSError, TimeoutError):
            pass
        finally:
            conn.close()

    def close(self):
        self._closed = True
        with self._cond:
            self._cond.notify_all()
        try:
            self._server.close()
        except OSError:
            pass


# --- TCPStore ---

cdef class TCPStore(Store):
    # cdef fields declared in _c10d.pxd

    def __init__(self, str host, int port, int world_size, bint is_master,
                 double timeout=_DEFAULT_TIMEOUT):
        self._host = host
        self._port = port
        self._world_size = world_size
        self._timeout = timeout
        self._server_inst = None
        self._lock = threading.Lock()
        if is_master:
            self._server_inst = _StoreServer(port, world_size, timeout=timeout)
            time.sleep(0.1)
        self._sock = self._connect(host, port, timeout)

    cdef object _connect(self, str host, int port, double timeout):
        cdef double deadline = time.monotonic() + timeout
        cdef object sock
        while True:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(max(deadline - time.monotonic(), 1.0))
                sock.connect((host, port))
                sock.settimeout(self._timeout)
                return sock
            except (ConnectionRefusedError, OSError):
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"TCPStore: could not connect to {host}:{port} "
                        f"within {timeout}s"
                    )
                time.sleep(0.1)

    cpdef set(self, str key, bytes value):
        if isinstance(value, str):
            value = (<str>value).encode("utf-8")
        with self._lock:
            self._sock.sendall(bytes([_CMD_SET]))
            _send_bytes(self._sock, key.encode("utf-8"))
            _send_bytes(self._sock, value)
            resp = _recvall(self._sock, 1)
        if resp is None or resp[0] != _RESP_OK:
            raise RuntimeError("TCPStore.set failed")

    cpdef bytes get(self, str key):
        with self._lock:
            self._sock.sendall(bytes([_CMD_GET]))
            _send_bytes(self._sock, key.encode("utf-8"))
            resp = _recvall(self._sock, 1)
            if resp is None or resp[0] != _RESP_VALUE:
                raise RuntimeError("TCPStore.get failed")
            return _recv_bytes(self._sock)

    cpdef wait(self, list keys, timeout=None):
        cdef bytes buf = struct.pack("!I", len(keys))
        cdef bytes kb
        cdef object old_timeout
        cdef bytes resp
        for k in keys:
            kb = k.encode("utf-8")
            buf += struct.pack("!I", len(kb)) + kb
        with self._lock:
            self._sock.sendall(bytes([_CMD_WAIT]))
            _send_bytes(self._sock, buf)
            old_timeout = self._sock.gettimeout()
            if timeout is not None:
                self._sock.settimeout(timeout)
            try:
                resp = _recvall(self._sock, 1)
                if resp is None or resp[0] != _RESP_OK:
                    raise RuntimeError("TCPStore.wait failed")
            finally:
                self._sock.settimeout(old_timeout)

    cpdef close(self):
        try:
            self._sock.close()
        except OSError:
            pass
        if self._server_inst is not None:
            self._server_inst.close()


# --- PrefixStore ---

cdef class PrefixStore(Store):
    # cdef fields declared in _c10d.pxd

    def __init__(self, str prefix, object store):
        self._prefix = prefix
        self._store = store

    cpdef set(self, str key, bytes value):
        self._store.set(f"{self._prefix}/{key}", value)

    cpdef bytes get(self, str key):
        return self._store.get(f"{self._prefix}/{key}")

    cpdef wait(self, list keys, timeout=None):
        self._store.wait([f"{self._prefix}/{k}" for k in keys], timeout)

    cpdef close(self):
        self._store.close()


# --- HashStore ---

cdef class HashStore(Store):
    # cdef fields declared in _c10d.pxd

    def __init__(self):
        self._data = {}
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    cpdef set(self, str key, bytes value):
        with self._cond:
            self._data[key] = value
            self._cond.notify_all()

    cpdef bytes get(self, str key):
        with self._cond:
            while key not in self._data:
                self._cond.wait(timeout=1.0)
            return self._data[key]

    cpdef wait(self, list keys, timeout=None):
        cdef double deadline
        cdef double remaining
        cdef str k
        cdef bint all_present
        if timeout is not None:
            deadline = time.monotonic() + <double>timeout
        else:
            deadline = time.monotonic() + _DEFAULT_TIMEOUT
        with self._cond:
            while True:
                all_present = True
                for k in keys:
                    if k not in self._data:
                        all_present = False
                        break
                if all_present:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"HashStore: WAIT timed out for keys {keys}")
                self._cond.wait(timeout=min(remaining, 1.0))

    cpdef close(self):
        pass


# ---------------------------------------------------------------------------
# ProcessGroup base class
# ---------------------------------------------------------------------------

cdef class ProcessGroup:
    # cdef fields declared in _c10d.pxd

    def __init__(self, int rank, int size):
        self._rank = rank
        self._size = size
        self._group_name = ""
        self._group_desc = ""
        self._ranks = None
        self._bound_device_id = None

    cpdef int rank(self):
        return self._rank

    cpdef int size(self):
        return self._size

    cpdef str name(self):
        return self._group_name

    @property
    def group_name(self):
        return self._group_name

    @property
    def group_desc(self):
        return self._group_desc

    @property
    def bound_device_id(self):
        return self._bound_device_id
