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
from datetime import timedelta
from enum import IntEnum


cdef double _to_seconds(object timeout):
    """Convert timeout to seconds. Accepts float, int, or timedelta."""
    if timeout is None:
        return _DEFAULT_TIMEOUT
    if isinstance(timeout, timedelta):
        return (<object>timeout).total_seconds()
    return <double>timeout


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

    def __getstate__(self):
        return self._op

    def __setstate__(self, state):
        self._op = int(state)

    def __copy__(self):
        return ReduceOp(self._op)

    def __deepcopy__(self, memo):
        return ReduceOp(self._op)


# Deprecated alias matching PyTorch
class reduce_op:
    SUM = ReduceOp.SUM
    PRODUCT = ReduceOp.PRODUCT
    MAX = ReduceOp.MAX
    MIN = ReduceOp.MIN


# ---------------------------------------------------------------------------
# BackendType enum
# ---------------------------------------------------------------------------

class BackendType(IntEnum):
    UNDEFINED = 0
    GLOO = 1
    NCCL = 2
    XCCL = 3
    UCC = 4
    MPI = 5
    CUSTOM = 6


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


cdef class AllreduceCoalescedOptions:
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
        self._result = None

    cpdef bint wait(self, timeout=None):
        if timeout is not None:
            _to_seconds(timeout)  # validate timedelta accepted
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
        if self._result is not None:
            return self._result
        return []

    def synchronize(self):
        self.wait()

    def block_current_stream(self):
        """Block the currently active stream on this operation.

        For GPU/NPU collectives this is equivalent to synchronize().
        """
        if self._stream is not None:
            try:
                from .._backends.npu import runtime as npu_runtime
                dev = self._device_id if self._device_id is not None else 0
                npu_runtime.get_runtime(dev).synchronize_stream(self._stream)
            except ImportError:
                pass

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
cdef int _CMD_ADD = 3
cdef int _CMD_DELETE = 4
cdef int _CMD_NUM_KEYS = 5
cdef int _CMD_CHECK = 6
cdef int _CMD_COMPARE_SET = 7

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

    cpdef int add(self, str key, int amount):
        raise NotImplementedError

    cpdef bint delete_key(self, str key):
        raise NotImplementedError

    cpdef int num_keys(self):
        raise NotImplementedError

    cpdef bint check(self, list keys):
        raise NotImplementedError

    cpdef bytes compare_set(self, str key, str expected, str desired):
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
                elif cmd == _CMD_ADD:
                    key = _recv_bytes(conn).decode("utf-8")
                    amount = struct.unpack("!i", _recvall(conn, 4))[0]
                    with self._cond:
                        old = 0
                        if key in self._data:
                            old = struct.unpack("!i", self._data[key])[0]
                        new_val = old + amount
                        self._data[key] = struct.pack("!i", new_val)
                        self._cond.notify_all()
                    conn.sendall(struct.pack("!i", new_val))
                elif cmd == _CMD_DELETE:
                    key = _recv_bytes(conn).decode("utf-8")
                    with self._cond:
                        deleted = key in self._data
                        if deleted:
                            del self._data[key]
                    conn.sendall(struct.pack("!?", deleted))
                elif cmd == _CMD_NUM_KEYS:
                    with self._cond:
                        count = len(self._data)
                    conn.sendall(struct.pack("!I", count))
                elif cmd == _CMD_CHECK:
                    raw = _recv_bytes(conn)
                    n = struct.unpack("!I", raw[:4])[0]
                    keys = []
                    offset = 4
                    for _ in range(n):
                        klen = struct.unpack("!I", raw[offset:offset + 4])[0]
                        offset += 4
                        keys.append(raw[offset:offset + klen].decode("utf-8"))
                        offset += klen
                    with self._cond:
                        found = True
                        for k in keys:
                            if k not in self._data:
                                found = False
                                break
                    conn.sendall(struct.pack("!?", found))
                elif cmd == _CMD_COMPARE_SET:
                    key = _recv_bytes(conn).decode("utf-8")
                    expected = _recv_bytes(conn)
                    desired = _recv_bytes(conn)
                    with self._cond:
                        current = self._data.get(key, b"")
                        if current == expected:
                            self._data[key] = desired
                            self._cond.notify_all()
                        result_val = self._data.get(key, b"")
                    conn.sendall(bytes([_RESP_VALUE]))
                    _send_bytes(conn, result_val)
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
                 timeout=_DEFAULT_TIMEOUT):
        self._host = host
        self._port = port
        self._world_size = world_size
        self._timeout = _to_seconds(timeout)
        self._server_inst = None
        self._lock = threading.Lock()
        if is_master:
            self._server_inst = _StoreServer(port, world_size, timeout=self._timeout)
            time.sleep(0.1)
        self._sock = self._connect(host, port, self._timeout)

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
        cdef double t
        for k in keys:
            kb = k.encode("utf-8")
            buf += struct.pack("!I", len(kb)) + kb
        with self._lock:
            self._sock.sendall(bytes([_CMD_WAIT]))
            _send_bytes(self._sock, buf)
            old_timeout = self._sock.gettimeout()
            if timeout is not None:
                t = _to_seconds(timeout)
                self._sock.settimeout(t)
            try:
                resp = _recvall(self._sock, 1)
                if resp is None or resp[0] != _RESP_OK:
                    raise RuntimeError("TCPStore.wait failed")
            finally:
                self._sock.settimeout(old_timeout)

    cpdef int add(self, str key, int amount):
        cdef bytes resp
        with self._lock:
            self._sock.sendall(bytes([_CMD_ADD]))
            _send_bytes(self._sock, key.encode("utf-8"))
            self._sock.sendall(struct.pack("!i", amount))
            resp = _recvall(self._sock, 4)
        if resp is None:
            raise RuntimeError("TCPStore.add failed")
        return struct.unpack("!i", resp)[0]

    cpdef bint delete_key(self, str key):
        cdef bytes resp
        with self._lock:
            self._sock.sendall(bytes([_CMD_DELETE]))
            _send_bytes(self._sock, key.encode("utf-8"))
            resp = _recvall(self._sock, 1)
        if resp is None:
            raise RuntimeError("TCPStore.delete_key failed")
        return struct.unpack("!?", resp)[0]

    cpdef int num_keys(self):
        cdef bytes resp
        with self._lock:
            self._sock.sendall(bytes([_CMD_NUM_KEYS]))
            resp = _recvall(self._sock, 4)
        if resp is None:
            raise RuntimeError("TCPStore.num_keys failed")
        return struct.unpack("!I", resp)[0]

    cpdef bint check(self, list keys):
        cdef bytes buf = struct.pack("!I", len(keys))
        cdef bytes kb
        cdef bytes resp
        for k in keys:
            kb = k.encode("utf-8")
            buf += struct.pack("!I", len(kb)) + kb
        with self._lock:
            self._sock.sendall(bytes([_CMD_CHECK]))
            _send_bytes(self._sock, buf)
            resp = _recvall(self._sock, 1)
        if resp is None:
            raise RuntimeError("TCPStore.check failed")
        return struct.unpack("!?", resp)[0]

    cpdef bytes compare_set(self, str key, str expected, str desired):
        cdef bytes resp
        with self._lock:
            self._sock.sendall(bytes([_CMD_COMPARE_SET]))
            _send_bytes(self._sock, key.encode("utf-8"))
            _send_bytes(self._sock, expected.encode("utf-8"))
            _send_bytes(self._sock, desired.encode("utf-8"))
            resp = _recvall(self._sock, 1)
            if resp is None or resp[0] != _RESP_VALUE:
                raise RuntimeError("TCPStore.compare_set failed")
            return _recv_bytes(self._sock)

    @property
    def host(self):
        return self._host

    @property
    def port(self):
        return self._port

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

    cpdef int add(self, str key, int amount):
        return self._store.add(f"{self._prefix}/{key}", amount)

    cpdef bint delete_key(self, str key):
        return self._store.delete_key(f"{self._prefix}/{key}")

    cpdef int num_keys(self):
        return self._store.num_keys()

    cpdef bint check(self, list keys):
        return self._store.check([f"{self._prefix}/{k}" for k in keys])

    cpdef bytes compare_set(self, str key, str expected, str desired):
        return self._store.compare_set(f"{self._prefix}/{key}", expected, desired)

    cpdef wait(self, list keys, timeout=None):
        self._store.wait([f"{self._prefix}/{k}" for k in keys], timeout)

    cpdef close(self):
        self._store.close()

    @property
    def underlying_store(self):
        return self._store


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

    cpdef int add(self, str key, int amount):
        cdef int old_val, new_val
        with self._cond:
            old_val = 0
            if key in self._data:
                old_val = struct.unpack("!i", self._data[key])[0]
            new_val = old_val + amount
            self._data[key] = struct.pack("!i", new_val)
            self._cond.notify_all()
        return new_val

    cpdef bint delete_key(self, str key):
        with self._lock:
            if key in self._data:
                del self._data[key]
                return True
            return False

    cpdef int num_keys(self):
        with self._lock:
            return len(self._data)

    cpdef bint check(self, list keys):
        cdef str k
        with self._lock:
            for k in keys:
                if k not in self._data:
                    return False
            return True

    cpdef bytes compare_set(self, str key, str expected, str desired):
        cdef bytes expected_b = expected.encode("utf-8")
        cdef bytes desired_b = desired.encode("utf-8")
        with self._cond:
            current = self._data.get(key, b"")
            if current == expected_b:
                self._data[key] = desired_b
                self._cond.notify_all()
            return self._data.get(key, b"")

    cpdef wait(self, list keys, timeout=None):
        cdef double deadline
        cdef double remaining
        cdef str k
        cdef bint all_present
        cdef double t = _to_seconds(timeout)
        deadline = time.monotonic() + t
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


# --- FileStore ---

import os as _os
import fcntl as _fcntl

cdef class FileStore(Store):
    """File-system-based distributed store.

    Uses a directory on a shared filesystem for coordination.
    Each key is stored as a separate file.
    """
    # cdef fields declared in _c10d.pxd

    def __init__(self, str file_name, int world_size=-1):
        self._path = file_name
        self._world_size = world_size
        _os.makedirs(file_name, exist_ok=True)

    cdef str _key_path(self, str key):
        # Sanitize key: replace / with _
        return _os.path.join(self._path, key.replace("/", "_"))

    cpdef set(self, str key, bytes value):
        cdef str path = self._key_path(key)
        with open(path, "wb") as f:
            _fcntl.flock(f, _fcntl.LOCK_EX)
            f.write(value)

    cpdef bytes get(self, str key):
        cdef str path = self._key_path(key)
        cdef double deadline = time.monotonic() + _DEFAULT_TIMEOUT
        while True:
            if _os.path.exists(path):
                with open(path, "rb") as f:
                    _fcntl.flock(f, _fcntl.LOCK_SH)
                    return f.read()
            if time.monotonic() >= deadline:
                raise TimeoutError(f"FileStore: GET '{key}' timed out")
            time.sleep(0.01)

    cpdef int add(self, str key, int amount):
        cdef str path = self._key_path(key)
        cdef int old_val, new_val
        # Use file locking for atomicity
        fd = _os.open(path, _os.O_RDWR | _os.O_CREAT)
        try:
            f = _os.fdopen(fd, "r+b")
            _fcntl.flock(f, _fcntl.LOCK_EX)
            data = f.read()
            old_val = 0
            if len(data) >= 4:
                old_val = struct.unpack("!i", data[:4])[0]
            new_val = old_val + amount
            f.seek(0)
            f.write(struct.pack("!i", new_val))
            f.truncate()
            f.flush()
            return new_val
        finally:
            try:
                f.close()
            except Exception:
                _os.close(fd)

    cpdef bint delete_key(self, str key):
        cdef str path = self._key_path(key)
        if _os.path.exists(path):
            _os.remove(path)
            return True
        return False

    cpdef int num_keys(self):
        if not _os.path.isdir(self._path):
            return 0
        return len(_os.listdir(self._path))

    cpdef bint check(self, list keys):
        cdef str k
        for k in keys:
            if not _os.path.exists(self._key_path(k)):
                return False
        return True

    cpdef bytes compare_set(self, str key, str expected, str desired):
        cdef str path = self._key_path(key)
        cdef bytes expected_b = expected.encode("utf-8")
        cdef bytes desired_b = desired.encode("utf-8")
        fd = _os.open(path, _os.O_RDWR | _os.O_CREAT)
        try:
            f = _os.fdopen(fd, "r+b")
            _fcntl.flock(f, _fcntl.LOCK_EX)
            current = f.read()
            if current == expected_b:
                f.seek(0)
                f.write(desired_b)
                f.truncate()
                f.flush()
                return desired_b
            return current
        finally:
            try:
                f.close()
            except Exception:
                _os.close(fd)

    cpdef wait(self, list keys, timeout=None):
        cdef double t = _to_seconds(timeout)
        cdef double deadline = time.monotonic() + t
        cdef str k
        cdef bint all_present
        while True:
            all_present = True
            for k in keys:
                if not _os.path.exists(self._key_path(k)):
                    all_present = False
                    break
            if all_present:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"FileStore: WAIT timed out for keys {keys}")
            time.sleep(0.01)

    cpdef close(self):
        pass

    @property
    def path(self):
        return self._path


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
        self._coalescing_ops = []
        self._coalescing_active = False

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

    @bound_device_id.setter
    def bound_device_id(self, value):
        self._bound_device_id = value

    def _set_group_name(self, str name):
        self._group_name = name

    def _set_group_desc(self, str desc):
        self._group_desc = desc

    # --- Collective dispatch (delegate to subclass) ---

    def allreduce(self, tensors, opts=None):
        raise NotImplementedError(f"{type(self).__name__}.allreduce not implemented")

    def broadcast(self, tensors, opts=None):
        raise NotImplementedError(f"{type(self).__name__}.broadcast not implemented")

    def allgather(self, output_tensors, input_tensors, opts=None):
        raise NotImplementedError(f"{type(self).__name__}.allgather not implemented")

    def reduce(self, tensors, opts=None):
        raise NotImplementedError(f"{type(self).__name__}.reduce not implemented")

    def reduce_scatter(self, output_tensors, input_tensors, opts=None):
        raise NotImplementedError(f"{type(self).__name__}.reduce_scatter not implemented")

    def gather(self, output_tensors, input_tensors, opts=None):
        raise NotImplementedError(f"{type(self).__name__}.gather not implemented")

    def scatter(self, output_tensors, input_tensors, opts=None):
        raise NotImplementedError(f"{type(self).__name__}.scatter not implemented")

    def alltoall(self, output_tensors, input_tensors, opts=None):
        raise NotImplementedError(f"{type(self).__name__}.alltoall not implemented")

    def alltoall_base(self, output, input, output_split_sizes,
                      input_split_sizes, opts=None):
        raise NotImplementedError(f"{type(self).__name__}.alltoall_base not implemented")

    def send(self, tensors, int dst, int tag=0):
        raise NotImplementedError(f"{type(self).__name__}.send not implemented")

    def recv(self, tensors, int src, int tag=0):
        raise NotImplementedError(f"{type(self).__name__}.recv not implemented")

    def recv_anysource(self, tensors, int tag=0):
        raise NotImplementedError(f"{type(self).__name__}.recv_anysource not implemented")

    def barrier(self, opts=None):
        raise NotImplementedError(f"{type(self).__name__}.barrier not implemented")

    # --- Coalescing ---

    def _start_coalescing(self, device=None):
        """Begin collecting collective operations for batched execution."""
        self._coalescing_ops = []
        self._coalescing_active = True

    def _end_coalescing(self, device=None):
        """Flush all queued operations. Returns Work."""
        self._coalescing_active = False
        ops = self._coalescing_ops
        self._coalescing_ops = []
        # Execute all queued ops sequentially (no actual batching yet)
        last_work = None
        for op_fn in ops:
            last_work = op_fn()
        if last_work is None:
            last_work = Work()
            last_work._completed = True
        return last_work

    def allreduce_coalesced(self, tensors, opts=None):
        """Allreduce multiple tensors. Default: sequential allreduce each."""
        works = []
        for t in tensors:
            works.append(self.allreduce(t, opts))
        # Return last work
        return works[-1] if works else Work()

    def _allgather_base(self, output, input, opts=None):
        """Allgather into a single flat output tensor."""
        return self.allgather(output, input, opts)

    def reduce_scatter_tensor_coalesced(self, outputs, inputs, opts=None):
        """Reduce-scatter multiple tensor pairs. Default: sequential."""
        works = []
        for out, inp in zip(outputs, inputs):
            works.append(self.reduce_scatter(out, inp, opts))
        return works[-1] if works else Work()

    def allgather_into_tensor_coalesced(self, outputs, inputs, opts=None):
        """Allgather multiple tensor pairs into flat buffers. Default: sequential."""
        works = []
        for out, inp in zip(outputs, inputs):
            works.append(self._allgather_base(out, inp, opts))
        return works[-1] if works else Work()

    BackendType = BackendType
