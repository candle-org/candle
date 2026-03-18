# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython Gloo backend: TcpTransport + numpy collectives + ProcessGroupGloo."""

import socket
import struct
import threading
import numpy as np
from enum import IntEnum

from ._c10d import RedOpType, Work, ProcessGroup


# ---------------------------------------------------------------------------
# Section 1: _recvall helper (from _tcp_transport.py)
# ---------------------------------------------------------------------------

cdef bytes _recvall(object sock, Py_ssize_t n):
    """Read exactly *n* bytes from a socket."""
    cdef bytearray buf = bytearray()
    cdef bytes chunk
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer connection closed")
        buf.extend(chunk)
    return bytes(buf)


# ---------------------------------------------------------------------------
# Section 2: Numpy collectives (from _numpy_collectives.py)
# ---------------------------------------------------------------------------

# ReduceOp int value -> numpy ufunc
_REDUCE_OPS = {
    int(RedOpType.SUM): np.add,
    int(RedOpType.PRODUCT): np.multiply,
    int(RedOpType.MAX): np.maximum,
    int(RedOpType.MIN): np.minimum,
    int(RedOpType.BAND): np.bitwise_and,
    int(RedOpType.BOR): np.bitwise_or,
    int(RedOpType.BXOR): np.bitwise_xor,
    # AVG is handled specially by the caller (sum then divide)
    int(RedOpType.AVG): np.add,
}


cdef object apply_reduce_op(int op, object a, object b):
    """Apply a reduce operation element-wise: result = op(a, b)."""
    cdef int op_int = int(op)
    fn = _REDUCE_OPS.get(op_int)
    if fn is None:
        raise ValueError(f"Unsupported reduce op: {op}")
    return fn(a, b)


cdef bytes serialize_array(object arr):
    """Serialize a numpy array to bytes.

    Wire format:
      1 byte  - length of dtype name
      N bytes - dtype name (ascii)
      4 bytes - ndim (uint32 big-endian)
      ndim*8 bytes - shape dims (uint64 big-endian each)
      rest    - raw data bytes
    """
    arr = np.ascontiguousarray(arr)
    cdef bytes dtype_name = arr.dtype.str.encode("ascii")
    cdef bytes header = struct.pack("!B", len(dtype_name)) + dtype_name
    header += struct.pack("!I", arr.ndim)
    for s in arr.shape:
        header += struct.pack("!Q", s)
    return header + arr.tobytes()


cdef object deserialize_array(bytes data):
    """Deserialize bytes back to a numpy array."""
    cdef Py_ssize_t offset = 0
    cdef int dtype_len = data[offset]
    offset += 1
    cdef bytes dtype_raw = data[offset:offset + dtype_len]
    cdef str dtype_name = dtype_raw.decode("ascii")
    offset += dtype_len
    cdef int ndim = struct.unpack("!I", data[offset:offset + 4])[0]
    offset += 4
    cdef list shape = []
    cdef int i
    for i in range(ndim):
        shape.append(struct.unpack("!Q", data[offset:offset + 8])[0])
        offset += 8
    cdef bytes body = data[offset:]
    return np.frombuffer(body, dtype=np.dtype(dtype_name)).reshape(tuple(shape))


# ---------------------------------------------------------------------------
# Section 3: TcpTransport (from _tcp_transport.py)
# ---------------------------------------------------------------------------

cdef class TcpTransport:
    """Full-mesh TCP connection manager for peer-to-peer communication."""

    cdef int _rank
    cdef int _world_size
    cdef double _timeout
    cdef bint _closed
    cdef dict _peers        # peer_rank -> socket
    cdef dict _locks        # peer_rank -> threading.Lock
    cdef object _server

    def __init__(self, object store, int rank, int world_size,
                 str prefix="", double timeout=300):
        self._rank = rank
        self._world_size = world_size
        self._timeout = timeout
        self._closed = False
        self._peers = {}
        self._locks = {}

        if world_size < 2:
            return

        # Bind ephemeral server socket for accepting incoming connections
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.settimeout(timeout)
        self._server.bind(("", 0))
        self._server.listen(world_size)
        cdef str host = socket.gethostname()
        cdef int port = self._server.getsockname()[1]

        # Publish our address to the store so peers can find us
        cdef str addr_key = f"gloo_addr_rank_{rank}"
        if prefix:
            addr_key = f"{prefix}/{addr_key}"
        store.set(addr_key, f"{host}:{port}".encode("utf-8"))

        # Collect all peer addresses
        cdef dict peer_addrs = {}
        cdef str peer_key
        cdef str addr_str
        cdef str h
        cdef str p_str
        cdef int p
        cdef int i
        for i in range(world_size):
            if i == rank:
                continue
            peer_key = f"gloo_addr_rank_{i}"
            if prefix:
                peer_key = f"{prefix}/{peer_key}"
            store.wait([peer_key])
            raw = store.get(peer_key)
            addr_str = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            h, p_str = addr_str.rsplit(":", 1)
            peer_addrs[i] = (h, int(p_str))

        # Establish connections with deterministic ordering to avoid deadlock:
        # rank i connects to rank j where j > i;
        # rank j accepts from rank i where i < j.
        #
        # Process lower ranks first (accept from them), then higher ranks
        # (connect to them).

        cdef object conn
        cdef bytes peer_rank_bytes
        cdef int peer_rank
        cdef object sock

        # Accept connections from ranks < self
        for i in sorted(r for r in range(world_size) if r < rank):
            conn, _ = self._server.accept()
            conn.settimeout(timeout)
            # Peer sends its rank as a 4-byte int so we know who connected
            peer_rank_bytes = _recvall(conn, 4)
            peer_rank = struct.unpack("!I", peer_rank_bytes)[0]
            self._peers[peer_rank] = conn
            self._locks[peer_rank] = threading.Lock()

        # Connect to ranks > self
        cdef int j
        for j in sorted(r for r in range(world_size) if r > rank):
            h_j, p_j = peer_addrs[j]
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect((h_j, p_j))
            # Send our rank so the peer knows who we are
            sock.sendall(struct.pack("!I", rank))
            self._peers[j] = sock
            self._locks[j] = threading.Lock()

    def send_to(self, int rank, bytes data):
        """Send data bytes to a peer rank. Thread-safe per socket."""
        cdef object lock = self._locks[rank]
        cdef object sock = self._peers[rank]
        with lock:
            # Wire format: 8-byte big-endian length + data
            sock.sendall(struct.pack("!Q", len(data)))
            sock.sendall(data)

    def recv_from(self, int rank):
        """Receive data bytes from a peer rank. Thread-safe per socket."""
        cdef object lock = self._locks[rank]
        cdef object sock = self._peers[rank]
        cdef bytes length_bytes
        cdef unsigned long long length
        with lock:
            length_bytes = _recvall(sock, 8)
            length = struct.unpack("!Q", length_bytes)[0]
            return _recvall(sock, length)

    def close(self):
        """Close all peer sockets and the server socket."""
        if self._closed:
            return
        self._closed = True
        for sock in self._peers.values():
            try:
                sock.close()
            except OSError:
                pass
        self._peers.clear()
        self._locks.clear()
        if hasattr(self, "_server"):
            try:
                self._server.close()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Section 4: ProcessGroupGloo (from _process_group_gloo.py)
# ---------------------------------------------------------------------------

cdef class ProcessGroupGloo:
    """Gloo-compatible process group using pure Python TCP transport.

    All collectives are synchronous (gather-to-root + broadcast-from-root).
    Work objects are returned already completed.

    Standalone cdef class (does not inherit from the Cython ProcessGroup to
    avoid cross-module inheritance complexity). Provides rank/size methods
    for API compatibility.
    """

    cdef int _rank
    cdef int _size
    cdef str _group_name
    cdef object _ranks
    cdef object _store
    cdef TcpTransport _transport

    def __init__(self, object store, int rank, int size,
                 str group_name="", object group_ranks=None):
        self._rank = rank
        self._size = size
        self._group_name = group_name
        self._ranks = group_ranks
        self._store = store
        self._transport = TcpTransport(store, rank, size, prefix=group_name)

    # -- rank / size accessors -----------------------------------------------

    def rank(self):
        return self._rank

    def size(self):
        return self._size

    # -- internal helpers ----------------------------------------------------

    def _tensor_to_numpy(self, tensor):
        """Convert a CPU tensor to a contiguous numpy array."""
        arr = tensor._numpy_view()
        return np.ascontiguousarray(arr)

    def _write_numpy_to_tensor(self, arr, tensor):
        """Write numpy array data back into a tensor in-place."""
        dst = tensor._numpy_view()
        np.copyto(dst, arr.reshape(dst.shape))

    def _make_work(self):
        """Return an already-completed Work object (synchronous collectives)."""
        work = Work(stream=None)
        work._completed = True
        return work

    # -- collective operations -----------------------------------------------

    def allreduce(self, tensor, op=0):
        """All-reduce: all ranks reduce and receive the result."""
        if self._size == 1:
            return self._make_work()

        arr = self._tensor_to_numpy(tensor)
        serialized = serialize_array(arr)

        # All non-root send to rank 0
        if self._rank != 0:
            self._transport.send_to(0, serialized)
        else:
            # Rank 0 receives from all peers and reduces
            result = arr.copy()
            for peer in range(1, self._size):
                peer_data = self._transport.recv_from(peer)
                peer_arr = deserialize_array(peer_data)
                result = apply_reduce_op(op, result, peer_arr)

            # Handle AVG: divide by world_size
            if int(op) == int(RedOpType.AVG):
                result = result / self._size

            # Rank 0 broadcasts result to all
            result_serialized = serialize_array(result)
            for peer in range(1, self._size):
                self._transport.send_to(peer, result_serialized)

            # Write result back into tensor
            self._write_numpy_to_tensor(result, tensor)

        # Non-root ranks receive the result
        if self._rank != 0:
            result_data = self._transport.recv_from(0)
            result = deserialize_array(result_data)
            self._write_numpy_to_tensor(result, tensor)

        return self._make_work()

    def broadcast(self, tensor, root=0):
        """Broadcast: root sends tensor to all other ranks."""
        if self._size == 1:
            return self._make_work()

        if self._rank == root:
            arr = self._tensor_to_numpy(tensor)
            serialized = serialize_array(arr)
            for peer in range(self._size):
                if peer != root:
                    self._transport.send_to(peer, serialized)
        else:
            data = self._transport.recv_from(root)
            arr = deserialize_array(data)
            self._write_numpy_to_tensor(arr, tensor)

        return self._make_work()

    def allgather(self, output_tensor, input_tensor):
        """All-gather: gather input from all ranks, concatenate into output."""
        if self._size == 1:
            # Just copy input to output
            np.copyto(output_tensor._numpy_view(), input_tensor._numpy_view())
            return self._make_work()

        arr = self._tensor_to_numpy(input_tensor)
        serialized = serialize_array(arr)

        # All ranks send to rank 0
        if self._rank != 0:
            self._transport.send_to(0, serialized)
        else:
            # Rank 0 gathers from all
            gathered = [arr]
            for peer in range(1, self._size):
                peer_data = self._transport.recv_from(peer)
                peer_arr = deserialize_array(peer_data)
                gathered.append(peer_arr)

            # Concatenate along first dimension
            result = np.concatenate(gathered, axis=0)
            result_serialized = serialize_array(result)

            # Rank 0 broadcasts concatenated result
            for peer in range(1, self._size):
                self._transport.send_to(peer, result_serialized)

            self._write_numpy_to_tensor(result, output_tensor)

        # Non-root ranks receive the result
        if self._rank != 0:
            result_data = self._transport.recv_from(0)
            result = deserialize_array(result_data)
            self._write_numpy_to_tensor(result, output_tensor)

        return self._make_work()

    def reduce(self, tensor, dst=0, op=0):
        """Reduce: all ranks send to dst, dst receives reduced result."""
        if self._size == 1:
            return self._make_work()

        arr = self._tensor_to_numpy(tensor)
        serialized = serialize_array(arr)

        # All non-dst send to dst
        if self._rank != dst:
            self._transport.send_to(dst, serialized)
        else:
            # dst receives from all and reduces
            result = arr.copy()
            for peer in range(self._size):
                if peer != dst:
                    peer_data = self._transport.recv_from(peer)
                    peer_arr = deserialize_array(peer_data)
                    result = apply_reduce_op(op, result, peer_arr)

            # Handle AVG
            if int(op) == int(RedOpType.AVG):
                result = result / self._size

            self._write_numpy_to_tensor(result, tensor)

        return self._make_work()

    def reduce_scatter(self, output_tensor, input_tensor, op=0):
        """Reduce-scatter: reduce input, split into chunks, scatter to ranks."""
        if self._size == 1:
            np.copyto(output_tensor._numpy_view(), input_tensor._numpy_view())
            return self._make_work()

        arr = self._tensor_to_numpy(input_tensor)
        serialized = serialize_array(arr)

        # All ranks send to rank 0
        if self._rank != 0:
            self._transport.send_to(0, serialized)
        else:
            # Rank 0 reduces
            result = arr.copy()
            for peer in range(1, self._size):
                peer_data = self._transport.recv_from(peer)
                peer_arr = deserialize_array(peer_data)
                result = apply_reduce_op(op, result, peer_arr)

            # Handle AVG
            if int(op) == int(RedOpType.AVG):
                result = result / self._size

            # Split into chunks (split along first dimension)
            chunk_size = result.shape[0] // self._size
            chunks = np.split(result, self._size, axis=0)

            # Send chunk i to rank i
            for peer in range(self._size):
                chunk_serialized = serialize_array(chunks[peer])
                if peer == 0:
                    self._write_numpy_to_tensor(chunks[peer], output_tensor)
                else:
                    self._transport.send_to(peer, chunk_serialized)

        # Non-root ranks receive their chunk
        if self._rank != 0:
            chunk_data = self._transport.recv_from(0)
            chunk = deserialize_array(chunk_data)
            self._write_numpy_to_tensor(chunk, output_tensor)

        return self._make_work()

    def scatter(self, output_tensor, input_tensor, src=0):
        """Scatter: src splits input into chunks, sends chunk i to rank i."""
        if self._size == 1:
            np.copyto(output_tensor._numpy_view(), input_tensor._numpy_view())
            return self._make_work()

        if self._rank == src:
            arr = self._tensor_to_numpy(input_tensor)
            # Split into chunks
            chunk_size = arr.shape[0] // self._size
            chunks = np.split(arr, self._size, axis=0)

            # Send chunk i to rank i
            for peer in range(self._size):
                chunk_serialized = serialize_array(chunks[peer])
                if peer == src:
                    self._write_numpy_to_tensor(chunks[peer], output_tensor)
                else:
                    self._transport.send_to(peer, chunk_serialized)
        else:
            # Receive chunk from src
            chunk_data = self._transport.recv_from(src)
            chunk = deserialize_array(chunk_data)
            self._write_numpy_to_tensor(chunk, output_tensor)

        return self._make_work()

    def barrier(self):
        """Barrier: all ranks synchronize."""
        if self._size == 1:
            return self._make_work()

        # All ranks send a 1-byte token to rank 0
        if self._rank != 0:
            self._transport.send_to(0, b"\x00")
        else:
            # Rank 0 waits for all
            for peer in range(1, self._size):
                self._transport.recv_from(peer)
            # Rank 0 sends ack to all
            for peer in range(1, self._size):
                self._transport.send_to(peer, b"\x00")

        # Non-root ranks wait for ack
        if self._rank != 0:
            self._transport.recv_from(0)

        return self._make_work()

    def send(self, tensor, dst):
        """Point-to-point send."""
        arr = self._tensor_to_numpy(tensor)
        serialized = serialize_array(arr)
        self._transport.send_to(dst, serialized)
        return self._make_work()

    def recv(self, tensor, src):
        """Point-to-point receive."""
        data = self._transport.recv_from(src)
        arr = deserialize_array(data)
        self._write_numpy_to_tensor(arr, tensor)
        work = self._make_work()
        work._source_rank = src
        return work

    def all_to_all(self, output_tensors, input_tensors):
        """All-to-all: each rank sends tensor i to rank i, receives from all.

        Uses P2P send/recv with deadlock-free ordering:
        - For peer < rank: recv first, then send
        - For peer > rank: send first, then recv
        - For peer == rank: local copy
        """
        if self._size == 1:
            np.copyto(output_tensors[0]._numpy_view(),
                      input_tensors[0]._numpy_view())
            return self._make_work()

        # Local copy first (rank sends to itself)
        np.copyto(output_tensors[self._rank]._numpy_view(),
                  input_tensors[self._rank]._numpy_view())

        # P2P exchange with deadlock-free ordering
        for peer in range(self._size):
            if peer == self._rank:
                continue

            if self._rank < peer:
                # Lower rank sends first, then receives
                arr = self._tensor_to_numpy(input_tensors[peer])
                serialized = serialize_array(arr)
                self._transport.send_to(peer, serialized)

                data = self._transport.recv_from(peer)
                arr = deserialize_array(data)
                self._write_numpy_to_tensor(arr, output_tensors[peer])
            else:
                # Higher rank receives first, then sends
                data = self._transport.recv_from(peer)
                arr = deserialize_array(data)
                self._write_numpy_to_tensor(arr, output_tensors[peer])

                arr = self._tensor_to_numpy(input_tensors[peer])
                serialized = serialize_array(arr)
                self._transport.send_to(peer, serialized)

        return self._make_work()

    def destroy(self):
        """Clean up transport resources."""
        self._transport.close()
