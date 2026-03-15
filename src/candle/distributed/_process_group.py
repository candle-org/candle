import os
import ctypes

from ._work import Work


class ProcessGroup:
    def __init__(self, rank, size):
        self._rank = rank
        self._size = size
        self._group_name = ""
        self._group_desc = ""
        self._ranks = None
        self._bound_device_id = None

    def rank(self):
        return self._rank

    def size(self):
        return self._size

    def name(self):
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


class ProcessGroupHCCL(ProcessGroup):
    """HCCL process group using direct ctypes bindings to libhccl.so.

    Initialization follows torch_npu's pattern:
    1. Try ranktable path (RANK_TABLE_FILE + HcclCommInitClusterInfoConfig)
    2. Fall back to root info path (HcclGetRootInfo + HcclCommInitRootInfoConfig)

    HCCL bindings are imported lazily when this class is instantiated, so
    importing the base ProcessGroup class does not require libhccl.so.
    """

    def __init__(self, store, rank, size, device_id=None, group_name="",
                 group_ranks=None):
        super().__init__(rank, size)
        if device_id is None:
            device_id = rank % 8
        self._device_id = device_id
        self._comm = None
        self._group_name = group_name
        self._ranks = group_ranks
        self._store = store
        self._init_hccl(store, group_name)

    @staticmethod
    def _load_bindings():
        from ._hccl.hccl_bindings import (
            get_bindings, HcclRootInfo, HcclCommConfig, HCCL_ROOT_INFO_BYTES,
            HCCL_COMM_CONFIG_COMM_NAME,
            hccl_comm_config_init, is_hccl_feature_supported,
            dtype_to_hccl, _check,
        )
        return (get_bindings, HcclRootInfo, HcclCommConfig,
                HCCL_ROOT_INFO_BYTES, HCCL_COMM_CONFIG_COMM_NAME,
                hccl_comm_config_init, is_hccl_feature_supported,
                dtype_to_hccl, _check)

    def _make_config(self):
        (_, _, HcclCommConfig, _, HCCL_COMM_CONFIG_COMM_NAME,
         hccl_comm_config_init, is_hccl_feature_supported, _, _) = self._load_bindings()
        config = HcclCommConfig()
        hccl_comm_config_init(config)
        # Old CANN (<=8.3) without COMM_NAME capability needs a truncated
        # size so the library ignores the (absent) commName field.
        # New CANN (>=8.5) must keep the full V8 size even when the
        # capability bit is unset — overwriting would corrupt the config.
        from ._hccl.hccl_bindings import _USE_V8
        if not _USE_V8 and not is_hccl_feature_supported(HCCL_COMM_CONFIG_COMM_NAME):
            import struct
            struct.pack_into("<Q", (ctypes.c_ubyte * 8).from_buffer(config), 0, 32)
        return config

    def _try_init_cluster_info(self):
        """Try ranktable-based init (preferred for multi-node)."""
        rank_table = os.environ.get("RANK_TABLE_FILE")
        if not rank_table or not os.path.isfile(rank_table):
            return False
        get_bindings = self._load_bindings()[0]
        bindings = get_bindings()
        if bindings.comm_init_cluster_info_config is None:
            return False
        config = self._make_config()
        comm = ctypes.c_void_p()
        ret = bindings.comm_init_cluster_info_config(
            rank_table.encode("utf-8"),
            ctypes.c_uint32(self._rank),
            ctypes.byref(config),
            ctypes.byref(comm),
        )
        if ret != 0:
            return False
        self._comm = comm
        return True

    def _init_root_info(self, store, prefix=""):
        """Root info broadcast path (standard NCCL-like init)."""
        (get_bindings, HcclRootInfo, _, HCCL_ROOT_INFO_BYTES,
         _, _, _, _, _check) = self._load_bindings()
        bindings = get_bindings()
        key = f"{prefix}/hccl_root_info" if prefix else "hccl_root_info"

        if self._rank == 0:
            root_info = HcclRootInfo()
            ret = bindings.get_root_info(ctypes.byref(root_info))
            _check(ret, "HcclGetRootInfo")
            store.set(key, ctypes.string_at(ctypes.addressof(root_info), HCCL_ROOT_INFO_BYTES))
        else:
            store.wait([key])

        raw = store.get(key)
        if len(raw) != HCCL_ROOT_INFO_BYTES:
            raise RuntimeError(
                f"HCCL root info size mismatch: expected {HCCL_ROOT_INFO_BYTES}, got {len(raw)}"
            )
        root_info = HcclRootInfo()
        ctypes.memmove(ctypes.addressof(root_info), raw, HCCL_ROOT_INFO_BYTES)

        config = self._make_config()
        comm = ctypes.c_void_p()
        ret = bindings.comm_init_root_info_config(
            ctypes.c_uint32(self._size),
            ctypes.byref(root_info),
            ctypes.c_uint32(self._rank),
            ctypes.byref(config),
            ctypes.byref(comm),
        )
        _check(ret, "HcclCommInitRootInfoConfig")
        self._comm = comm

    def _init_hccl(self, store, prefix=""):
        from .._backends.npu import state as npu_state
        npu_state.set_device(self._device_id)

        # Try ranktable path first, fall back to root info
        if not self._try_init_cluster_info():
            self._init_root_info(store, prefix)

    def _stream(self):
        from .._backends.npu import state as npu_state
        return npu_state.current_stream(self._device_id).stream

    def _make_work(self, stream, source_rank=-1):
        return Work(stream=stream, device_id=self._device_id,
                    source_rank=source_rank)

    def _tensor_args(self, tensor):
        _, _, _, _, _, _, _, dtype_to_hccl, _ = self._load_bindings()
        ptr = ctypes.c_void_p(tensor.storage().data_ptr())
        count = ctypes.c_uint64(tensor.numel())
        hccl_dtype = ctypes.c_int32(dtype_to_hccl(tensor.dtype))
        return ptr, count, hccl_dtype

    def allreduce(self, tensor, op=0):
        get_bindings, _, _, _, _, _, _, _, _check = self._load_bindings()
        bindings = get_bindings()
        stream = self._stream()
        ptr, count, dt = self._tensor_args(tensor)
        ret = bindings.all_reduce(
            ptr, ptr, count, dt, ctypes.c_int32(int(op)),
            self._comm, ctypes.c_void_p(int(stream)),
        )
        _check(ret, "HcclAllReduce")
        return self._make_work(stream)

    def broadcast(self, tensor, root=0):
        get_bindings, _, _, _, _, _, _, _, _check = self._load_bindings()
        bindings = get_bindings()
        stream = self._stream()
        ptr, count, dt = self._tensor_args(tensor)
        ret = bindings.broadcast(
            ptr, count, dt, ctypes.c_uint32(root),
            self._comm, ctypes.c_void_p(int(stream)),
        )
        _check(ret, "HcclBroadcast")
        return self._make_work(stream)

    def allgather(self, output_tensor, input_tensor):
        get_bindings, _, _, _, _, _, _, _, _check = self._load_bindings()
        bindings = get_bindings()
        stream = self._stream()
        in_ptr, in_count, dt = self._tensor_args(input_tensor)
        out_ptr = ctypes.c_void_p(output_tensor.storage().data_ptr())
        ret = bindings.all_gather(
            in_ptr, out_ptr, in_count, dt,
            self._comm, ctypes.c_void_p(int(stream)),
        )
        _check(ret, "HcclAllGather")
        return self._make_work(stream)

    def reduce_scatter(self, output_tensor, input_tensor, op=0):
        get_bindings, _, _, _, _, _, _, _, _check = self._load_bindings()
        bindings = get_bindings()
        stream = self._stream()
        in_ptr = ctypes.c_void_p(input_tensor.storage().data_ptr())
        out_ptr, out_count, dt = self._tensor_args(output_tensor)
        ret = bindings.reduce_scatter(
            in_ptr, out_ptr, out_count, dt, ctypes.c_int32(int(op)),
            self._comm, ctypes.c_void_p(int(stream)),
        )
        _check(ret, "HcclReduceScatter")
        return self._make_work(stream)

    def reduce(self, tensor, dst=0, op=0):
        get_bindings, _, _, _, _, _, _, _, _check = self._load_bindings()
        bindings = get_bindings()
        stream = self._stream()
        ptr, count, dt = self._tensor_args(tensor)
        ret = bindings.reduce(
            ptr, ptr, count, dt, ctypes.c_int32(int(op)), ctypes.c_uint32(dst),
            self._comm, ctypes.c_void_p(int(stream)),
        )
        _check(ret, "HcclReduce")
        return self._make_work(stream)

    def scatter(self, output_tensor, input_tensor, src=0):
        get_bindings, _, _, _, _, _, _, _, _check = self._load_bindings()
        bindings = get_bindings()
        stream = self._stream()
        in_ptr = ctypes.c_void_p(input_tensor.storage().data_ptr())
        out_ptr, out_count, dt = self._tensor_args(output_tensor)
        ret = bindings.scatter(
            in_ptr, out_ptr, out_count, dt, ctypes.c_uint32(src),
            self._comm, ctypes.c_void_p(int(stream)),
        )
        _check(ret, "HcclScatter")
        return self._make_work(stream)

    def barrier(self):
        get_bindings, _, _, _, _, _, _, _, _check = self._load_bindings()
        bindings = get_bindings()
        stream = self._stream()
        ret = bindings.barrier(
            self._comm, ctypes.c_void_p(int(stream)),
        )
        _check(ret, "HcclBarrier")
        return self._make_work(stream)

    def send(self, tensor, dst):
        get_bindings, _, _, _, _, _, _, _, _check = self._load_bindings()
        bindings = get_bindings()
        stream = self._stream()
        ptr, count, dt = self._tensor_args(tensor)
        ret = bindings.send(
            ptr, count, dt, ctypes.c_uint32(dst),
            self._comm, ctypes.c_void_p(int(stream)),
        )
        _check(ret, "HcclSend")
        return self._make_work(stream)

    def recv(self, tensor, src):
        get_bindings, _, _, _, _, _, _, _, _check = self._load_bindings()
        bindings = get_bindings()
        stream = self._stream()
        ptr, count, dt = self._tensor_args(tensor)
        ret = bindings.recv(
            ptr, count, dt, ctypes.c_uint32(src),
            self._comm, ctypes.c_void_p(int(stream)),
        )
        _check(ret, "HcclRecv")
        return self._make_work(stream, source_rank=src)

    def _p2p_exchange(self, send_ptr, recv_ptr, count, hccl_dtype, peer,
                      stream, bindings, _check):
        """Send/recv with a single peer using deadlock-free ordering."""
        if self._rank < peer:
            ret = bindings.send(
                send_ptr, ctypes.c_uint64(count), hccl_dtype,
                ctypes.c_uint32(peer), self._comm,
                ctypes.c_void_p(int(stream)))
            _check(ret, f"HcclSend to {peer}")
            ret = bindings.recv(
                recv_ptr, ctypes.c_uint64(count), hccl_dtype,
                ctypes.c_uint32(peer), self._comm,
                ctypes.c_void_p(int(stream)))
            _check(ret, f"HcclRecv from {peer}")
        else:
            ret = bindings.recv(
                recv_ptr, ctypes.c_uint64(count), hccl_dtype,
                ctypes.c_uint32(peer), self._comm,
                ctypes.c_void_p(int(stream)))
            _check(ret, f"HcclRecv from {peer}")
            ret = bindings.send(
                send_ptr, ctypes.c_uint64(count), hccl_dtype,
                ctypes.c_uint32(peer), self._comm,
                ctypes.c_void_p(int(stream)))
            _check(ret, f"HcclSend to {peer}")

    def all_to_all(self, output_tensors, input_tensors):
        """All-to-all using native HcclAlltoAll/HcclAlltoAllV or P2P fallback.

        CANN >=8.5 (V8): Uses native HcclAlltoAll (equal split) or
        HcclAlltoAllV (unequal split) for any world size.
        CANN <=8.3: Falls back to P2P send/recv (preserved for compat).
        """
        from ._hccl.hccl_bindings import _USE_V8
        from .._backends.npu import runtime as npu_runtime
        get_bindings, _, _, _, _, _, _, dtype_to_hccl, _check = self._load_bindings()
        bindings = get_bindings()
        stream = self._stream()

        if _USE_V8:
            import candle as torch
            dtype = input_tensors[0].dtype
            itemsize = dtype.itemsize
            world_size = self._size

            equal_split = (
                len({t.numel() for t in input_tensors}) == 1 and
                len({t.numel() for t in output_tensors}) == 1
            )

            if equal_split:
                count_per_rank = input_tensors[0].numel()
                total_count = count_per_rank * world_size

                send_flat = torch.empty(total_count, dtype=dtype,
                                        device=input_tensors[0].device)
                recv_flat = torch.empty(total_count, dtype=dtype,
                                        device=output_tensors[0].device)

                dst_base = send_flat.storage().data_ptr()
                for i, t in enumerate(input_tensors):
                    npu_runtime.memcpy_d2d(
                        dst_base + i * count_per_rank * itemsize,
                        count_per_rank * itemsize,
                        t.storage().data_ptr(),
                    )

                # Ensure pack memcpy is visible before the collective
                dev_id = self._device_id if self._device_id is not None else 0
                npu_runtime.get_runtime(dev_id).synchronize_stream(stream)

                ret = bindings.all_to_all(
                    ctypes.c_void_p(send_flat.storage().data_ptr()),
                    ctypes.c_uint64(count_per_rank),
                    ctypes.c_int32(dtype_to_hccl(dtype)),
                    ctypes.c_void_p(recv_flat.storage().data_ptr()),
                    ctypes.c_uint64(count_per_rank),
                    ctypes.c_int32(dtype_to_hccl(dtype)),
                    self._comm,
                    ctypes.c_void_p(int(stream)))
                _check(ret, "HcclAlltoAll")

                dev_id = self._device_id if self._device_id is not None else 0
                npu_runtime.get_runtime(dev_id).synchronize_stream(stream)

                src_base = recv_flat.storage().data_ptr()
                for i, t in enumerate(output_tensors):
                    npu_runtime.memcpy_d2d(
                        t.storage().data_ptr(),
                        count_per_rank * itemsize,
                        src_base + i * count_per_rank * itemsize,
                    )
            else:
                # Unequal split: pack, use HcclAlltoAllV, unpack
                send_counts = [t.numel() for t in input_tensors]
                recv_counts = [t.numel() for t in output_tensors]
                total_send = sum(send_counts)
                total_recv = sum(recv_counts)

                send_flat = torch.empty(total_send, dtype=dtype,
                                        device=input_tensors[0].device)
                recv_flat = torch.empty(total_recv, dtype=dtype,
                                        device=output_tensors[0].device)

                # Pack input tensors into send_flat
                dst_base = send_flat.storage().data_ptr()
                offset = 0
                for i, t in enumerate(input_tensors):
                    nbytes = send_counts[i] * itemsize
                    npu_runtime.memcpy_d2d(
                        dst_base + offset, nbytes,
                        t.storage().data_ptr(),
                    )
                    offset += nbytes

                # Ensure pack memcpy is visible before the collective
                dev_id = self._device_id if self._device_id is not None else 0
                npu_runtime.get_runtime(dev_id).synchronize_stream(stream)

                # Build counts and displacements (uint64 arrays)
                ArrayU64 = ctypes.c_uint64 * world_size
                sc = ArrayU64(*send_counts)
                rc = ArrayU64(*recv_counts)
                sd = ArrayU64()
                rd = ArrayU64()
                d = 0
                for i, c in enumerate(send_counts):
                    sd[i] = d
                    d += c
                d = 0
                for i, c in enumerate(recv_counts):
                    rd[i] = d
                    d += c

                hccl_dt = ctypes.c_int32(dtype_to_hccl(dtype))
                ret = bindings.all_to_all_v(
                    ctypes.c_void_p(send_flat.storage().data_ptr()),
                    ctypes.cast(sc, ctypes.c_void_p),
                    ctypes.cast(sd, ctypes.c_void_p),
                    hccl_dt,
                    ctypes.c_void_p(recv_flat.storage().data_ptr()),
                    ctypes.cast(rc, ctypes.c_void_p),
                    ctypes.cast(rd, ctypes.c_void_p),
                    hccl_dt,
                    self._comm,
                    ctypes.c_void_p(int(stream)))
                _check(ret, "HcclAlltoAllV")

                dev_id = self._device_id if self._device_id is not None else 0
                npu_runtime.get_runtime(dev_id).synchronize_stream(stream)

                # Unpack recv buffer into output tensors
                src_base = recv_flat.storage().data_ptr()
                offset = 0
                for i, t in enumerate(output_tensors):
                    nbytes = recv_counts[i] * itemsize
                    npu_runtime.memcpy_d2d(
                        t.storage().data_ptr(), nbytes,
                        src_base + offset,
                    )
                    offset += nbytes

            return self._make_work(stream)

        # CANN <=8.3: P2P fallback (preserved for backward compat)
        for peer in range(self._size):
            numel = input_tensors[peer].numel()
            hccl_dt = ctypes.c_int32(dtype_to_hccl(input_tensors[peer].dtype))
            itemsize = input_tensors[peer].dtype.itemsize
            if peer == self._rank:
                nbytes = numel * itemsize
                npu_runtime.memcpy_d2d(
                    output_tensors[peer].storage().data_ptr(), nbytes,
                    input_tensors[peer].storage().data_ptr(),
                )
            else:
                send_ptr = ctypes.c_void_p(input_tensors[peer].storage().data_ptr())
                recv_ptr = ctypes.c_void_p(output_tensors[peer].storage().data_ptr())
                self._p2p_exchange(send_ptr, recv_ptr, numel, hccl_dt,
                                   peer, stream, bindings, _check)

        return self._make_work(stream)

    def all_to_all_single(self, output, input, count_per_rank):
        """All-to-all on contiguous buffers using native HcclAlltoAll."""
        from ._hccl.hccl_bindings import _USE_V8
        get_bindings, _, _, _, _, _, _, dtype_to_hccl, _check = self._load_bindings()
        bindings = get_bindings()
        stream = self._stream()

        if _USE_V8:
            # CANN >=8.5: use native HcclAlltoAll for all world sizes
            ret = bindings.all_to_all(
                ctypes.c_void_p(input.storage().data_ptr()),
                ctypes.c_uint64(count_per_rank),
                ctypes.c_int32(dtype_to_hccl(input.dtype)),
                ctypes.c_void_p(output.storage().data_ptr()),
                ctypes.c_uint64(count_per_rank),
                ctypes.c_int32(dtype_to_hccl(output.dtype)),
                self._comm,
                ctypes.c_void_p(int(stream)))
            _check(ret, "HcclAlltoAll")
            return self._make_work(stream)

        # CANN <=8.3: P2P fallback (preserved for backward compat)
        from .._backends.npu import runtime as npu_runtime
        hccl_dt = ctypes.c_int32(dtype_to_hccl(input.dtype))
        itemsize = input.dtype.itemsize
        chunk_bytes = count_per_rank * itemsize
        in_base = input.storage().data_ptr()
        out_base = output.storage().data_ptr()

        for peer in range(self._size):
            if peer == self._rank:
                npu_runtime.memcpy_d2d(
                    out_base + peer * chunk_bytes, chunk_bytes,
                    in_base + peer * chunk_bytes,
                )
            else:
                send_ptr = ctypes.c_void_p(in_base + peer * chunk_bytes)
                recv_ptr = ctypes.c_void_p(out_base + peer * chunk_bytes)
                self._p2p_exchange(send_ptr, recv_ptr, count_per_rank,
                                   hccl_dt, peer, stream, bindings, _check)

        return self._make_work(stream)

    def all_to_all_single_v(self, output, input, input_split_sizes,
                            output_split_sizes):
        """All-to-all with variable splits on contiguous buffers.

        Uses HcclAlltoAllV on CANN >=8.5, P2P fallback on older CANN.
        """
        from ._hccl.hccl_bindings import _USE_V8
        get_bindings, _, _, _, _, _, _, dtype_to_hccl, _check = self._load_bindings()
        bindings = get_bindings()
        stream = self._stream()

        if _USE_V8:
            world_size = self._size
            ArrayU64 = ctypes.c_uint64 * world_size

            sc = ArrayU64(*[int(s) for s in input_split_sizes])
            rc = ArrayU64(*[int(s) for s in output_split_sizes])
            sd = ArrayU64()
            rd = ArrayU64()
            d = 0
            for i, s in enumerate(input_split_sizes):
                sd[i] = d
                d += int(s)
            d = 0
            for i, s in enumerate(output_split_sizes):
                rd[i] = d
                d += int(s)

            hccl_dt = ctypes.c_int32(dtype_to_hccl(input.dtype))
            ret = bindings.all_to_all_v(
                ctypes.c_void_p(input.storage().data_ptr()),
                ctypes.cast(sc, ctypes.c_void_p),
                ctypes.cast(sd, ctypes.c_void_p),
                hccl_dt,
                ctypes.c_void_p(output.storage().data_ptr()),
                ctypes.cast(rc, ctypes.c_void_p),
                ctypes.cast(rd, ctypes.c_void_p),
                hccl_dt,
                self._comm,
                ctypes.c_void_p(int(stream)))
            _check(ret, "HcclAlltoAllV")
            return self._make_work(stream)

        # CANN <=8.3: P2P fallback (preserved for backward compat)
        from .._backends.npu import runtime as npu_runtime
        hccl_dt = ctypes.c_int32(dtype_to_hccl(input.dtype))
        itemsize = input.dtype.itemsize
        in_base = input.storage().data_ptr()
        out_base = output.storage().data_ptr()

        # Pre-compute byte offsets per peer
        in_offsets = []
        d = 0
        for s in input_split_sizes:
            in_offsets.append(d)
            d += int(s) * itemsize
        out_offsets = []
        d = 0
        for s in output_split_sizes:
            out_offsets.append(d)
            d += int(s) * itemsize

        for peer in range(self._size):
            send_numel = int(input_split_sizes[peer])
            recv_numel = int(output_split_sizes[peer])
            if peer == self._rank:
                nbytes = send_numel * itemsize
                if nbytes > 0:
                    npu_runtime.memcpy_d2d(
                        out_base + out_offsets[peer], nbytes,
                        in_base + in_offsets[peer],
                    )
            else:
                send_ptr = ctypes.c_void_p(in_base + in_offsets[peer])
                recv_ptr = ctypes.c_void_p(out_base + out_offsets[peer])
                if self._rank < peer:
                    if send_numel > 0:
                        ret = bindings.send(
                            send_ptr, ctypes.c_uint64(send_numel), hccl_dt,
                            ctypes.c_uint32(peer), self._comm,
                            ctypes.c_void_p(int(stream)))
                        _check(ret, f"HcclSend to {peer}")
                    if recv_numel > 0:
                        ret = bindings.recv(
                            recv_ptr, ctypes.c_uint64(recv_numel), hccl_dt,
                            ctypes.c_uint32(peer), self._comm,
                            ctypes.c_void_p(int(stream)))
                        _check(ret, f"HcclRecv from {peer}")
                else:
                    if recv_numel > 0:
                        ret = bindings.recv(
                            recv_ptr, ctypes.c_uint64(recv_numel), hccl_dt,
                            ctypes.c_uint32(peer), self._comm,
                            ctypes.c_void_p(int(stream)))
                        _check(ret, f"HcclRecv from {peer}")
                    if send_numel > 0:
                        ret = bindings.send(
                            send_ptr, ctypes.c_uint64(send_numel), hccl_dt,
                            ctypes.c_uint32(peer), self._comm,
                            ctypes.c_void_p(int(stream)))
                        _check(ret, f"HcclSend to {peer}")

        return self._make_work(stream)

    def destroy(self):
        if self._comm is not None:
            get_bindings = self._load_bindings()[0]
            bindings = get_bindings()
            bindings.comm_destroy(self._comm)
            self._comm = None
