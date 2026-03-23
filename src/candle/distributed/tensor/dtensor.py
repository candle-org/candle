"""DTensor -- Distributed Tensor for FSDP2.

A lightweight metadata container wrapping a local tensor shard with
placement and mesh information.  FSDP manages communication manually;
DTensor does NOT perform automatic redistribution in MVP.
"""
from ..._tensor import Tensor

# ---------------------------------------------------------------------------
# Cython fastpath for shard/offset bookkeeping hot loops.
# Falls back gracefully to the pure-Python implementations below when the
# compiled extension is not available (e.g. editable install without build).
# ---------------------------------------------------------------------------
try:
    from .._dtensor_fastpath import (  # pylint: disable=no-name-in-module
        compute_local_shape_and_global_offset_cy as _cy_local_shape_offset,
        compute_global_shape_cy as _cy_global_shape,
        compute_global_stride_cy as _cy_global_stride,
        normalize_shard_dim_cy as _cy_normalize_dim,
        compute_gather_sizes_cy as _cy_gather_sizes,
        compute_scatter_sizes_cy as _cy_scatter_sizes,
    )
    _DTENSOR_FASTPATH_ACTIVE = True
except ImportError:
    _DTENSOR_FASTPATH_ACTIVE = False

# Expose fastpath module reference for test introspection
try:
    from .. import _dtensor_fastpath as _fp  # pylint: disable=no-name-in-module
except ImportError:
    _fp = None


class TensorMeta:
    """Global tensor metadata (shape as if the tensor were not sharded)."""
    __slots__ = ("shape", "stride", "dtype")

    def __init__(self, shape, stride, dtype):
        self.shape = shape
        self.stride = stride
        self.dtype = dtype

    def __repr__(self):
        return f"TensorMeta(shape={self.shape}, dtype={self.dtype})"


class DTensorSpec:
    """Metadata describing how a tensor is distributed."""
    __slots__ = ("mesh", "placements", "tensor_meta")

    def __init__(self, mesh, placements, tensor_meta=None):
        self.mesh = mesh
        self.placements = tuple(placements)
        self.tensor_meta = tensor_meta

    def has_shard_placement(self):
        """Return True if any placement is a Shard."""
        from .placement import Shard
        return any(isinstance(p, Shard) for p in self.placements)

    def __repr__(self):
        return f"DTensorSpec(placements={self.placements}, meta={self.tensor_meta})"


class DTensor(Tensor):
    """Distributed Tensor -- sharded parameter container for FSDP2.

    DTensor is a Tensor subclass that carries distributed placement metadata
    alongside the local tensor shard.  It hooks into the dispatch system via
    ``__torch_dispatch__`` to guard against accidental compute on sharded
    parameters (FSDP must unshard before forward/backward).
    """

    def __init__(self, local_tensor, spec, *, requires_grad=None):
        if not isinstance(local_tensor, Tensor):
            raise TypeError(
                f"local_tensor must be a candle Tensor, "
                f"got {type(local_tensor).__name__}"
            )
        if requires_grad is None:
            requires_grad = local_tensor.requires_grad
        # Set _local_tensor before super().__init__ because the grad property
        # setter (triggered by super().__init__ setting self.grad = None)
        # needs _local_tensor to exist.
        self._local_tensor = local_tensor
        self._spec = spec
        super().__init__(
            local_tensor._storage,
            local_tensor.shape,
            local_tensor.stride,
            local_tensor.offset,
            requires_grad=requires_grad,
        )

    @property
    def placements(self):
        """Return the tuple of Placement objects."""
        return self._spec.placements

    @property
    def device_mesh(self):
        """Return the DeviceMesh this tensor is distributed over."""
        return self._spec.mesh

    @staticmethod
    def from_local(local_tensor, device_mesh, placements):
        """Construct a DTensor from a local shard."""
        global_shape = _compute_global_shape(
            local_tensor.shape, device_mesh, placements
        )
        global_stride = _compute_global_stride(
            local_tensor.stride, device_mesh, placements
        )
        tensor_meta = TensorMeta(
            shape=global_shape,
            stride=global_stride,
            dtype=local_tensor.dtype,
        )
        spec = DTensorSpec(device_mesh, placements, tensor_meta)
        return DTensor(local_tensor, spec)

    def to_local(self):
        """Extract the local tensor shard."""
        return self._local_tensor

    def redistribute(self, device_mesh, placements):
        """Redistribute this DTensor to a new placement on *device_mesh*.

        Supported transitions (single-mesh-dim, NPU/HCCL path):

        * Shard(d)  -> Replicate  : all_gather_into_tensor
        * Replicate -> Shard(d)   : reduce_scatter_tensor (equal split)
        * X         -> X          : no-op copy

        Args:
            device_mesh: target DeviceMesh (must match self.device_mesh for now).
            placements:  sequence of Placement objects for the new distribution.

        Returns:
            A new DTensor with the requested placement.
        """
        from .placement import Shard, Replicate
        from .. import is_initialized
        import candle.distributed as dist

        placements = list(placements)
        current = list(self.placements)

        # No-op when placement is identical
        if placements == current:
            new_spec = DTensorSpec(
                device_mesh, placements,
                TensorMeta(self._spec.tensor_meta.shape,
                           self._spec.tensor_meta.stride,
                           self._spec.tensor_meta.dtype),
            )
            return DTensor(self._local_tensor, new_spec)

        # Only 1-D mesh supported for now
        if len(placements) != 1 or len(current) != 1:
            raise NotImplementedError(
                "DTensor.redistribute currently supports 1-D mesh only."
            )

        src_p = current[0]
        dst_p = placements[0]
        global_shape = self._spec.tensor_meta.shape
        mesh_size = device_mesh.size(0)

        def _has_pg():
            return bool(device_mesh._dim_groups)

        if isinstance(src_p, Shard) and isinstance(dst_p, Replicate):
            # all_gather: collect shards from all ranks into a full tensor
            shard_dim = src_p.dim
            if _DTENSOR_FASTPATH_ACTIVE:
                full_sizes = _cy_gather_sizes(
                    tuple(self._local_tensor.shape), shard_dim, mesh_size
                )
            else:
                _fs = list(self._local_tensor.shape)
                _fs[shard_dim] = _fs[shard_dim] * mesh_size
                full_sizes = tuple(_fs)
            import candle
            full = candle.empty(full_sizes, dtype=self._local_tensor.dtype)
            if is_initialized() and _has_pg():
                dist.all_gather_into_tensor(
                    full, self._local_tensor, group=device_mesh.get_group(0)
                )
            else:
                # Single-rank / no-PG path (unit tests)
                import numpy as np
                import candle as _candle
                local_np = self._local_tensor.numpy()
                tiled = np.tile(
                    local_np,
                    (mesh_size,) + (1,) * (local_np.ndim - 1)
                )
                full = _candle.tensor(tiled)
            new_spec = DTensorSpec(
                device_mesh, placements,
                TensorMeta(global_shape,
                           self._spec.tensor_meta.stride,
                           self._spec.tensor_meta.dtype),
            )
            return DTensor(full, new_spec)

        if isinstance(src_p, Replicate) and isinstance(dst_p, Shard):
            # reduce_scatter: scatter equal chunks
            shard_dim = dst_p.dim
            if _DTENSOR_FASTPATH_ACTIVE:
                local_sizes = _cy_scatter_sizes(
                    tuple(self._local_tensor.shape), shard_dim, mesh_size
                )
            else:
                _ls = list(self._local_tensor.shape)
                _ls[shard_dim] = _ls[shard_dim] // mesh_size
                local_sizes = tuple(_ls)
            import candle
            out = candle.empty(local_sizes, dtype=self._local_tensor.dtype)
            if is_initialized() and _has_pg():
                dist.reduce_scatter_tensor(
                    out, self._local_tensor, group=device_mesh.get_group(0)
                )
            else:
                import numpy as np
                import candle as _candle
                local_np = self._local_tensor.numpy()
                # For single-rank, take the first slice along shard_dim
                slices = [slice(None)] * local_np.ndim
                slices[shard_dim] = slice(0, local_sizes[shard_dim])
                out = _candle.tensor(np.ascontiguousarray(local_np[tuple(slices)]))
            new_spec = DTensorSpec(
                device_mesh, placements,
                TensorMeta(global_shape,
                           self._spec.tensor_meta.stride,
                           self._spec.tensor_meta.dtype),
            )
            return DTensor(out, new_spec)

        raise NotImplementedError(
            f"DTensor.redistribute: unsupported transition "
            f"{src_p} -> {dst_p}. "
            "Supported: Shard->Replicate, Replicate->Shard."
        )

    def full_tensor(self):
        """Gather the full (global) tensor on every rank.

        * If already Replicate: returns the local tensor directly (no comm).
        * If Shard(d): calls all_gather_into_tensor and returns the result.

        Returns a plain Tensor (not a DTensor).
        """
        from .placement import Replicate
        if len(self.placements) == 1 and isinstance(self.placements[0], Replicate):
            return self._local_tensor
        # Gather via redistribute then unwrap
        replicated = self.redistribute(self.device_mesh, [Replicate()])
        return replicated._local_tensor

    @property
    def grad(self):
        """Proxy grad access to the local shard so optimizers see the reduced gradient."""
        return self._local_tensor.grad

    @grad.setter
    def grad(self, value):
        """Proxy grad assignment to the local shard."""
        self._local_tensor.grad = value

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # Delegate everything to __torch_dispatch__ via the dispatch system.
        return NotImplemented

    # Optimizer step ops that should operate directly on local shards
    _SHARD_PASSTHROUGH_OPS = frozenset({
        "_sgd_step", "_adam_step", "_adamw_step",
    })

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # Collect DTensorSpecs from all DTensor args
        specs = []

        def _extract(val):
            if isinstance(val, DTensor):
                specs.append(val._spec)
            elif isinstance(val, (list, tuple)):
                for v in val:
                    _extract(v)

        for a in args:
            _extract(a)
        if kwargs:
            for v in kwargs.values():
                _extract(v)

        # Block direct compute on sharded DTensors, except optimizer steps
        if func not in cls._SHARD_PASSTHROUGH_OPS:
            for spec in specs:
                if spec.has_shard_placement():
                    raise RuntimeError(
                        f"{func}: direct compute on sharded DTensor is "
                        f"not supported. Ensure fully_shard() hooks unshard "
                        f"parameters before forward."
                    )

        # For replicated DTensors, unwrap to local tensors and redispatch
        def _unwrap(val):
            if isinstance(val, DTensor):
                return val._local_tensor
            if isinstance(val, (list, tuple)):
                return type(val)(_unwrap(v) for v in val)
            return val

        new_args = _unwrap(args)
        new_kwargs = {k: _unwrap(v) for k, v in (kwargs or {}).items()}
        from ..._dispatch.dispatcher import dispatch
        return dispatch(func, None, *new_args, **new_kwargs)

    # ------------------------------------------------------------------
    # Checkpointable protocol (DCP)
    # ------------------------------------------------------------------

    def __create_write_items__(self, fqn, obj):
        """Checkpointable protocol: create WriteItem for this DTensor's local shard."""
        from ..checkpoint.planner import WriteItem, WriteItemType, TensorWriteData
        from ..checkpoint.metadata import ChunkStorageMetadata, MetadataIndex, TensorProperties
        from .placement import Partial

        if any(isinstance(p, Partial) for p in self.placements):
            raise RuntimeError("Cannot checkpoint DTensor with Partial placement")

        sizes, offsets = compute_local_shape_and_global_offset(
            self._spec.tensor_meta.shape, self.device_mesh, self.placements
        )
        chunk = ChunkStorageMetadata(offsets=offsets, sizes=sizes)
        props = TensorProperties(
            dtype=self._local_tensor.dtype,
            requires_grad=self.requires_grad,
        )
        index = MetadataIndex(fqn, offset=offsets)
        return [WriteItem(
            index=index,
            type=WriteItemType.SHARD,
            tensor_data=TensorWriteData(
                chunk=chunk, properties=props,
                size=self._spec.tensor_meta.shape,
            ),
        )]

    def __create_chunk_list__(self):
        """Checkpointable protocol: describe this rank's local chunk for load planning."""
        from ..checkpoint.metadata import ChunkStorageMetadata

        sizes, offsets = compute_local_shape_and_global_offset(
            self._spec.tensor_meta.shape, self.device_mesh, self.placements
        )
        return [ChunkStorageMetadata(offsets=offsets, sizes=sizes)]

    def __repr__(self):
        return (
            f"DTensor(local_shape={self._local_tensor.shape}, "
            f"placements={self.placements})"
        )


def compute_local_shape_and_global_offset(global_shape, mesh, placements):
    """Compute local shard shape and global offset from distribution spec.

    Delegates to the Cython fastpath when available, otherwise uses the
    pure-Python fallback.

    Args:
        global_shape: the global (unsharded) tensor shape.
        mesh: DeviceMesh instance.
        placements: sequence of Placement objects.

    Returns:
        (local_shape, global_offset) as tuples of ints.
    """
    if _DTENSOR_FASTPATH_ACTIVE:
        return _cy_local_shape_offset(tuple(global_shape), mesh, placements)
    return _compute_local_shape_and_global_offset_py(global_shape, mesh, placements)


def _compute_local_shape_and_global_offset_py(global_shape, mesh, placements):
    """Pure-Python fallback for compute_local_shape_and_global_offset."""
    from .placement import Shard
    local_shape = list(global_shape)
    global_offset = [0] * len(global_shape)
    for mesh_dim, placement in enumerate(placements):
        if isinstance(placement, Shard):
            dim = placement.dim
            num_chunks = mesh.size(mesh_dim)
            local_rank = mesh.get_local_rank(mesh_dim)
            chunk_size = global_shape[dim] // num_chunks
            remainder = global_shape[dim] % num_chunks
            if local_rank < remainder:
                local_shape[dim] = chunk_size + 1
                global_offset[dim] = local_rank * (chunk_size + 1)
            else:
                local_shape[dim] = chunk_size
                global_offset[dim] = (
                    remainder * (chunk_size + 1)
                    + (local_rank - remainder) * chunk_size
                )
    return tuple(local_shape), tuple(global_offset)


def _compute_global_shape(local_shape, mesh, placements):
    """Compute the global (unsharded) shape from a local shard shape."""
    if _DTENSOR_FASTPATH_ACTIVE:
        return _cy_global_shape(tuple(local_shape), mesh, placements)
    return _compute_global_shape_py(local_shape, mesh, placements)


def _compute_global_shape_py(local_shape, mesh, placements):
    """Pure-Python fallback for _compute_global_shape."""
    from .placement import Shard
    global_shape = list(local_shape)
    for placement in placements:
        if isinstance(placement, Shard):
            global_shape[placement.dim] *= mesh.size()
    return tuple(global_shape)


def _compute_global_stride(local_stride, mesh, placements):  # pylint: disable=unused-argument
    """Compute the global stride (identity for MVP)."""
    if _DTENSOR_FASTPATH_ACTIVE:
        return _cy_global_stride(local_stride)
    return (
        tuple(local_stride)
        if not isinstance(local_stride, tuple)
        else local_stride
    )
