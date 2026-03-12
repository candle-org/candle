"""DTensor -- Distributed Tensor for FSDP2.

A lightweight metadata container wrapping a local tensor shard with
placement and mesh information.  FSDP manages communication manually;
DTensor does NOT perform automatic redistribution in MVP.
"""
from ..._tensor import Tensor


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

    def __repr__(self):
        return (
            f"DTensor(local_shape={self._local_tensor.shape}, "
            f"placements={self.placements})"
        )


def _compute_global_shape(local_shape, mesh, placements):
    """Compute the global (unsharded) shape from a local shard shape."""
    from .placement import Shard
    global_shape = list(local_shape)
    for placement in placements:
        if isinstance(placement, Shard):
            global_shape[placement.dim] *= mesh.size()
    return tuple(global_shape)


def _compute_global_stride(local_stride, mesh, placements):
    """Compute the global stride (identity for MVP)."""
    return (
        tuple(local_stride)
        if not isinstance(local_stride, tuple)
        else local_stride
    )
