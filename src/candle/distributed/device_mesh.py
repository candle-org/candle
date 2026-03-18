"""DeviceMesh — multi-dimensional device topology abstraction.

Supports 1D mesh (pure FSDP) and 2D mesh (HSDP: replicate + shard).
API aligned with torch.distributed.device_mesh.
"""


class DeviceMesh:
    """Logical arrangement of devices for distributed training.

    Usage:
        # 1D: pure FSDP
        mesh = DeviceMesh("npu", (world_size,), mesh_dim_names=("shard",))

        # 2D: HSDP (replicate across nodes, shard within node)
        mesh = DeviceMesh("npu", (num_replicas, shard_size),
                          mesh_dim_names=("replicate", "shard"))
    """

    def __init__(self, device_type, mesh_shape, *, mesh_dim_names=None):
        if isinstance(mesh_shape, int):
            mesh_shape = (mesh_shape,)
        self.device_type = device_type
        self._mesh_shape = tuple(mesh_shape)
        self.mesh_dim_names = mesh_dim_names
        self._dim_groups = []
        self._init_process_groups()

    def _init_process_groups(self):
        """Create ProcessGroups per mesh dimension."""
        from . import is_initialized
        if not is_initialized():
            return
        if len(self._mesh_shape) == 1:
            from . import _get_default_group
            self._dim_groups = [_get_default_group()]
        elif len(self._mesh_shape) == 2:
            self._init_2d_process_groups()
        else:
            raise NotImplementedError(
                f"DeviceMesh supports up to 2D mesh, got shape {self._mesh_shape}"
            )

    def _init_2d_process_groups(self):
        """Create sub-process-groups for a 2D mesh.

        For mesh shape ``(num_replicas, shard_size)``:
        - dim 0 (replicate): groups of ranks at the same position within
          each shard group (e.g., ranks [0, 2] and [1, 3] for 2x2).
        - dim 1 (shard): groups of contiguous ranks within each replica
          (e.g., ranks [0, 1] and [2, 3] for 2x2).
        """
        from . import new_group, get_rank
        num_replicas, shard_size = self._mesh_shape
        world_size = num_replicas * shard_size
        my_rank = get_rank()

        # dim 0: replicate groups — ranks that share the same intra-shard position
        # For rank r, its shard-local position is r % shard_size
        # Its replicate group contains ranks at that position across all replicas
        my_shard_pos = my_rank % shard_size
        replicate_ranks = [
            replica * shard_size + my_shard_pos
            for replica in range(num_replicas)
        ]
        replicate_pg = new_group(replicate_ranks)

        # dim 1: shard groups — contiguous ranks within each replica
        my_replica = my_rank // shard_size
        shard_ranks = [
            my_replica * shard_size + pos
            for pos in range(shard_size)
        ]
        shard_pg = new_group(shard_ranks)

        self._dim_groups = [replicate_pg, shard_pg]

    def get_group(self, mesh_dim=0):
        """Get the ProcessGroup for a mesh dimension."""
        if not self._dim_groups:
            raise RuntimeError(
                "DeviceMesh process groups not initialized. "
                "Call dist.init_process_group() first."
            )
        return self._dim_groups[mesh_dim]

    def size(self, mesh_dim=0):
        """Number of devices along a mesh dimension."""
        return self._mesh_shape[mesh_dim]

    def get_local_rank(self, mesh_dim=0):
        """Return this rank's position along *mesh_dim*."""
        from . import get_rank
        return get_rank(self.get_group(mesh_dim))

    @property
    def ndim(self):
        """Number of mesh dimensions."""
        return len(self._mesh_shape)

    def __repr__(self):
        return (
            f"DeviceMesh(device_type={self.device_type!r}, "
            f"mesh_shape={self._mesh_shape}, "
            f"mesh_dim_names={self.mesh_dim_names})"
        )


def init_device_mesh(device_type, mesh_shape, *, mesh_dim_names=None):
    """Create a DeviceMesh. Convenience function matching PyTorch API."""
    return DeviceMesh(device_type, mesh_shape, mesh_dim_names=mesh_dim_names)
