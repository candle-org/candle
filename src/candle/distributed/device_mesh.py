"""DeviceMesh — multi-dimensional device topology abstraction.

MVP: 1D mesh only (pure FSDP). API aligned with torch.distributed.device_mesh.
"""


class DeviceMesh:
    """Logical arrangement of devices for distributed training.

    Usage:
        # After dist.init_process_group()
        mesh = DeviceMesh("npu", (world_size,), mesh_dim_names=("shard",))
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
        """Create ProcessGroups per mesh dimension.

        1D MVP: reuse the global WORLD process group.
        """
        from . import is_initialized
        if not is_initialized():
            return
        if len(self._mesh_shape) == 1:
            from . import _get_default_group
            self._dim_groups = [_get_default_group()]
        else:
            raise NotImplementedError(
                f"DeviceMesh only supports 1D mesh in MVP, got shape {self._mesh_shape}"
            )

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
