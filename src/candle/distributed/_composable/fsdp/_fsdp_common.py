"""Common utilities for FSDP2."""


class FSDPMeshInfo:
    """Mesh information for FSDP parameter groups."""

    def __init__(self, mesh):
        self.mesh = mesh
        self.shard_mesh_size = mesh.size(0)
        pg = mesh.get_group(0)
        self.shard_process_group = pg
        self.shard_mesh_rank = pg.rank() if pg is not None else 0
