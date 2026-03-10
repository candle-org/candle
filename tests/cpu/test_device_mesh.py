"""Tests for DeviceMesh (1D MVP, no distributed init required for basic tests)."""
from candle.distributed.device_mesh import DeviceMesh


def test_device_mesh_basic():
    """DeviceMesh stores device type and shape."""
    mesh = DeviceMesh.__new__(DeviceMesh)
    mesh.device_type = "cpu"
    mesh.mesh_dim_names = ("shard",)
    mesh._mesh_shape = (4,)
    assert mesh.device_type == "cpu"
    assert mesh.mesh_dim_names == ("shard",)


def test_device_mesh_ndim():
    """1D mesh should have ndim=1."""
    mesh = DeviceMesh.__new__(DeviceMesh)
    mesh._mesh_shape = (4,)
    assert mesh.ndim == 1


def test_device_mesh_size():
    """size() should return the mesh dimension size."""
    mesh = DeviceMesh.__new__(DeviceMesh)
    mesh._mesh_shape = (8,)
    assert mesh.size(0) == 8
